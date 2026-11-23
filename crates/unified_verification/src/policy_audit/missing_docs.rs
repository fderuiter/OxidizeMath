use super::lint_policy::{
    attr_suppresses_missing_docs, is_vendored_egui_plot_exempt, meta_suppresses_missing_docs,
};
use crate::whitelist::{PathMatchOptions, VerificationWhitelist, path_matches_lossy};
use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::collections::HashSet;
use std::path::Path;
use syn::visit::Visit;

const PATH_MATCH_OPTS: PathMatchOptions = PathMatchOptions {
    trim_dot_slash: true,
    require_non_empty: true,
    match_file_name: false,
};

pub(crate) fn is_vendored_missing_docs_exempt(
    member: &str,
    parsed: &toml::Value,
    whitelist: &VerificationWhitelist,
) -> bool {
    if !is_vendored_egui_plot_exempt(member, parsed) {
        return false;
    }
    let Some(cfg) = &whitelist.missing_docs else {
        return false;
    };
    let member_path = Path::new(member.trim_end_matches('/'));
    cfg.vendored_crates
        .iter()
        .any(|vendored| path_matches_lossy(member_path, vendored, PATH_MATCH_OPTS))
}

fn tokens_contain_missing_docs(tokens: &[TokenTree]) -> bool {
    for idx in 0..tokens.len() {
        if let TokenTree::Ident(id) = &tokens[idx]
            && id == "missing_docs"
        {
            if idx == 0 {
                return true;
            }
            if idx >= 3
                && let (TokenTree::Ident(ns), TokenTree::Punct(p1), TokenTree::Punct(p2)) =
                    (&tokens[idx - 3], &tokens[idx - 2], &tokens[idx - 1])
                && ns == "rust"
                && p1.as_char() == ':'
                && p2.as_char() == ':'
            {
                return true;
            }
            if let TokenTree::Punct(prev) = &tokens[idx - 1]
                && prev.as_char() != ':'
            {
                return true;
            }
        }
    }
    false
}

fn attr_tokens_suppress_missing_docs(attr_stream: TokenStream) -> bool {
    if let Ok(meta) = syn::parse2::<syn::Meta>(attr_stream.clone())
        && meta_suppresses_missing_docs(&meta)
    {
        return true;
    }
    let mut groups = vec![attr_stream];
    while let Some(stream) = groups.pop() {
        let toks: Vec<TokenTree> = stream.into_iter().collect();
        for pair in toks.windows(2) {
            if let (TokenTree::Ident(id), TokenTree::Group(g)) = (&pair[0], &pair[1]) {
                if (id == "allow" || id == "expect") && g.delimiter() == Delimiter::Parenthesis {
                    let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                    if tokens_contain_missing_docs(&inner) {
                        return true;
                    }
                } else if id == "cfg_attr" {
                    groups.push(g.stream());
                }
            }
        }
    }
    false
}

pub(crate) struct MissingDocsVisitor {
    pub(crate) file_path: String,
    pub(crate) has_inner_bypass: bool,
    pub(crate) outer_count: usize,
}

impl MissingDocsVisitor {
    pub(crate) fn new(file_path: String) -> Self {
        Self {
            file_path,
            has_inner_bypass: false,
            outer_count: 0,
        }
    }

    fn report_inner_violation(&mut self, scope: &str) {
        println!(
            "[!] Unauthorized inner missing_docs bypass in {} ({scope})",
            self.file_path
        );
        self.has_inner_bypass = true;
    }

    fn scan_macro_tokens(&mut self, root_stream: TokenStream, macro_label: &str) {
        let mut streams = vec![root_stream];
        while let Some(stream) = streams.pop() {
            let tokens: Vec<TokenTree> = stream.into_iter().collect();
            for i in 0..tokens.len() {
                if let TokenTree::Group(group) = &tokens[i] {
                    streams.push(group.stream());
                }
                let TokenTree::Punct(hash) = &tokens[i] else {
                    continue;
                };
                if hash.as_char() != '#' {
                    continue;
                }
                if i + 2 < tokens.len()
                    && matches!(&tokens[i + 1], TokenTree::Punct(p) if p.as_char() == '!')
                    && let TokenTree::Group(bracket) = &tokens[i + 2]
                    && bracket.delimiter() == Delimiter::Bracket
                {
                    if attr_tokens_suppress_missing_docs(bracket.stream()) {
                        self.report_inner_violation(macro_label);
                    }
                } else if i + 1 < tokens.len()
                    && let TokenTree::Group(bracket) = &tokens[i + 1]
                    && bracket.delimiter() == Delimiter::Bracket
                    && attr_tokens_suppress_missing_docs(bracket.stream())
                {
                    self.outer_count += 1;
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for MissingDocsVisitor {
    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        if i.mac.path.is_ident("macro_rules") {
            let name = i
                .ident
                .as_ref()
                .map_or_else(|| "<anon>".to_string(), ToString::to_string);
            let label = format!("macro_rules! `{name}`");
            self.scan_macro_tokens(i.mac.tokens.clone(), &label);
        }
        syn::visit::visit_item_macro(self, i);
    }

    fn visit_impl_item_macro(&mut self, i: &'ast syn::ImplItemMacro) {
        if i.mac.path.is_ident("macro_rules") {
            self.scan_macro_tokens(i.mac.tokens.clone(), "impl macro_rules!");
        }
        syn::visit::visit_impl_item_macro(self, i);
    }

    fn visit_trait_item_macro(&mut self, i: &'ast syn::TraitItemMacro) {
        if i.mac.path.is_ident("macro_rules") {
            self.scan_macro_tokens(i.mac.tokens.clone(), "trait macro_rules!");
        }
        syn::visit::visit_trait_item_macro(self, i);
    }

    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        if attr_suppresses_missing_docs(i) {
            match i.style {
                syn::AttrStyle::Inner(_) => {
                    self.report_inner_violation("inner `#![...]` attribute");
                }
                syn::AttrStyle::Outer => {
                    self.outer_count += 1;
                }
            }
        }
        syn::visit::visit_attribute(self, i);
    }
}

fn lookup_file_cap<'a>(
    file_path: &str,
    whitelist: &'a VerificationWhitelist,
) -> Option<(&'a str, usize)> {
    let cfg = whitelist.missing_docs.as_ref()?;
    let actual = Path::new(file_path);
    cfg.files
        .iter()
        .find(|(key, _)| path_matches_lossy(actual, key, PATH_MATCH_OPTS))
        .map(|(key, &cap)| (key.as_str(), cap))
}

pub(crate) fn audit_file_missing_docs(
    ast: &syn::File,
    file_path: &str,
    whitelist: &VerificationWhitelist,
    matched_keys: &mut HashSet<String>,
) -> bool {
    let mut visitor = MissingDocsVisitor::new(file_path.to_string());
    visitor.visit_file(ast);

    let matched = lookup_file_cap(file_path, whitelist);
    if let Some((key, _)) = matched {
        matched_keys.insert(key.to_string());
    }
    let cap = matched.map_or(0, |(_, c)| c);
    let mut ok = !visitor.has_inner_bypass;

    if visitor.outer_count > cap {
        println!(
            "[!] Unauthorized missing_docs bypass in {}: found {} #[allow(missing_docs)]/#[expect(missing_docs)] attribute(s), whitelisted cap is {}",
            file_path, visitor.outer_count, cap
        );
        ok = false;
    } else if visitor.outer_count < cap {
        println!(
            "[!] Unratcheted missing_docs cap in {}: found {} #[allow(missing_docs)] attribute(s), but verification_whitelist.toml allows {}; lower or remove the cap",
            file_path, visitor.outer_count, cap
        );
        ok = false;
    }

    ok
}

fn crate_prefix_of_whitelist_key(key: &str) -> Option<String> {
    let norm = key.trim().replace('\\', "/");
    let trimmed = norm.trim_start_matches("./");
    let segs: Vec<&str> = trimmed.split('/').filter(|s| !s.is_empty()).collect();
    if segs.len() < 2 {
        return None;
    }
    if (segs[0] == "crates" || segs[0] == "apps") && segs.len() >= 3 {
        Some(format!("{}/{}", segs[0], segs[1]))
    } else if segs[0] != "crates" && segs[0] != "apps" {
        Some(segs[0].to_string())
    } else {
        None
    }
}

pub(crate) fn check_stale_member_missing_docs_caps(
    member_dir: &Path,
    whitelist: &VerificationWhitelist,
    matched_keys: &HashSet<String>,
) -> bool {
    let Some(cfg) = &whitelist.missing_docs else {
        return true;
    };
    let mut ok = true;
    for (key, &cap) in &cfg.files {
        let Some(crate_prefix) = crate_prefix_of_whitelist_key(key) else {
            continue;
        };
        if path_matches_lossy(member_dir, &crate_prefix, PATH_MATCH_OPTS)
            && !matched_keys.contains(key)
        {
            println!(
                "[!] Stale missing_docs cap in verification_whitelist.toml for missing file {} (cap = {}); remove the entry",
                key, cap
            );
            ok = false;
        }
    }
    ok
}
