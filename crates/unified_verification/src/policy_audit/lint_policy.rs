use proc_macro2::{Delimiter, TokenStream, TokenTree};
use std::fs;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

pub(crate) const DEFAULT_TOO_MANY_LINES_THRESHOLD: usize = 60;

const BLOCKED_CLIPPY_GROUPS: &[&str] = &[
    "all",
    "pedantic",
    "nursery",
    "restriction",
    "complexity",
    "correctness",
    "suspicious",
    "style",
    "perf",
    "cargo",
];

const INNER_BLOCKED_LINTS: &[&str] = &[
    "all",
    "pedantic",
    "nursery",
    "restriction",
    "complexity",
    "correctness",
    "suspicious",
    "style",
    "perf",
    "cargo",
    "too_many_lines",
    "cognitive_complexity",
];

const SCOPE_EXTRA_BLOCKED_LINTS: &[&str] = &["too_many_lines", "cognitive_complexity"];
const TOO_MANY_LINES_ONLY: &[&str] = &["too_many_lines"];
const MAX_ATTR_STACK_STEPS: usize = 256;

pub(crate) fn load_too_many_lines_threshold() -> usize {
    let candidates = [".clippy.toml", "../../.clippy.toml"];
    for candidate in candidates {
        let Ok(content) = fs::read_to_string(candidate) else {
            continue;
        };
        let Ok(parsed) = toml::from_str::<toml::Value>(&content) else {
            continue;
        };
        if let Some(limit) = parsed
            .get("too-many-lines-threshold")
            .and_then(toml::Value::as_integer)
            .and_then(|v| usize::try_from(v).ok())
        {
            return limit;
        }
    }
    DEFAULT_TOO_MANY_LINES_THRESHOLD
}

pub(crate) fn cwd_is_workspace_root() -> bool {
    fs::read_to_string("Cargo.toml")
        .map(|c| c.contains("[workspace]"))
        .unwrap_or(false)
}

pub(crate) fn cwd_is_member_crate_dir() -> bool {
    !cwd_is_workspace_root()
        && Path::new("Cargo.toml").exists()
        && fs::read_to_string("../../Cargo.toml")
            .map(|c| c.contains("[workspace]"))
            .unwrap_or(false)
}

pub(crate) fn resolve_member_dir(member: &str) -> PathBuf {
    let direct = PathBuf::from(member);
    if direct.join("Cargo.toml").exists() || direct.is_absolute() || !cwd_is_member_crate_dir() {
        return direct;
    }
    let candidate = Path::new("../..").join(member);
    if candidate.join("Cargo.toml").exists() {
        candidate
    } else {
        direct
    }
}

pub(crate) fn is_vendored_egui_plot(member: &str) -> bool {
    let norm = member.replace('\\', "/");
    let trimmed = norm.trim_end_matches('/');
    trimmed == "egui_plot" || trimmed.ends_with("/egui_plot")
}

pub(crate) fn has_explicit_egui_plot_lints(parsed: &toml::Value) -> bool {
    let Some(lints) = parsed.get("lints").and_then(toml::Value::as_table) else {
        return false;
    };
    if lints.contains_key("workspace") {
        return false;
    }
    let has_rust = lints
        .get("rust")
        .and_then(toml::Value::as_table)
        .is_some_and(|t| !t.is_empty());
    let has_clippy = lints
        .get("clippy")
        .and_then(toml::Value::as_table)
        .is_some_and(|t| !t.is_empty());
    has_rust && has_clippy
}

pub(crate) fn is_vendored_egui_plot_exempt(member: &str, parsed: &toml::Value) -> bool {
    is_vendored_egui_plot(member) && has_explicit_egui_plot_lints(parsed)
}

pub(crate) fn has_workspace_lints_inheritance(parsed: &toml::Value) -> bool {
    parsed
        .get("lints")
        .and_then(toml::Value::as_table)
        .is_some_and(|t| {
            t.len() == 1 && t.get("workspace").and_then(toml::Value::as_bool) == Some(true)
        })
}

pub(crate) fn check_manifest_lints(
    member: &str,
    manifest_path: &Path,
    parsed: &toml::Value,
) -> bool {
    if is_vendored_egui_plot(member) {
        if has_explicit_egui_plot_lints(parsed) || has_workspace_lints_inheritance(parsed) {
            return true;
        }
        eprintln!(
            "[!] Lint policy mismatch in {}: vendored egui_plot must define explicit `[lints.rust]` and `[lints.clippy]` tables (or `[lints] workspace = true`)",
            manifest_path.display()
        );
        return false;
    }

    if has_workspace_lints_inheritance(parsed) {
        return true;
    }

    eprintln!(
        "[!] Lint policy mismatch in {}: first-party crates must inherit workspace lints via `[lints] workspace = true`",
        manifest_path.display()
    );
    false
}

fn matched_clippy_lint(path: &syn::Path, names: &[&'static str]) -> Option<&'static str> {
    if path.segments.len() != 2
        || path.segments.iter().any(|seg| !seg.arguments.is_empty())
        || path.segments[0].ident != "clippy"
    {
        return None;
    }
    let target = &path.segments[1].ident;
    names.iter().copied().find(|&name| target == name)
}

pub(crate) fn find_suppressed_clippy_lint_in_meta(
    root_meta: &syn::Meta,
    names: &[&'static str],
) -> Option<&'static str> {
    let mut stack = vec![root_meta.clone()];
    let mut steps = 0usize;
    while let Some(meta) = stack.pop() {
        if steps >= MAX_ATTR_STACK_STEPS {
            break;
        }
        steps += 1;
        let syn::Meta::List(list) = meta else {
            continue;
        };
        if list.path.is_ident("allow") || list.path.is_ident("expect") {
            let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                continue;
            };
            for item in nested {
                if let syn::Meta::Path(lint_path) = item
                    && let Some(found) = matched_clippy_lint(&lint_path, names)
                {
                    return Some(found);
                }
            }
        } else if list.path.is_ident("cfg_attr") {
            let Ok(nested) = list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
            ) else {
                continue;
            };
            for item in nested.into_iter().skip(1) {
                stack.push(item);
            }
        }
    }
    None
}

pub(crate) fn find_suppressed_clippy_lint(
    attr: &syn::Attribute,
    names: &[&'static str],
) -> Option<&'static str> {
    find_suppressed_clippy_lint_in_meta(&attr.meta, names)
}

fn fn_line_count(sig: &syn::Signature, block: Option<&syn::Block>) -> usize {
    let start = sig.fn_token.span.start().line;
    let Some(body) = block else {
        return 1;
    };
    let end = body.brace_token.span.close().end().line;
    end.saturating_sub(start) + 1
}

fn find_clippy_lint_in_tokens(
    tokens: &[TokenTree],
    names: &[&'static str],
) -> Option<&'static str> {
    for window in tokens.windows(4) {
        if let (
            TokenTree::Ident(ns),
            TokenTree::Punct(p1),
            TokenTree::Punct(p2),
            TokenTree::Ident(lint),
        ) = (&window[0], &window[1], &window[2], &window[3])
            && ns == "clippy"
            && p1.as_char() == ':'
            && p2.as_char() == ':'
            && let Some(&matched) = names.iter().find(|&&n| lint == n)
        {
            return Some(matched);
        }
    }
    None
}

fn find_suppressed_in_attr_tokens(
    attr_stream: TokenStream,
    names: &[&'static str],
) -> Option<&'static str> {
    if let Ok(meta) = syn::parse2::<syn::Meta>(attr_stream.clone())
        && let Some(found) = find_suppressed_clippy_lint_in_meta(&meta, names)
    {
        return Some(found);
    }
    let mut groups = vec![attr_stream];
    while let Some(stream) = groups.pop() {
        let toks: Vec<TokenTree> = stream.into_iter().collect();
        for pair in toks.windows(2) {
            if let (TokenTree::Ident(id), TokenTree::Group(g)) = (&pair[0], &pair[1]) {
                if (id == "allow" || id == "expect") && g.delimiter() == Delimiter::Parenthesis {
                    let inner: Vec<TokenTree> = g.stream().into_iter().collect();
                    if let Some(found) = find_clippy_lint_in_tokens(&inner, names) {
                        return Some(found);
                    }
                } else if id == "cfg_attr" {
                    groups.push(g.stream());
                }
            }
        }
    }
    None
}

fn next_macro_item_keyword(tokens: &[TokenTree], mut idx: usize) -> Option<String> {
    while idx < tokens.len() {
        if let TokenTree::Punct(p) = &tokens[idx]
            && p.as_char() == '#'
        {
            idx += 1;
            if idx < tokens.len()
                && let TokenTree::Punct(bang) = &tokens[idx]
                && bang.as_char() == '!'
            {
                idx += 1;
            }
            if idx < tokens.len()
                && matches!(&tokens[idx], TokenTree::Group(g) if g.delimiter() == Delimiter::Bracket)
            {
                idx += 1;
                continue;
            }
        }
        match &tokens[idx] {
            TokenTree::Ident(id)
                if id == "pub"
                    || id == "unsafe"
                    || id == "async"
                    || id == "const"
                    || id == "default" =>
            {
                idx += 1;
            }
            TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis => idx += 1,
            TokenTree::Ident(id) => return Some(id.to_string()),
            _ => return None,
        }
    }
    None
}

pub(crate) struct ClippyBypassVisitor {
    pub(crate) file_path: String,
    pub(crate) max_fn_lines: usize,
    pub(crate) has_unauthorized: bool,
}

impl ClippyBypassVisitor {
    pub(crate) fn new(file_path: String, max_fn_lines: usize) -> Self {
        Self {
            file_path,
            max_fn_lines,
            has_unauthorized: false,
        }
    }

    fn report_violation(&mut self, scope: &str, lint_name: &str) {
        println!(
            "[!] Unauthorized Clippy bypass in {} ({scope}: clippy::{lint_name})",
            self.file_path
        );
        self.has_unauthorized = true;
    }

    fn check_outer_scope_attrs(&mut self, attrs: &[syn::Attribute], scope: &str) {
        for attr in attrs {
            if matches!(attr.style, syn::AttrStyle::Outer)
                && let Some(lint) = find_suppressed_clippy_lint(attr, SCOPE_EXTRA_BLOCKED_LINTS)
            {
                self.report_violation(scope, lint);
            }
        }
    }

    fn check_fn_stale_too_many_lines(
        &mut self,
        attrs: &[syn::Attribute],
        sig: &syn::Signature,
        block: Option<&syn::Block>,
    ) {
        let lines = fn_line_count(sig, block);
        if lines > self.max_fn_lines {
            return;
        }
        for attr in attrs {
            if matches!(attr.style, syn::AttrStyle::Outer)
                && let Some(lint) = find_suppressed_clippy_lint(attr, TOO_MANY_LINES_ONLY)
            {
                let label = format!(
                    "stale fn `{}` ({lines} <= {} lines) attribute",
                    sig.ident, self.max_fn_lines
                );
                self.report_violation(&label, lint);
            }
        }
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
                    if let Some(lint) =
                        find_suppressed_in_attr_tokens(bracket.stream(), INNER_BLOCKED_LINTS)
                    {
                        self.report_violation(macro_label, lint);
                    }
                } else if i + 1 < tokens.len()
                    && let TokenTree::Group(bracket) = &tokens[i + 1]
                    && bracket.delimiter() == Delimiter::Bracket
                {
                    let target_kw = next_macro_item_keyword(&tokens, i + 2);
                    let blocked = if matches!(target_kw.as_deref(), Some("mod" | "impl" | "trait"))
                    {
                        INNER_BLOCKED_LINTS
                    } else {
                        BLOCKED_CLIPPY_GROUPS
                    };
                    if let Some(lint) = find_suppressed_in_attr_tokens(bracket.stream(), blocked) {
                        self.report_violation(macro_label, lint);
                    }
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for ClippyBypassVisitor {
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        let mod_label = format!("mod `{}` attribute", i.ident);
        self.check_outer_scope_attrs(&i.attrs, &mod_label);
        syn::visit::visit_item_mod(self, i);
    }

    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        self.check_outer_scope_attrs(&i.attrs, "impl block attribute");
        syn::visit::visit_item_impl(self, i);
    }

    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        let trait_label = format!("trait `{}` attribute", i.ident);
        self.check_outer_scope_attrs(&i.attrs, &trait_label);
        syn::visit::visit_item_trait(self, i);
    }

    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.check_fn_stale_too_many_lines(&i.attrs, &i.sig, Some(&i.block));
        syn::visit::visit_item_fn(self, i);
    }

    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        self.check_fn_stale_too_many_lines(&i.attrs, &i.sig, Some(&i.block));
        syn::visit::visit_impl_item_fn(self, i);
    }

    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        self.check_fn_stale_too_many_lines(&i.attrs, &i.sig, i.default.as_ref());
        syn::visit::visit_trait_item_fn(self, i);
    }

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

    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        match i.style {
            syn::AttrStyle::Inner(_) => {
                if let Some(lint) = find_suppressed_clippy_lint(i, INNER_BLOCKED_LINTS) {
                    self.report_violation("inner `#![...]` attribute", lint);
                }
            }
            syn::AttrStyle::Outer => {
                if let Some(lint) = find_suppressed_clippy_lint(i, BLOCKED_CLIPPY_GROUPS) {
                    self.report_violation("item `#[...]` attribute", lint);
                }
            }
        }
        syn::visit::visit_attribute(self, i);
    }
}
