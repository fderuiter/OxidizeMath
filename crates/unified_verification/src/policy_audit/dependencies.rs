use crate::whitelist::{PathMatchOptions, UnusedDependencyEntry};
use proc_macro2::{TokenStream, TokenTree};
use std::collections::HashSet;
use std::path::Path;
use syn::visit::Visit;

const DEP_SECTIONS: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];
const MAX_META_STACK_STEPS: usize = 256;

fn extend_deps_from_table(parent: &toml::map::Map<String, toml::Value>, out: &mut HashSet<String>) {
    for section in DEP_SECTIONS {
        if let Some(table) = parent.get(*section).and_then(toml::Value::as_table) {
            for key in table.keys() {
                out.insert(key.replace('-', "_"));
            }
        }
    }
}

pub(crate) fn collect_expected_dependencies(parsed: &toml::Value) -> HashSet<String> {
    let mut expected_deps = HashSet::new();
    let Some(root_table) = parsed.as_table() else {
        return expected_deps;
    };

    extend_deps_from_table(root_table, &mut expected_deps);

    if let Some(targets) = root_table.get("target").and_then(toml::Value::as_table) {
        for target_cfg in targets.values() {
            if let Some(target_table) = target_cfg.as_table() {
                extend_deps_from_table(target_table, &mut expected_deps);
            }
        }
    }

    expected_deps
}

pub(crate) fn manifest_package_name(parsed: &toml::Value) -> Option<&str> {
    parsed
        .get("package")
        .and_then(|pkg| pkg.get("name"))
        .and_then(toml::Value::as_str)
}

pub(crate) fn is_whitelisted_unused_dep(
    member: &str,
    package_name: Option<&str>,
    expected_dep: &str,
    entries: &[UnusedDependencyEntry],
) -> bool {
    let opts = PathMatchOptions {
        trim_dot_slash: true,
        require_non_empty: true,
        match_file_name: false,
    };
    let member_path = Path::new(member);
    let norm_pkg = package_name.map(|n| n.replace('-', "_"));

    entries.iter().any(|entry| {
        let dep_norm = entry.dependency.replace('-', "_");
        if dep_norm != expected_dep {
            return false;
        }
        let crate_matches_path =
            crate::whitelist::path_matches_lossy(member_path, &entry.krate, opts);
        let entry_pkg_norm = entry.krate.replace('-', "_");
        let crate_matches_pkg =
            !entry.krate.contains('/') && norm_pkg.as_deref() == Some(entry_pkg_norm.as_str());
        crate_matches_path || crate_matches_pkg
    })
}

fn derive_crate_for_trait(trait_ident: &syn::Ident) -> Option<&'static str> {
    if trait_ident == "Serialize" || trait_ident == "Deserialize" {
        Some("serde")
    } else if trait_ident == "Error" {
        Some("thiserror")
    } else {
        None
    }
}

fn is_compiler_lint_or_cfg_attr(path: &syn::Path) -> bool {
    path.is_ident("cfg")
        || path.is_ident("allow")
        || path.is_ident("expect")
        || path.is_ident("warn")
        || path.is_ident("deny")
        || path.is_ident("forbid")
}

pub(crate) struct DependencyVisitor {
    pub(crate) expected_deps: HashSet<String>,
    pub(crate) used_deps: HashSet<String>,
}

impl DependencyVisitor {
    pub(crate) fn new(expected_deps: HashSet<String>) -> Self {
        Self {
            expected_deps,
            used_deps: HashSet::new(),
        }
    }

    pub(crate) fn check_name(&mut self, name: &str) {
        if self.expected_deps.contains(name) {
            self.used_deps.insert(name.to_string());
        }
    }

    pub(crate) fn check_ident(&mut self, ident: &syn::Ident) {
        self.check_name(&ident.to_string());
    }

    pub(crate) fn scan_token_stream(&mut self, stream: TokenStream) {
        let mut streams = vec![stream];
        while let Some(current) = streams.pop() {
            for token in current {
                match token {
                    TokenTree::Ident(ident) => self.check_ident(&ident),
                    TokenTree::Group(group) => streams.push(group.stream()),
                    _ => {}
                }
            }
        }
    }

    fn inspect_derive_list(&mut self, list: &syn::MetaList) {
        let Ok(paths) = list.parse_args_with(
            syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
        ) else {
            self.scan_token_stream(list.tokens.clone());
            return;
        };
        for path in paths {
            if let Some(first) = path.segments.first() {
                self.check_ident(&first.ident);
            }
            if let Some(last) = path.segments.last()
                && let Some(mapped_crate) = derive_crate_for_trait(&last.ident)
            {
                self.check_name(mapped_crate);
            }
        }
    }

    fn inspect_meta(&mut self, root_meta: &syn::Meta) {
        let mut stack = vec![root_meta.clone()];
        let mut steps = 0usize;
        while let Some(meta) = stack.pop() {
            if steps >= MAX_META_STACK_STEPS {
                break;
            }
            steps += 1;
            if let Some(first) = meta.path().segments.first() {
                self.check_ident(&first.ident);
            }
            match meta {
                syn::Meta::NameValue(nv) => {
                    self.visit_expr(&nv.value);
                }
                syn::Meta::List(list) => {
                    if list.path.is_ident("cfg_attr") {
                        if let Ok(nested) = list.parse_args_with(
                            syn::punctuated::Punctuated::<syn::Meta, syn::Token![,]>::parse_terminated,
                        ) {
                            for item in nested.into_iter().skip(1) {
                                stack.push(item);
                            }
                        }
                    } else if list.path.is_ident("derive") {
                        self.inspect_derive_list(&list);
                    } else if !is_compiler_lint_or_cfg_attr(&list.path) {
                        self.scan_token_stream(list.tokens);
                    }
                }
                syn::Meta::Path(_) => {}
            }
        }
    }
}

impl<'ast> Visit<'ast> for DependencyVisitor {
    fn visit_path(&mut self, i: &'ast syn::Path) {
        if let Some(segment) = i.segments.first() {
            self.check_ident(&segment.ident);
        }
        syn::visit::visit_path(self, i);
    }

    fn visit_use_path(&mut self, i: &'ast syn::UsePath) {
        self.check_ident(&i.ident);
        syn::visit::visit_use_path(self, i);
    }

    fn visit_use_name(&mut self, i: &'ast syn::UseName) {
        self.check_ident(&i.ident);
        syn::visit::visit_use_name(self, i);
    }

    fn visit_macro(&mut self, i: &'ast syn::Macro) {
        if let Some(segment) = i.path.segments.first() {
            self.check_ident(&segment.ident);
        }
        self.scan_token_stream(i.tokens.clone());
        syn::visit::visit_macro(self, i);
    }

    fn visit_item_extern_crate(&mut self, i: &'ast syn::ItemExternCrate) {
        self.check_ident(&i.ident);
        syn::visit::visit_item_extern_crate(self, i);
    }

    fn visit_attribute(&mut self, i: &'ast syn::Attribute) {
        self.inspect_meta(&i.meta);
        syn::visit::visit_attribute(self, i);
    }
}
