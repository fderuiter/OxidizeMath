pub use crate::whitelist::DeadCodeEntry;
use crate::whitelist::{PathMatchOptions, ROOT_ONLY_WHITELIST_CANDIDATES};
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use walkdir::WalkDir;

struct DependencyVisitor {
    expected_deps: HashSet<String>,
    used_deps: HashSet<String>,
}

impl DependencyVisitor {
    fn check_ident(&mut self, ident: &syn::Ident) {
        let name = ident.to_string();
        if self.expected_deps.contains(&name) {
            self.used_deps.insert(name);
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

        // Extract identifiers from inside the macro's token stream
        let mut streams = vec![i.tokens.clone()];
        while let Some(stream) = streams.pop() {
            for token in stream {
                match token {
                    proc_macro2::TokenTree::Ident(ident) => {
                        let name = ident.to_string();
                        if self.expected_deps.contains(&name) {
                            self.used_deps.insert(name);
                        }
                    }
                    proc_macro2::TokenTree::Group(group) => {
                        streams.push(group.stream());
                    }
                    _ => {}
                }
            }
        }

        syn::visit::visit_macro(self, i);
    }
    fn visit_item_extern_crate(&mut self, i: &'ast syn::ItemExternCrate) {
        self.check_ident(&i.ident);
        syn::visit::visit_item_extern_crate(self, i);
    }
}

fn is_dead_code_lint_name(ident: &syn::Ident) -> bool {
    ident == "dead_code" || ident == "unused" || ident == "warnings"
}

fn is_dead_code_lint_path(path: &syn::Path) -> bool {
    if path.segments.iter().any(|seg| !seg.arguments.is_empty()) {
        return false;
    }
    match path.segments.len() {
        1 => is_dead_code_lint_name(&path.segments[0].ident),
        2 => path.segments[0].ident == "rust" && is_dead_code_lint_name(&path.segments[1].ident),
        _ => false,
    }
}

fn attr_suppresses_dead_code(attr: &syn::Attribute) -> bool {
    let mut stack = vec![attr.meta.clone()];
    while let Some(meta) = stack.pop() {
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
                if let syn::Meta::Path(lint_path) = item {
                    if is_dead_code_lint_path(&lint_path) {
                        return true;
                    }
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
    false
}

fn impl_self_ident(ty: &syn::Type) -> Option<String> {
    let mut current = ty;
    while let syn::Type::Group(group) = current {
        current = &group.elem;
    }
    if let syn::Type::Paren(paren) = current {
        current = &paren.elem;
    }
    if let syn::Type::Path(type_path) = current {
        type_path.path.segments.last().map(|s| s.ident.to_string())
    } else {
        None
    }
}

struct DeadCodeVisitor {
    file_path: String,
    allowed: Vec<DeadCodeEntry>,
    has_unauthorized: bool,
}

impl DeadCodeVisitor {
    fn is_authorized(&self, ident: Option<&str>) -> bool {
        let opts = PathMatchOptions {
            trim_dot_slash: false,
            require_non_empty: false,
            match_file_name: false,
        };
        let actual_path = Path::new(&self.file_path);
        for entry in &self.allowed {
            if crate::whitelist::path_matches_lossy(actual_path, &entry.file, opts) {
                match (&entry.identifier, ident) {
                    (None, _) => return true,
                    (Some(allowed_id), Some(id)) if allowed_id == id => return true,
                    _ => {}
                }
            }
        }
        false
    }

    fn check_attrs(&mut self, attrs: &[syn::Attribute], ident: Option<&str>) {
        for attr in attrs {
            if attr_suppresses_dead_code(attr) && !self.is_authorized(ident) {
                println!(
                    "[!] Unauthorized dead code bypass in {} (identifier: {:?})",
                    self.file_path, ident
                );
                self.has_unauthorized = true;
            }
        }
    }
}

impl<'ast> Visit<'ast> for DeadCodeVisitor {
    fn visit_file(&mut self, i: &'ast syn::File) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_file(self, i);
    }
    fn visit_item_fn(&mut self, i: &'ast syn::ItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_item_fn(self, i);
    }
    fn visit_item_struct(&mut self, i: &'ast syn::ItemStruct) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_struct(self, i);
    }
    fn visit_item_enum(&mut self, i: &'ast syn::ItemEnum) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_enum(self, i);
    }
    fn visit_item_union(&mut self, i: &'ast syn::ItemUnion) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_union(self, i);
    }
    fn visit_item_type(&mut self, i: &'ast syn::ItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_type(self, i);
    }
    fn visit_item_mod(&mut self, i: &'ast syn::ItemMod) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_mod(self, i);
    }
    fn visit_item_const(&mut self, i: &'ast syn::ItemConst) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_const(self, i);
    }
    fn visit_item_static(&mut self, i: &'ast syn::ItemStatic) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_static(self, i);
    }
    fn visit_item_trait(&mut self, i: &'ast syn::ItemTrait) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_trait(self, i);
    }
    fn visit_item_trait_alias(&mut self, i: &'ast syn::ItemTraitAlias) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_item_trait_alias(self, i);
    }
    fn visit_item_impl(&mut self, i: &'ast syn::ItemImpl) {
        let ident = impl_self_ident(&i.self_ty);
        self.check_attrs(&i.attrs, ident.as_deref());
        syn::visit::visit_item_impl(self, i);
    }
    fn visit_item_use(&mut self, i: &'ast syn::ItemUse) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_item_use(self, i);
    }
    fn visit_item_macro(&mut self, i: &'ast syn::ItemMacro) {
        let ident = i.ident.as_ref().map(ToString::to_string);
        self.check_attrs(&i.attrs, ident.as_deref());
        syn::visit::visit_item_macro(self, i);
    }
    fn visit_item_foreign_mod(&mut self, i: &'ast syn::ItemForeignMod) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_item_foreign_mod(self, i);
    }
    fn visit_field(&mut self, i: &'ast syn::Field) {
        if let Some(ident) = &i.ident {
            self.check_attrs(&i.attrs, Some(&ident.to_string()));
        } else {
            self.check_attrs(&i.attrs, None);
        }
        syn::visit::visit_field(self, i);
    }
    fn visit_variant(&mut self, i: &'ast syn::Variant) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_variant(self, i);
    }
    fn visit_impl_item_fn(&mut self, i: &'ast syn::ImplItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_impl_item_fn(self, i);
    }
    fn visit_impl_item_const(&mut self, i: &'ast syn::ImplItemConst) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_impl_item_const(self, i);
    }
    fn visit_impl_item_type(&mut self, i: &'ast syn::ImplItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_impl_item_type(self, i);
    }
    fn visit_trait_item_fn(&mut self, i: &'ast syn::TraitItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_trait_item_fn(self, i);
    }
    fn visit_trait_item_const(&mut self, i: &'ast syn::TraitItemConst) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_trait_item_const(self, i);
    }
    fn visit_trait_item_type(&mut self, i: &'ast syn::TraitItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_trait_item_type(self, i);
    }
    fn visit_foreign_item_fn(&mut self, i: &'ast syn::ForeignItemFn) {
        self.check_attrs(&i.attrs, Some(&i.sig.ident.to_string()));
        syn::visit::visit_foreign_item_fn(self, i);
    }
    fn visit_foreign_item_static(&mut self, i: &'ast syn::ForeignItemStatic) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_foreign_item_static(self, i);
    }
    fn visit_foreign_item_type(&mut self, i: &'ast syn::ForeignItemType) {
        self.check_attrs(&i.attrs, Some(&i.ident.to_string()));
        syn::visit::visit_foreign_item_type(self, i);
    }
    fn visit_local(&mut self, i: &'ast syn::Local) {
        self.check_attrs(&i.attrs, None);
        syn::visit::visit_local(self, i);
    }
}

pub fn run_policy_audit(members: &[&str]) -> bool {
    let whitelist_path = crate::whitelist::resolve_whitelist_path(ROOT_ONLY_WHITELIST_CANDIDATES)
        .unwrap_or_else(|| Path::new(crate::whitelist::WHITELIST_FILE_NAME).to_path_buf());
    run_policy_audit_with_whitelist(members, &whitelist_path)
}

pub(crate) fn run_policy_audit_with_whitelist(members: &[&str], whitelist_path: &Path) -> bool {
    let whitelist = match crate::whitelist::load_whitelist_strict(None, &[whitelist_path]) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {}", err);
            return false;
        }
    };
    let mut success = true;

    println!("Running Policy Audit for unused dependencies and dead code bypasses...");

    for member in members {
        let member_path = member.to_string();

        let cargo_toml_path = format!("{}/Cargo.toml", member_path);
        let content = fs::read_to_string(&cargo_toml_path).unwrap_or_default();
        let parsed: toml::Value =
            toml::from_str(&content).unwrap_or_else(|_| toml::Value::Table(Default::default()));

        let mut deps = Vec::new();
        if let Some(d) = parsed.get("dependencies").and_then(|v| v.as_table()) {
            for (k, _) in d {
                deps.push(k.clone());
            }
        }
        if let Some(d) = parsed.get("dev-dependencies").and_then(|v| v.as_table()) {
            for (k, _) in d {
                deps.push(k.clone());
            }
        }

        let mut expected_deps: HashSet<String> =
            deps.into_iter().map(|d| d.replace("-", "_")).collect();

        // Commonly implicitly used macros or dependencies without direct path usage
        expected_deps.remove("thiserror");
        expected_deps.remove("serde");
        expected_deps.remove("approx");
        expected_deps.remove("wasm_bindgen");
        expected_deps.remove("console_error_panic_hook");
        expected_deps.remove("proc_macro2");
        expected_deps.remove("quote");
        expected_deps.remove("syn");
        expected_deps.remove("serde_json");

        if member_path == "crates/markdown_tests" {
            // markdown_tests generates test files in the target directory which are not scanned by WalkDir.
            // All its dependencies are considered used.
            continue;
        }

        let mut dep_visitor = DependencyVisitor {
            expected_deps: expected_deps.clone(),
            used_deps: HashSet::new(),
        };

        for entry in WalkDir::new(&member_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if entry.path().extension().and_then(|s| s.to_str()) == Some("rs") {
                let file_path = entry.path().to_string_lossy().replace("\\", "/");
                if let Ok(content) = fs::read_to_string(entry.path()) {
                    if let Ok(ast) = syn::parse_file(&content) {
                        // Check dependencies
                        dep_visitor.visit_file(&ast);

                        // Check dead code
                        let mut dc_visitor = DeadCodeVisitor {
                            file_path: file_path.clone(),
                            allowed: whitelist.dead_code.clone(),
                            has_unauthorized: false,
                        };
                        dc_visitor.visit_file(&ast);
                        if dc_visitor.has_unauthorized {
                            success = false;
                        }
                    }
                }
            }
        }

        for expected in expected_deps {
            if !dep_visitor.used_deps.contains(&expected) {
                // If it's the current crate, ignore
                let crate_name = member.split('/').last().unwrap_or("").replace("-", "_");
                if expected == crate_name {
                    continue;
                }

                // Some special cases where cargo checks might not easily see usage
                if expected == "document_features"
                    || expected == "eframe"
                    || expected == "math_explorer"
                    || expected == "math_commons"
                    || expected == "scientific_metadata"
                {
                    continue;
                }

                println!("[!] Unused dependency detected in {}: {}", member, expected);
                success = false;
            }
        }
    }

    success
}

#[cfg(test)]
mod tests;
