pub(crate) mod lint_policy;

use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use walkdir::WalkDir;

#[derive(Deserialize, Debug, Default)]
pub struct Whitelist {
    #[serde(default)]
    dead_code: Vec<DeadCodeEntry>,
}

#[derive(Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct DeadCodeEntry {
    file: String,
    identifier: Option<String>,
}

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
                if let syn::Meta::Path(lint_path) = item
                    && is_dead_code_lint_path(&lint_path)
                {
                    return true;
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
        for entry in &self.allowed {
            let norm_entry = entry.file.replace('\\', "/");
            let path_matches =
                self.file_path == norm_entry || self.file_path.ends_with(&format!("/{norm_entry}"));
            if path_matches {
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

fn load_whitelist(whitelist_path: &Path) -> Option<Whitelist> {
    if !whitelist_path.exists() {
        return Some(Whitelist::default());
    }
    let content = match fs::read_to_string(whitelist_path) {
        Ok(c) => c,
        Err(err) => {
            eprintln!("[!] Failed to read {}: {}", whitelist_path.display(), err);
            return None;
        }
    };
    match toml::from_str::<Whitelist>(&content) {
        Ok(whitelist) => Some(whitelist),
        Err(err) => {
            eprintln!("[!] Failed to parse {}: {}", whitelist_path.display(), err);
            None
        }
    }
}

fn collect_expected_dependencies(parsed: &toml::Value) -> HashSet<String> {
    let mut deps = Vec::new();
    for section in ["dependencies", "dev-dependencies"] {
        if let Some(table) = parsed.get(section).and_then(toml::Value::as_table) {
            deps.extend(table.keys().cloned());
        }
    }

    let mut expected_deps: HashSet<String> =
        deps.into_iter().map(|d| d.replace('-', "_")).collect();

    // Commonly implicitly used macros or dependencies without direct path usage
    for implicit in [
        "thiserror",
        "serde",
        "approx",
        "wasm_bindgen",
        "console_error_panic_hook",
        "proc_macro2",
        "quote",
        "syn",
        "serde_json",
    ] {
        expected_deps.remove(implicit);
    }
    expected_deps
}

fn is_implicitly_used_dep(member: &str, expected: &str) -> bool {
    let crate_name = member
        .split('/')
        .next_back()
        .unwrap_or("")
        .replace('-', "_");
    if expected == crate_name {
        return true;
    }
    matches!(
        expected,
        "document_features" | "eframe" | "math_explorer" | "math_commons" | "scientific_metadata"
    )
}

fn audit_member_sources(
    member_path: &str,
    whitelist: &Whitelist,
    check_clippy_bypasses: bool,
    dep_visitor: &mut DependencyVisitor,
) -> bool {
    let mut success = true;
    for entry in WalkDir::new(member_path)
        .into_iter()
        .filter_entry(|e| e.file_name() != "target")
        .filter_map(Result::ok)
    {
        if entry.path().extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let file_path = entry.path().to_string_lossy().replace('\\', "/");
        if let Ok(content) = fs::read_to_string(entry.path())
            && let Ok(ast) = syn::parse_file(&content)
        {
            dep_visitor.visit_file(&ast);

            let mut dc_visitor = DeadCodeVisitor {
                file_path: file_path.clone(),
                allowed: whitelist.dead_code.clone(),
                has_unauthorized: false,
            };
            dc_visitor.visit_file(&ast);
            if dc_visitor.has_unauthorized {
                success = false;
            }

            if check_clippy_bypasses {
                let mut clippy_visitor = lint_policy::ClippyBypassVisitor {
                    file_path,
                    has_unauthorized: false,
                };
                clippy_visitor.visit_file(&ast);
                if clippy_visitor.has_unauthorized {
                    success = false;
                }
            }
        }
    }
    success
}

fn audit_single_member(member: &str, whitelist: &Whitelist) -> bool {
    let member_dir = lint_policy::resolve_member_dir(member);
    let cargo_toml_path = member_dir.join("Cargo.toml");
    let Ok(content) = fs::read_to_string(&cargo_toml_path) else {
        eprintln!("[!] Failed to read {}", cargo_toml_path.display());
        return false;
    };
    let parsed: toml::Value = match toml::from_str(&content) {
        Ok(val) => val,
        Err(err) => {
            eprintln!("[!] Failed to parse {}: {}", cargo_toml_path.display(), err);
            return false;
        }
    };

    let mut success = lint_policy::check_manifest_lints(member, &cargo_toml_path, &parsed);
    let check_clippy_bypasses = !lint_policy::is_vendored_egui_plot_exempt(member, &parsed);
    let expected_deps = collect_expected_dependencies(&parsed);
    let mut dep_visitor = DependencyVisitor {
        expected_deps: expected_deps.clone(),
        used_deps: HashSet::new(),
    };

    let member_scan_path = member_dir.to_string_lossy();
    if !audit_member_sources(
        &member_scan_path,
        whitelist,
        check_clippy_bypasses,
        &mut dep_visitor,
    ) {
        success = false;
    }

    // markdown_tests generates test files in OUT_DIR which are not scanned by WalkDir.
    if member != "crates/markdown_tests" {
        for expected in expected_deps {
            if !dep_visitor.used_deps.contains(&expected)
                && !is_implicitly_used_dep(member, &expected)
            {
                println!("[!] Unused dependency detected in {}: {}", member, expected);
                success = false;
            }
        }
    }

    success
}

pub fn run_policy_audit(members: &[&str]) -> bool {
    let whitelist_path =
        lint_policy::resolve_whitelist_path(Path::new("verification_whitelist.toml"));
    run_policy_audit_with_whitelist(members, &whitelist_path)
}

pub(crate) fn run_policy_audit_with_whitelist(members: &[&str], whitelist_path: &Path) -> bool {
    let Some(whitelist) = load_whitelist(whitelist_path) else {
        return false;
    };
    let mut success = true;

    println!(
        "Running Policy Audit for lint inheritance, Clippy/dead-code bypasses, and unused dependencies..."
    );

    for member in members {
        if !audit_single_member(member, &whitelist) {
            success = false;
        }
    }

    success
}

#[cfg(test)]
mod tests;
