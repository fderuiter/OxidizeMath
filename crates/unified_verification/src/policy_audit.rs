use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use syn::visit::Visit;
use walkdir::WalkDir;

#[derive(Deserialize, Debug)]
pub struct Whitelist {
    dead_code: Vec<DeadCodeEntry>,
}

#[derive(Deserialize, Debug)]
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

struct DeadCodeVisitor {
    file_path: String,
    allowed: Vec<DeadCodeEntry>,
    has_unauthorized: bool,
}

impl DeadCodeVisitor {
    fn is_authorized(&self, ident: Option<&str>) -> bool {
        for entry in &self.allowed {
            let mut path_matches = self.file_path == entry.file;
            if !path_matches {
                if self.file_path.ends_with(&entry.file) {
                    path_matches = true;
                }
            }
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
            let meta_str = quote::quote!(#attr).to_string();
            if meta_str.contains("allow") && meta_str.contains("dead_code") {
                if !self.is_authorized(ident) {
                    println!(
                        "[!] Unauthorized dead code bypass in {} (identifier: {:?})",
                        self.file_path, ident
                    );
                    self.has_unauthorized = true;
                }
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
}

pub fn run_policy_audit(members: &[&str]) -> bool {
    let mut success = true;

    let whitelist_path = "verification_whitelist.toml";
    let whitelist: Whitelist = if Path::new(whitelist_path).exists() {
        let content = fs::read_to_string(whitelist_path).unwrap();
        toml::from_str(&content).unwrap_or_else(|_| Whitelist { dead_code: vec![] })
    } else {
        Whitelist { dead_code: vec![] }
    };

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
                        if content.contains("allow(dead_code)") {
                            // Convert allowed list to a cloned version
                            let allowed = whitelist
                                .dead_code
                                .iter()
                                .map(|e| DeadCodeEntry {
                                    file: e.file.clone(),
                                    identifier: e.identifier.clone(),
                                })
                                .collect();

                            let mut dc_visitor = DeadCodeVisitor {
                                file_path: file_path.clone(),
                                allowed,
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
mod tests {
    use super::*;

    #[test]
    fn test_policy_audit_workspace_run() {
        let _lock = crate::DIR_LOCK.lock().unwrap();
        let members = vec!["crates/pure_math"];
        let success = run_policy_audit(&members);
        assert!(success, "pure_math should pass policy audit");
    }
}
