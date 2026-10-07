use std::collections::HashMap;
use syn::visit::{self, Visit};
use syn::{ImplItemFn, ItemFn, ItemMacro, Macro, TraitItemFn};

#[allow(missing_docs)]
pub struct AstVisitor {
    #[allow(missing_docs)]
    pub verified_modules: Vec<String>,
    #[allow(missing_docs)]
    pub module_tiers: HashMap<String, String>,
    #[allow(missing_docs)]
    pub has_vacuous_bypass: bool,
    #[allow(missing_docs)]
    pub total_funcs: usize,
    #[allow(missing_docs)]
    pub total_asserts: usize,
    #[allow(missing_docs)]
    pub verified_funcs: usize,
    #[allow(missing_docs)]
    pub verified_asserts: usize,
    #[allow(missing_docs)]
    pub semantic_integrity_funcs: usize,
    #[allow(missing_docs)]
    pub active_submodules: Vec<String>,
    #[allow(missing_docs)]
    pub opted_out: bool,
}

impl Default for AstVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl AstVisitor {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            verified_modules: Vec::new(),
            module_tiers: HashMap::new(),
            has_vacuous_bypass: false,
            total_funcs: 0,
            total_asserts: 0,
            verified_funcs: 0,
            verified_asserts: 0,
            semantic_integrity_funcs: 0,
            active_submodules: Vec::new(),
            opted_out: false,
        }
    }

    fn process_fn_attrs(&mut self, attrs: &[syn::Attribute], assert_count: usize) {
        self.total_funcs += 1;

        let mut is_verified = false;
        let mut has_semantic = false;
        for attr in attrs {
            let attr_str = quote::quote!(#attr).to_string().replace(" ", "");

            if attr_str.contains("opt_out") {
                continue;
            }

            if attr_str.contains("verified_engine::verified") || attr_str.contains("#[verified]") {
                is_verified = true;
                has_semantic = true;
            }

            if attr_str.contains("embed_theory") {
                is_verified = true;
                has_semantic = true;

                let s = attr_str.replace("\\", "/");
                if let Some(pos) = s.find('"') {
                    if let Some(end) = s[pos + 1..].find('"') {
                        let raw_path = &s[pos + 1..pos + 1 + end];
                        let clean_name = raw_path
                            .trim_start_matches("papers/")
                            .trim_start_matches("spec:")
                            .trim_start_matches("registry:")
                            .trim_end_matches(".tex");
                        if !clean_name.is_empty() {
                            self.verified_modules.push(clean_name.to_string());
                        }
                    }
                }
            }

            let doc_text = quote::quote!(#attr).to_string();
            let extracted = crate::traceability::TraceabilityEngine::<crate::vfs::DefaultVfs>::extract_citations(&doc_text);
            if !extracted.is_empty() {
                has_semantic = true;
                for cite in extracted {
                    self.verified_modules.push(cite);
                }
            }
        }

        if self.opted_out {
            is_verified = false;
        }

        if is_verified {
            self.verified_funcs += 1;
        }
        if has_semantic {
            self.semantic_integrity_funcs += 1;
        }

        self.total_asserts += assert_count;
        if is_verified {
            self.verified_asserts += assert_count;
        }
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_file(&mut self, node: &'ast syn::File) {
        for attr in &node.attrs {
            let attr_str = quote::quote!(#attr).to_string().replace(" ", "");
            if attr_str.contains("opt_out") {
                self.opted_out = true;
            }
        }
        visit::visit_file(self, node);
    }

    fn visit_item(&mut self, node: &'ast syn::Item) {
        let attrs: &[syn::Attribute] = match node {
            syn::Item::Struct(i) => &i.attrs,
            syn::Item::Enum(i) => &i.attrs,
            syn::Item::Impl(i) => &i.attrs,
            syn::Item::Fn(i) => &i.attrs,
            syn::Item::Mod(i) => &i.attrs,
            syn::Item::Trait(i) => &i.attrs,
            _ => &[],
        };
        for attr in attrs {
            let attr_str = quote::quote!(#attr).to_string().replace(" ", "");
            if attr_str.contains("opt_out") {
                self.opted_out = true;
            }
        }
        visit::visit_item(self, node);
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.content.is_none() {
            self.active_submodules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        if let Some(ident) = node.mac.path.segments.last().map(|s| &s.ident) {
            let name = ident.to_string();
            let tier = if name == "theory_verification" {
                Some("Deterministic")
            } else if name == "stochastic_signature_verification" {
                Some("Stochastic")
            } else if name == "empirical_verification" {
                Some("Empirical")
            } else {
                None
            };

            if let Some(tier_name) = tier {
                let tokens = node.mac.tokens.to_string();

                // Detect vacuous bypass: zero initializations in stochastic/empirical
                if tokens.contains("zeros(")
                    || tokens.contains("zeros_like")
                    || tokens.contains("0.0")
                    || tokens.contains("fill(0)")
                {
                    self.has_vacuous_bypass = true;
                }

                if let Some(idx) = tokens.find("module = \"") {
                    let start = idx + 10;
                    if let Some(end) = tokens[start..].find('"') {
                        let module_name = &tokens[start..start + end];
                        self.verified_modules.push(module_name.to_string());
                        self.module_tiers
                            .insert(module_name.to_string(), tier_name.to_string());
                        self.semantic_integrity_funcs += 1;
                        self.total_funcs += 1;
                        self.verified_funcs += 1;
                        let assert_count = tokens.matches("assert").count().max(3);
                        self.total_asserts += assert_count;
                        self.verified_asserts += assert_count;
                    }
                }
            }
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let mut assert_visitor = AssertVisitor { count: 0 };
        visit::visit_item_fn(&mut assert_visitor, node);
        self.process_fn_attrs(&node.attrs, assert_visitor.count);

        // Continue visiting inside the function (if there are nested items)
        visit::visit_item_fn(self, node);
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        let mut assert_visitor = AssertVisitor { count: 0 };
        visit::visit_impl_item_fn(&mut assert_visitor, node);
        self.process_fn_attrs(&node.attrs, assert_visitor.count);

        visit::visit_impl_item_fn(self, node);
    }

    fn visit_trait_item_fn(&mut self, node: &'ast TraitItemFn) {
        if node.default.is_some() {
            let mut assert_visitor = AssertVisitor { count: 0 };
            visit::visit_trait_item_fn(&mut assert_visitor, node);
            self.process_fn_attrs(&node.attrs, assert_visitor.count);
        }

        visit::visit_trait_item_fn(self, node);
    }
}

struct AssertVisitor {
    count: usize,
}

impl<'ast> Visit<'ast> for AssertVisitor {
    fn visit_macro(&mut self, node: &'ast Macro) {
        if let Some(ident) = node.path.segments.last().map(|s| &s.ident) {
            let name = ident.to_string();
            if name == "assert"
                || name == "assert_eq"
                || name == "assert_ne"
                || name == "debug_assert"
                || name == "debug_assert_eq"
                || name == "debug_assert_ne"
            {
                self.count += 1;
            }
        }
        visit::visit_macro(self, node);
    }
}
