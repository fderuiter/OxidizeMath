mod attrs;

use attrs::{
    attrs_mark_test_fn, attrs_mention_cfg_test, count_block_assertions, extract_opt_outs_only,
    parse_verification_macro_tokens, scan_attributes,
};
use std::collections::HashMap;
use std::path::Path;
use syn::visit::{self, Visit};
use syn::{Attribute, ImplItem, ImplItemFn, Item, ItemFn, ItemMacro, TraitItem, TraitItemFn};

/// Returns `true` when `filepath` resides in a dedicated test, benchmark, or example path.
pub fn is_dedicated_test_path(filepath: &Path) -> bool {
    let stem = filepath.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if stem == "tests" || stem.ends_with("_tests") || stem.starts_with("tests_") {
        return true;
    }
    filepath.components().any(|c| {
        matches!(
            c.as_os_str().to_str(),
            Some("tests" | "benches" | "examples")
        )
    })
}

#[allow(missing_docs)]
pub struct AstVisitor {
    #[allow(missing_docs)]
    pub verified_modules: Vec<String>,
    #[allow(missing_docs)]
    pub module_tiers: HashMap<String, String>,
    #[allow(missing_docs)]
    pub has_vacuous_bypass: bool,
    #[allow(missing_docs)]
    pub has_verification_macro: bool,
    #[allow(missing_docs)]
    pub total_funcs: usize,
    #[allow(missing_docs)]
    pub total_asserts: usize,
    #[allow(missing_docs)]
    pub verified_funcs: usize,
    #[allow(missing_docs)]
    pub verified_asserts: usize,
    #[allow(missing_docs)]
    pub prod_funcs: usize,
    #[allow(missing_docs)]
    pub prod_asserts: usize,
    #[allow(missing_docs)]
    pub verified_prod_funcs: usize,
    #[allow(missing_docs)]
    pub verified_prod_asserts: usize,
    #[allow(missing_docs)]
    pub test_funcs: usize,
    #[allow(missing_docs)]
    pub test_asserts: usize,
    #[allow(missing_docs)]
    pub verified_test_funcs: usize,
    #[allow(missing_docs)]
    pub verified_test_asserts: usize,
    #[allow(missing_docs)]
    pub semantic_integrity_funcs: usize,
    #[allow(missing_docs)]
    pub opted_out_funcs: usize,
    #[allow(missing_docs)]
    pub active_submodules: Vec<String>,
    #[allow(missing_docs)]
    pub opted_out: bool,
    #[allow(missing_docs)]
    pub opt_outs: Vec<String>,
    #[allow(missing_docs)]
    pub legacy_funcs: Vec<String>,
    in_opt_out_scope: bool,
    in_test_scope: bool,
}

impl Default for AstVisitor {
    fn default() -> Self {
        Self::new()
    }
}

impl AstVisitor {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self::with_test_scope(false)
    }

    /// Creates a new [`AstVisitor`] with an explicit initial test-scope flag.
    pub fn with_test_scope(initial_test_scope: bool) -> Self {
        Self {
            verified_modules: Vec::new(),
            module_tiers: HashMap::new(),
            has_vacuous_bypass: false,
            has_verification_macro: false,
            total_funcs: 0,
            total_asserts: 0,
            verified_funcs: 0,
            verified_asserts: 0,
            prod_funcs: 0,
            prod_asserts: 0,
            verified_prod_funcs: 0,
            verified_prod_asserts: 0,
            test_funcs: 0,
            test_asserts: 0,
            verified_test_funcs: 0,
            verified_test_asserts: 0,
            semantic_integrity_funcs: 0,
            opted_out_funcs: 0,
            active_submodules: Vec::new(),
            opted_out: false,
            opt_outs: Vec::new(),
            legacy_funcs: Vec::new(),
            in_opt_out_scope: false,
            in_test_scope: initial_test_scope,
        }
    }

    fn record_fn_metrics(
        &mut self,
        fn_name: &str,
        attrs: &[Attribute],
        assert_count: usize,
    ) -> (bool, bool) {
        if fn_name.ends_with("_legacy") {
            self.legacy_funcs.push(fn_name.to_string());
        }

        let scanned = scan_attributes(attrs, &mut self.verified_modules);
        let fn_has_opt_out = !scanned.opt_outs.is_empty();
        self.opt_outs.extend(scanned.opt_outs);

        let fn_in_test = self.in_test_scope || attrs_mark_test_fn(attrs);
        let effective_opt_out = self.in_opt_out_scope || fn_has_opt_out;
        let is_verified = scanned.is_verified && !effective_opt_out;

        self.total_funcs += 1;
        self.total_asserts += assert_count;
        if effective_opt_out {
            self.opted_out_funcs += 1;
        }
        if scanned.has_semantic && !effective_opt_out {
            self.semantic_integrity_funcs += 1;
        }
        if is_verified {
            self.verified_funcs += 1;
            self.verified_asserts += assert_count;
        }

        if fn_in_test {
            self.test_funcs += 1;
            self.test_asserts += assert_count;
            if is_verified {
                self.verified_test_funcs += 1;
                self.verified_test_asserts += assert_count;
            }
        } else {
            self.prod_funcs += 1;
            self.prod_asserts += assert_count;
            if is_verified {
                self.verified_prod_funcs += 1;
                self.verified_prod_asserts += assert_count;
            }
        }

        (effective_opt_out, fn_in_test)
    }
}

fn non_fn_item_attrs(node: &Item) -> Option<&[Attribute]> {
    match node {
        Item::Fn(_) => None,
        Item::Const(i) => Some(&i.attrs),
        Item::Enum(i) => Some(&i.attrs),
        Item::ExternCrate(i) => Some(&i.attrs),
        Item::ForeignMod(i) => Some(&i.attrs),
        Item::Impl(i) => Some(&i.attrs),
        Item::Macro(i) => Some(&i.attrs),
        Item::Mod(i) => Some(&i.attrs),
        Item::Static(i) => Some(&i.attrs),
        Item::Struct(i) => Some(&i.attrs),
        Item::Trait(i) => Some(&i.attrs),
        Item::TraitAlias(i) => Some(&i.attrs),
        Item::Type(i) => Some(&i.attrs),
        Item::Union(i) => Some(&i.attrs),
        Item::Use(i) => Some(&i.attrs),
        _ => Some(&[]),
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_file(&mut self, node: &'ast syn::File) {
        let reasons = extract_opt_outs_only(&node.attrs);
        if !reasons.is_empty() {
            self.opted_out = true;
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_mention_cfg_test(&node.attrs) {
            self.in_test_scope = true;
        }
        visit::visit_file(self, node);
    }

    fn visit_item(&mut self, node: &'ast Item) {
        let Some(attrs) = non_fn_item_attrs(node) else {
            visit::visit_item(self, node);
            return;
        };
        let reasons = extract_opt_outs_only(attrs);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        if !reasons.is_empty() {
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_mention_cfg_test(attrs) {
            self.in_test_scope = true;
        }
        visit::visit_item(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_impl_item(&mut self, node: &'ast ImplItem) {
        let attrs: Option<&[Attribute]> = match node {
            ImplItem::Fn(_) => None,
            ImplItem::Const(i) => Some(&i.attrs),
            ImplItem::Type(i) => Some(&i.attrs),
            ImplItem::Macro(i) => Some(&i.attrs),
            _ => Some(&[]),
        };
        let Some(attrs) = attrs else {
            visit::visit_impl_item(self, node);
            return;
        };
        let reasons = extract_opt_outs_only(attrs);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        if !reasons.is_empty() {
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_mention_cfg_test(attrs) {
            self.in_test_scope = true;
        }
        visit::visit_impl_item(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_trait_item(&mut self, node: &'ast TraitItem) {
        let attrs: Option<&[Attribute]> = match node {
            TraitItem::Fn(_) => None,
            TraitItem::Const(i) => Some(&i.attrs),
            TraitItem::Type(i) => Some(&i.attrs),
            TraitItem::Macro(i) => Some(&i.attrs),
            _ => Some(&[]),
        };
        let Some(attrs) = attrs else {
            visit::visit_trait_item(self, node);
            return;
        };
        let reasons = extract_opt_outs_only(attrs);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        if !reasons.is_empty() {
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_mention_cfg_test(attrs) {
            self.in_test_scope = true;
        }
        visit::visit_trait_item(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
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
            } else {
                None
            };

            if let Some(tier_name) = tier {
                self.has_verification_macro = true;
                let parsed = parse_verification_macro_tokens(&node.mac.tokens);
                if parsed.has_vacuous_bypass {
                    self.has_vacuous_bypass = true;
                }
                if let Some(module_name) = parsed.module_name {
                    self.verified_modules.push(module_name.clone());
                    self.module_tiers.insert(module_name, tier_name.to_string());
                    self.semantic_integrity_funcs += 1;
                    self.total_funcs += 1;
                    self.verified_funcs += 1;
                    self.test_funcs += 1;
                    self.verified_test_funcs += 1;
                    self.total_asserts += parsed.assert_count;
                    self.verified_asserts += parsed.assert_count;
                    self.test_asserts += parsed.assert_count;
                    self.verified_test_asserts += parsed.assert_count;
                }
            }
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let asserts = count_block_assertions(&node.block);
        let name = node.sig.ident.to_string();
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        let (eff_opt, eff_test) = self.record_fn_metrics(&name, &node.attrs, asserts);
        self.in_opt_out_scope = eff_opt;
        self.in_test_scope = eff_test;
        visit::visit_item_fn(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        let asserts = count_block_assertions(&node.block);
        let name = node.sig.ident.to_string();
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        let (eff_opt, eff_test) = self.record_fn_metrics(&name, &node.attrs, asserts);
        self.in_opt_out_scope = eff_opt;
        self.in_test_scope = eff_test;
        visit::visit_impl_item_fn(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_trait_item_fn(&mut self, node: &'ast TraitItemFn) {
        let name = node.sig.ident.to_string();
        let Some(ref block) = node.default else {
            if name.ends_with("_legacy") {
                self.legacy_funcs.push(name);
            }
            self.opt_outs.extend(extract_opt_outs_only(&node.attrs));
            visit::visit_trait_item_fn(self, node);
            return;
        };
        let asserts = count_block_assertions(block);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        let (eff_opt, eff_test) = self.record_fn_metrics(&name, &node.attrs, asserts);
        self.in_opt_out_scope = eff_opt;
        self.in_test_scope = eff_test;
        visit::visit_trait_item_fn(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }
}

#[cfg(test)]
#[path = "ast_visitor_tests.rs"]
mod tests;
