mod helpers;
mod stochastic;

pub use helpers::{
    attrs_mark_test_fn, attrs_require_cfg_test, collect_ast_file_debt,
    extract_opt_out_reasons_from_attr,
};

use helpers::{
    attr_has_opt_out, impl_item_attrs, inspect_non_doc_verified_attr, item_attrs,
    parse_verification_macro, trait_item_attrs,
};
use std::collections::HashMap;
use stochastic::{find_stochastic_zero_init_line_in_block, span_line};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{
    Expr, ImplItem, ImplItemFn, Item, ItemFn, ItemMacro, Macro, Stmt, TraitItem, TraitItemFn,
};

#[allow(missing_docs)]
pub struct AstVisitor {
    #[allow(missing_docs)]
    pub verified_modules: Vec<String>,
    #[allow(missing_docs)]
    pub module_tiers: HashMap<String, String>,
    #[allow(missing_docs)]
    pub module_spans: HashMap<String, usize>,
    #[allow(missing_docs)]
    pub has_vacuous_bypass: bool,
    #[allow(missing_docs)]
    pub vacuous_bypass_line: Option<usize>,
    #[allow(missing_docs)]
    pub first_unverified_fn: Option<(String, usize)>,
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
    /// Explicit `verified(opt_out = "...")` reason strings recorded across visited non-test scopes.
    pub opt_out_reasons: Vec<String>,
    file_opted_out: bool,
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
            module_spans: HashMap::new(),
            has_vacuous_bypass: false,
            vacuous_bypass_line: None,
            first_unverified_fn: None,
            total_funcs: 0,
            total_asserts: 0,
            verified_funcs: 0,
            verified_asserts: 0,
            semantic_integrity_funcs: 0,
            active_submodules: Vec::new(),
            opted_out: false,
            opt_out_reasons: Vec::new(),
            file_opted_out: false,
        }
    }

    fn record_module(&mut self, name: String, line: usize) {
        if line > 0 {
            self.module_spans.entry(name.clone()).or_insert(line);
        }
        self.verified_modules.push(name);
    }

    fn inspect_fn_attr(&mut self, attr: &syn::Attribute) -> (bool, bool) {
        if attr_has_opt_out(attr) {
            return (false, false);
        }
        let attr_line = span_line(attr.span());
        if attr.path().is_ident("doc") {
            let doc_text = quote::quote!(#attr).to_string();
            let extracted =
                crate::traceability::TraceabilityEngine::<crate::vfs::DefaultVfs>::extract_citations(
                    &doc_text,
                );
            for cite in &extracted {
                self.record_module(cite.clone(), attr_line);
            }
            return (false, !extracted.is_empty());
        }

        let (is_verified, embed_module) = inspect_non_doc_verified_attr(attr);
        if let Some(module_name) = embed_module {
            self.record_module(module_name, attr_line);
        }
        (is_verified, is_verified)
    }

    fn process_fn_attrs(
        &mut self,
        attrs: &[syn::Attribute],
        fn_ident: &syn::Ident,
        assert_count: usize,
    ) {
        self.total_funcs += 1;

        let mut is_verified = false;
        let mut has_semantic = false;
        for attr in attrs {
            let (attr_verified, attr_semantic) = self.inspect_fn_attr(attr);
            is_verified |= attr_verified;
            has_semantic |= attr_semantic;
        }

        if self.opted_out {
            is_verified = false;
        } else if !is_verified && self.first_unverified_fn.is_none() {
            self.first_unverified_fn = Some((fn_ident.to_string(), span_line(fn_ident.span())));
        }

        if is_verified {
            self.verified_funcs += 1;
            self.verified_asserts += assert_count;
        }
        if has_semantic {
            self.semantic_integrity_funcs += 1;
        }
        self.total_asserts += assert_count;
    }

    fn record_scoped_opt_outs(&mut self, attrs: &[syn::Attribute], allow_inner: bool) {
        for attr in attrs {
            if matches!(attr.style, syn::AttrStyle::Outer) || allow_inner {
                let reasons = extract_opt_out_reasons_from_attr(attr);
                if !reasons.is_empty() {
                    self.opted_out = true;
                    self.opt_out_reasons.extend(reasons);
                }
            }
        }
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_file(&mut self, node: &'ast syn::File) {
        for attr in &node.attrs {
            if matches!(attr.style, syn::AttrStyle::Inner(_)) {
                let reasons = extract_opt_out_reasons_from_attr(attr);
                if !reasons.is_empty() {
                    self.file_opted_out = true;
                    self.opted_out = true;
                    self.opt_out_reasons.extend(reasons);
                }
            }
        }
        if attrs_require_cfg_test(&node.attrs) {
            self.opted_out = self.file_opted_out;
            return;
        }
        visit::visit_file(self, node);
        self.opted_out = self.file_opted_out;
    }

    fn visit_item(&mut self, node: &'ast Item) {
        let attrs = item_attrs(node);
        if attrs_require_cfg_test(attrs) {
            return;
        }
        if let Item::Fn(item_fn) = node
            && attrs_mark_test_fn(&item_fn.attrs)
        {
            return;
        }
        let prev = self.opted_out;
        let allow_inner = matches!(node, Item::Mod(m) if m.content.is_some());
        self.record_scoped_opt_outs(attrs, allow_inner);
        visit::visit_item(self, node);
        self.opted_out = prev;
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if attrs_require_cfg_test(&node.attrs) {
            return;
        }
        if node.content.is_none() {
            self.active_submodules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        if attrs_require_cfg_test(&node.attrs) {
            return;
        }
        if let Some(ident) = node.mac.path.segments.last().map(|s| &s.ident) {
            let name = ident.to_string();
            let spec = if name == "theory_verification" {
                Some(("Deterministic", "test"))
            } else if name == "stochastic_signature_verification" {
                Some(("Stochastic", "stochastic_test"))
            } else {
                None
            };

            if let Some((tier_name, test_key)) = spec
                && let Some((module_name, test_block)) =
                    parse_verification_macro(&node.mac, test_key)
            {
                let macro_line = span_line(node.span());
                if name == "stochastic_signature_verification"
                    && let Some(block) = &test_block
                    && let Some(zero_line) = find_stochastic_zero_init_line_in_block(block)
                {
                    self.has_vacuous_bypass = true;
                    let recorded = if zero_line > 0 { zero_line } else { macro_line };
                    if self.vacuous_bypass_line.is_none() && recorded > 0 {
                        self.vacuous_bypass_line = Some(recorded);
                    }
                }

                self.record_module(module_name.clone(), macro_line);
                self.module_tiers.insert(module_name, tier_name.to_string());
                self.semantic_integrity_funcs += 1;
                self.total_funcs += 1;
                self.verified_funcs += 1;

                let mut assert_visitor = AssertVisitor { count: 0 };
                if let Some(block) = &test_block {
                    assert_visitor.visit_block(block);
                }
                self.total_asserts += assert_visitor.count;
                self.verified_asserts += assert_visitor.count;
            }
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_impl_item(&mut self, node: &'ast ImplItem) {
        let attrs = impl_item_attrs(node);
        if attrs_require_cfg_test(attrs) {
            return;
        }
        if let ImplItem::Fn(item_fn) = node
            && attrs_mark_test_fn(&item_fn.attrs)
        {
            return;
        }
        let prev = self.opted_out;
        self.record_scoped_opt_outs(attrs, false);
        visit::visit_impl_item(self, node);
        self.opted_out = prev;
    }

    fn visit_trait_item(&mut self, node: &'ast TraitItem) {
        let attrs = trait_item_attrs(node);
        if attrs_require_cfg_test(attrs) {
            return;
        }
        if let TraitItem::Fn(item_fn) = node
            && attrs_mark_test_fn(&item_fn.attrs)
        {
            return;
        }
        let prev = self.opted_out;
        self.record_scoped_opt_outs(attrs, false);
        visit::visit_trait_item(self, node);
        self.opted_out = prev;
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        if attrs_require_cfg_test(&node.attrs) || attrs_mark_test_fn(&node.attrs) {
            return;
        }
        let prev = self.opted_out;
        if node.attrs.iter().any(attr_has_opt_out) {
            self.opted_out = true;
        }
        let mut assert_visitor = AssertVisitor { count: 0 };
        visit::visit_item_fn(&mut assert_visitor, node);
        self.process_fn_attrs(&node.attrs, &node.sig.ident, assert_visitor.count);
        visit::visit_item_fn(self, node);
        self.opted_out = prev;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        if attrs_require_cfg_test(&node.attrs) || attrs_mark_test_fn(&node.attrs) {
            return;
        }
        let prev = self.opted_out;
        if node.attrs.iter().any(attr_has_opt_out) {
            self.opted_out = true;
        }
        let mut assert_visitor = AssertVisitor { count: 0 };
        visit::visit_impl_item_fn(&mut assert_visitor, node);
        self.process_fn_attrs(&node.attrs, &node.sig.ident, assert_visitor.count);
        visit::visit_impl_item_fn(self, node);
        self.opted_out = prev;
    }

    fn visit_trait_item_fn(&mut self, node: &'ast TraitItemFn) {
        if attrs_require_cfg_test(&node.attrs) || attrs_mark_test_fn(&node.attrs) {
            return;
        }
        let prev = self.opted_out;
        if node.attrs.iter().any(attr_has_opt_out) {
            self.opted_out = true;
        }
        if node.default.is_some() {
            let mut assert_visitor = AssertVisitor { count: 0 };
            visit::visit_trait_item_fn(&mut assert_visitor, node);
            self.process_fn_attrs(&node.attrs, &node.sig.ident, assert_visitor.count);
        }
        visit::visit_trait_item_fn(self, node);
        self.opted_out = prev;
    }
}

fn expr_attrs(expr: &Expr) -> &[syn::Attribute] {
    match expr {
        Expr::Block(e) => &e.attrs,
        Expr::Call(e) => &e.attrs,
        Expr::If(e) => &e.attrs,
        Expr::Match(e) => &e.attrs,
        Expr::MethodCall(e) => &e.attrs,
        Expr::Macro(e) => &e.attrs,
        _ => &[],
    }
}

struct AssertVisitor {
    count: usize,
}

impl<'ast> Visit<'ast> for AssertVisitor {
    fn visit_item(&mut self, node: &'ast Item) {
        if attrs_require_cfg_test(item_attrs(node)) {
            return;
        }
        visit::visit_item(self, node);
    }

    fn visit_stmt(&mut self, node: &'ast Stmt) {
        let skipped = match node {
            Stmt::Local(local) => attrs_require_cfg_test(&local.attrs),
            Stmt::Macro(mac) => attrs_require_cfg_test(&mac.attrs),
            Stmt::Item(item) => attrs_require_cfg_test(item_attrs(item)),
            Stmt::Expr(expr, _) => attrs_require_cfg_test(expr_attrs(expr)),
        };
        if skipped {
            return;
        }
        visit::visit_stmt(self, node);
    }

    fn visit_expr(&mut self, node: &'ast Expr) {
        if attrs_require_cfg_test(expr_attrs(node)) {
            return;
        }
        visit::visit_expr(self, node);
    }

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

#[cfg(test)]
mod tests;
