mod maturity;
mod stochastic;

use maturity::{
    FileEvidenceScanner, extract_impl_maturity_method, extract_item_theory_maturity,
    inspect_theory_verification_macro,
};
use scientific_metadata::theory::MaturityTier;
use std::collections::HashMap;
use stochastic::{find_stochastic_zero_init_line, span_line};
use syn::spanned::Spanned;
use syn::visit::{self, Visit};
use syn::{ImplItemFn, ItemFn, ItemMacro, Macro, TraitItemFn};

#[allow(missing_docs)]
pub struct AstVisitor {
    #[allow(missing_docs)]
    pub verified_modules: Vec<String>,
    #[allow(missing_docs)]
    pub module_tiers: HashMap<String, String>,
    #[allow(missing_docs)]
    pub module_spans: HashMap<String, usize>,
    /// Modules with a non-vacuous `theory_verification!` deterministic contract (`test = { ... }`).
    pub non_vacuous_deterministic_modules: Vec<String>,
    /// Modules with an empty `theory_verification!` test block and their source line numbers.
    pub vacuous_deterministic_modules: Vec<(String, usize)>,
    /// Modules verified via `stochastic_signature_verification!`.
    pub stochastic_modules: Vec<String>,
    /// Explicit item-level `MaturityTier` declarations (`(type_name, tier, line)`).
    pub declared_item_maturities: Vec<(String, MaturityTier, usize)>,
    /// Whether this file contains a `VerifiedAllocator` or `VerifiedEngine::run_verified` proof.
    pub has_zero_alloc_proof: bool,
    /// Whether this file initializes a deterministic seeded RNG or deterministic contract.
    pub has_seeded_rng: bool,
    /// Whether this file invokes unseeded `thread_rng`.
    pub has_unseeded_rng: bool,
    /// Whether this file defines golden constants, equality assertions, or fixture includes.
    pub has_golden_fixture: bool,
    /// Total assertion macros counted for maturity density qualification.
    pub maturity_asserts: usize,
    /// Total non-opted-out functions counted for maturity density qualification.
    pub active_funcs: usize,
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
    file_opted_out: bool,
    in_cfg_test: bool,
}

impl Default for AstVisitor {
    fn default() -> Self {
        Self::new()
    }
}

fn attr_has_opt_out(attr: &syn::Attribute) -> bool {
    if attr.path().is_ident("doc") {
        return false;
    }
    quote::quote!(#attr)
        .to_string()
        .replace(' ', "")
        .contains("opt_out")
}

fn attr_is_cfg_test(attr: &syn::Attribute) -> bool {
    if !attr.path().is_ident("cfg") {
        return false;
    }
    quote::quote!(#attr)
        .to_string()
        .replace(' ', "")
        .contains("cfg(test)")
}

fn item_attrs(node: &syn::Item) -> &[syn::Attribute] {
    match node {
        syn::Item::Struct(i) => &i.attrs,
        syn::Item::Enum(i) => &i.attrs,
        syn::Item::Union(i) => &i.attrs,
        syn::Item::Impl(i) => &i.attrs,
        syn::Item::Fn(i) => &i.attrs,
        syn::Item::Mod(i) => &i.attrs,
        syn::Item::Trait(i) => &i.attrs,
        syn::Item::Const(i) => &i.attrs,
        syn::Item::Static(i) => &i.attrs,
        syn::Item::Type(i) => &i.attrs,
        syn::Item::Macro(i) => &i.attrs,
        _ => &[],
    }
}

impl AstVisitor {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        Self {
            verified_modules: Vec::new(),
            module_tiers: HashMap::new(),
            module_spans: HashMap::new(),
            non_vacuous_deterministic_modules: Vec::new(),
            vacuous_deterministic_modules: Vec::new(),
            stochastic_modules: Vec::new(),
            declared_item_maturities: Vec::new(),
            has_zero_alloc_proof: false,
            has_seeded_rng: false,
            has_unseeded_rng: false,
            has_golden_fixture: false,
            maturity_asserts: 0,
            active_funcs: 0,
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
            file_opted_out: false,
            in_cfg_test: false,
        }
    }

    fn record_module(&mut self, name: String, line: usize) {
        if line > 0 {
            self.module_spans.entry(name.clone()).or_insert(line);
        }
        self.verified_modules.push(name);
    }

    fn extract_embed_theory(&mut self, attr_str: &str, line: usize) {
        let s = attr_str.replace('\\', "/");
        if let Some((pos, end)) = s
            .find('"')
            .and_then(|p| s[p + 1..].find('"').map(|e| (p, e)))
        {
            let raw_path = &s[pos + 1..pos + 1 + end];
            let clean_name = raw_path
                .trim_start_matches("papers/")
                .trim_start_matches("spec:")
                .trim_start_matches("registry:")
                .trim_end_matches(".tex");
            if !clean_name.is_empty() {
                self.record_module(clean_name.to_string(), line);
            }
        }
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

        let attr_str = quote::quote!(#attr).to_string().replace(' ', "");
        let mut is_verified = attr_str.contains("verified_engine::verified")
            || attr_str.contains("#[verified]")
            || attr_str.starts_with("#[verified(");
        if attr_str.contains("embed_theory") {
            is_verified = true;
            self.extract_embed_theory(&attr_str, attr_line);
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
        } else {
            self.active_funcs += 1;
            if !is_verified && self.first_unverified_fn.is_none() {
                self.first_unverified_fn = Some((fn_ident.to_string(), span_line(fn_ident.span())));
            }
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

    fn record_verification_macro(&mut self, name: &str, node: &ItemMacro) {
        let tier_name = match name {
            "theory_verification" => "Deterministic",
            "stochastic_signature_verification" => "Stochastic",
            _ => return,
        };
        let macro_line = span_line(node.span());
        if name == "stochastic_signature_verification"
            && let Some(zero_line) = find_stochastic_zero_init_line(&node.mac)
        {
            self.has_vacuous_bypass = true;
            let recorded = if zero_line > 0 { zero_line } else { macro_line };
            if self.vacuous_bypass_line.is_none() && recorded > 0 {
                self.vacuous_bypass_line = Some(recorded);
            }
        }
        let tv_info =
            (name == "theory_verification").then(|| inspect_theory_verification_macro(&node.mac));
        let tokens = node.mac.tokens.to_string();
        let Some(idx) = tokens.find("module = \"") else {
            return;
        };
        let start = idx + 10;
        let Some(end) = tokens[start..].find('"') else {
            return;
        };
        let module_name = tokens[start..start + end].to_string();
        self.record_module(module_name.clone(), macro_line);
        self.module_tiers
            .insert(module_name.clone(), tier_name.to_string());
        if let Some(info) = tv_info {
            if info.has_non_empty_constants {
                self.has_golden_fixture = true;
            }
            if info.has_non_empty_test {
                self.non_vacuous_deterministic_modules.push(module_name);
                self.has_seeded_rng = true;
            } else {
                self.vacuous_deterministic_modules
                    .push((module_name, macro_line));
            }
        } else {
            self.stochastic_modules.push(module_name);
        }
        self.semantic_integrity_funcs += 1;
        self.total_funcs += 1;
        self.active_funcs += 1;
        self.verified_funcs += 1;
        let assert_count = tokens.matches("assert").count().max(3);
        self.total_asserts += assert_count;
        self.verified_asserts += assert_count;
        self.maturity_asserts += tokens.matches("assert").count();
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_file(&mut self, node: &'ast syn::File) {
        if node
            .attrs
            .iter()
            .any(|a| matches!(a.style, syn::AttrStyle::Inner(_)) && attr_has_opt_out(a))
        {
            self.file_opted_out = true;
            self.opted_out = true;
        }
        let mut scanner = FileEvidenceScanner::default();
        scanner.visit_file(node);
        self.has_zero_alloc_proof |= scanner.has_zero_alloc_proof;
        self.has_seeded_rng |= scanner.has_seeded_rng;
        self.has_unseeded_rng |= scanner.has_unseeded_rng;
        self.has_golden_fixture |= scanner.has_golden_fixture;
        self.maturity_asserts += scanner.maturity_asserts;

        visit::visit_file(self, node);
        self.opted_out = self.file_opted_out;
    }

    fn visit_item(&mut self, node: &'ast syn::Item) {
        let prev = self.opted_out;
        let prev_cfg_test = self.in_cfg_test;
        let attrs = item_attrs(node);
        let allow_inner = matches!(node, syn::Item::Mod(m) if m.content.is_some());
        if attrs.iter().any(|a| {
            (matches!(a.style, syn::AttrStyle::Outer) || allow_inner) && attr_has_opt_out(a)
        }) {
            self.opted_out = true;
        }
        if attrs.iter().any(attr_is_cfg_test) {
            self.in_cfg_test = true;
        }
        if !self.in_cfg_test {
            match node {
                syn::Item::Struct(s) => {
                    if let Some(decl) = extract_item_theory_maturity(&s.ident.to_string(), &s.attrs)
                    {
                        self.declared_item_maturities.push(decl);
                    }
                }
                syn::Item::Enum(e) => {
                    if let Some(decl) = extract_item_theory_maturity(&e.ident.to_string(), &e.attrs)
                    {
                        self.declared_item_maturities.push(decl);
                    }
                }
                syn::Item::Impl(imp) if imp.trait_.is_some() => {
                    for item in &imp.items {
                        if let syn::ImplItem::Fn(m) = item
                            && let Some(decl) = extract_impl_maturity_method(&imp.self_ty, m)
                        {
                            self.declared_item_maturities.push(decl);
                        }
                    }
                }
                _ => {}
            }
        }
        visit::visit_item(self, node);
        self.opted_out = prev;
        self.in_cfg_test = prev_cfg_test;
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if node.content.is_none() {
            self.active_submodules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        if let Some(ident) = node.mac.path.segments.last().map(|s| &s.ident) {
            self.record_verification_macro(&ident.to_string(), node);
        }
        visit::visit_item_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
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

#[cfg(test)]
mod tests;
