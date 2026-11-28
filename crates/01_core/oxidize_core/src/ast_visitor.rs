mod attrs;
mod debt;
mod maturity;
mod pyramid;
mod vacuous;

pub use attrs::{attrs_mark_test_fn, attrs_require_cfg_test, extract_opt_out_reasons_from_attr};
pub use debt::collect_ast_file_debt;
pub use pyramid::FormalProofMacroBinding;

use attrs::{
    attrs_require_cfg_test_filtered, count_block_assertions, extract_opt_outs_filtered,
    non_fn_impl_item_attrs, non_fn_item_attrs, non_fn_trait_item_attrs, scan_attributes,
};
use maturity::{FileEvidenceScanner, collect_item_maturity_decls};
use scientific_metadata::theory::MaturityTier;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;
use syn::visit::{self, Visit};
use syn::{Attribute, ImplItem, ImplItemFn, Item, ItemFn, ItemMacro, TraitItem, TraitItemFn};
use vacuous::span_line;

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

/// Single-pass `syn` visitor collecting verification attributes, citations, and stochastic integrity metrics.
pub struct AstVisitor {
    /// Registered module keys verified via macros, `#[embed_theory]`, or `[cite:...]` annotations.
    pub verified_modules: Vec<String>,
    /// Verification tier (`Deterministic` or `Stochastic`) recorded per module key.
    pub module_tiers: HashMap<String, String>,
    /// Per-tier verification macro counts across the 6-tier scientific pyramid.
    pub tier_counts: BTreeMap<String, usize>,
    /// Structured `Tier6_Formal` proof and theorem bindings extracted from `formal_proof_verification!`.
    pub formal_proof_bindings: Vec<FormalProofMacroBinding>,
    /// First 1-indexed source line recorded for each verified module key.
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
    /// Whether a stochastic verification block contained a vacuous zero-initialized buffer.
    pub has_vacuous_bypass: bool,
    /// First 1-indexed source line where a vacuous zero-initialized bypass occurred.
    pub vacuous_bypass_line: Option<usize>,
    /// Whether the file invokes `theory_verification!` or `stochastic_signature_verification!`.
    pub has_verification_macro: bool,
    /// Name and 1-indexed line of the first function missing `#[verified]` when not opted out.
    pub first_unverified_fn: Option<(String, usize)>,
    /// Total function items visited in the file.
    pub total_funcs: usize,
    /// Total assertion macro invocations visited across functions and verification macros.
    pub total_asserts: usize,
    /// Count of functions carrying `#[verified]` or `#[embed_theory]`.
    pub verified_funcs: usize,
    /// Count of assertions occurring inside verified functions and verification macros.
    pub verified_asserts: usize,
    /// Total non-test production functions visited in the file.
    pub prod_funcs: usize,
    /// Total assertions recorded inside non-test production functions.
    pub prod_asserts: usize,
    /// Count of `#[verified]` non-test production functions.
    pub verified_prod_funcs: usize,
    /// Assertions recorded inside `#[verified]` non-test production functions.
    pub verified_prod_asserts: usize,
    /// Total test-scoped functions (`#[test]`, `#[cfg(test)]`, dedicated test files, or verification macros).
    pub test_funcs: usize,
    /// Total assertions recorded inside test-scoped functions and verification macros.
    pub test_asserts: usize,
    /// Count of `#[verified]` test-scoped functions and verification macro blocks.
    pub verified_test_funcs: usize,
    /// Assertions recorded inside `#[verified]` test-scoped functions and verification macro blocks.
    pub verified_test_asserts: usize,
    /// Count of functions or macros carrying semantic theory links or citations.
    pub semantic_integrity_funcs: usize,
    /// Count of functions explicitly exempted via `opt_out = "..."`.
    pub opted_out_funcs: usize,
    /// Out-of-line child module identifiers (`mod foo;`) discovered in the file.
    pub active_submodules: Vec<String>,
    /// Whether the active scope is opted out of verification tracking (`opt_out`).
    pub opted_out: bool,
    /// Ordered list of `opt_out` reason strings encountered across file and item attributes.
    pub opt_outs: Vec<String>,
    /// Non-test function identifiers ending with `_legacy` discovered in the AST.
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
    /// Constructs a zeroed [`AstVisitor`] ready to inspect a parsed `syn::File`.
    pub fn new() -> Self {
        Self::with_test_scope(false)
    }

    /// Creates a new [`AstVisitor`] with an explicit initial test-scope flag.
    pub fn with_test_scope(initial_test_scope: bool) -> Self {
        Self {
            verified_modules: Vec::new(),
            module_tiers: HashMap::new(),
            tier_counts: BTreeMap::new(),
            formal_proof_bindings: Vec::new(),
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
            has_verification_macro: false,
            first_unverified_fn: None,
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

    fn record_module(&mut self, name: String, line: usize) {
        if line > 0 {
            self.module_spans.entry(name.clone()).or_insert(line);
        }
        self.verified_modules.push(name);
    }

    fn accumulate_fn_counts(
        &mut self,
        assert_count: usize,
        effective_opt_out: bool,
        has_semantic: bool,
        is_verified: bool,
        fn_in_test: bool,
    ) {
        self.total_funcs += 1;
        self.total_asserts += assert_count;
        if effective_opt_out {
            self.opted_out_funcs += 1;
        } else {
            self.active_funcs += 1;
        }
        if has_semantic {
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
    }

    fn record_fn_metrics(
        &mut self,
        fn_ident: &syn::Ident,
        attrs: &[Attribute],
        assert_count: usize,
    ) -> (bool, bool) {
        let fn_name = fn_ident.to_string();
        let fn_in_test = self.in_test_scope || attrs_mark_test_fn(attrs);
        if !fn_in_test && fn_name.ends_with("_legacy") {
            self.legacy_funcs.push(fn_name.clone());
        }

        let scanned = scan_attributes(attrs);
        let fn_has_opt_out = !scanned.opt_outs.is_empty();
        self.opt_outs.extend(scanned.opt_outs);
        for (module_name, line) in scanned.modules_with_lines {
            self.record_module(module_name, line);
        }

        let is_verified = scanned.is_verified && !fn_has_opt_out;
        let effective_opt_out = fn_has_opt_out || (self.in_opt_out_scope && !is_verified);
        let child_opt_out_scope = self.in_opt_out_scope || fn_has_opt_out;
        if !fn_in_test && !effective_opt_out && !is_verified && self.first_unverified_fn.is_none() {
            self.first_unverified_fn = Some((fn_name, span_line(fn_ident.span())));
        }

        self.accumulate_fn_counts(
            assert_count,
            effective_opt_out,
            scanned.has_semantic,
            is_verified,
            fn_in_test,
        );
        (child_opt_out_scope, fn_in_test)
    }
}

impl<'ast> Visit<'ast> for AstVisitor {
    fn visit_file(&mut self, node: &'ast syn::File) {
        let reasons = extract_opt_outs_filtered(&node.attrs, true, false);
        if !reasons.is_empty() {
            self.opted_out = true;
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_require_cfg_test_filtered(&node.attrs, true, false) {
            self.in_test_scope = true;
        }
        let mut scanner = FileEvidenceScanner::default();
        scanner.visit_file(node);
        self.has_zero_alloc_proof |= scanner.has_zero_alloc_proof;
        self.has_seeded_rng |= scanner.has_seeded_rng;
        self.has_unseeded_rng |= scanner.has_unseeded_rng;
        self.has_golden_fixture |= scanner.has_golden_fixture;
        self.maturity_asserts += scanner.maturity_asserts;

        visit::visit_file(self, node);
    }

    fn visit_item(&mut self, node: &'ast Item) {
        let Some(attrs) = non_fn_item_attrs(node) else {
            visit::visit_item(self, node);
            return;
        };
        let allow_inner = matches!(node, Item::Mod(m) if m.content.is_some());
        let reasons = extract_opt_outs_filtered(attrs, allow_inner, true);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        if !reasons.is_empty() {
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_require_cfg_test_filtered(attrs, allow_inner, true) {
            self.in_test_scope = true;
        }
        if !self.in_test_scope {
            collect_item_maturity_decls(node, &mut self.declared_item_maturities);
        }
        visit::visit_item(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_impl_item(&mut self, node: &'ast ImplItem) {
        let Some(attrs) = non_fn_impl_item_attrs(node) else {
            visit::visit_impl_item(self, node);
            return;
        };
        let reasons = extract_opt_outs_filtered(attrs, false, true);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        if !reasons.is_empty() {
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_require_cfg_test_filtered(attrs, false, true) {
            self.in_test_scope = true;
        }
        visit::visit_impl_item(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_trait_item(&mut self, node: &'ast TraitItem) {
        let Some(attrs) = non_fn_trait_item_attrs(node) else {
            visit::visit_trait_item(self, node);
            return;
        };
        let reasons = extract_opt_outs_filtered(attrs, false, true);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        if !reasons.is_empty() {
            self.in_opt_out_scope = true;
            self.opt_outs.extend(reasons);
        }
        if attrs_require_cfg_test_filtered(attrs, false, true) {
            self.in_test_scope = true;
        }
        visit::visit_trait_item(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        if !self.in_test_scope && !attrs_require_cfg_test(&node.attrs) && node.content.is_none() {
            self.active_submodules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        pyramid::record_verification_macro(self, node);
        visit::visit_item_macro(self, node);
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let asserts = count_block_assertions(&node.block);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        let (eff_opt, eff_test) = self.record_fn_metrics(&node.sig.ident, &node.attrs, asserts);
        self.in_opt_out_scope = eff_opt;
        self.in_test_scope = eff_test;
        visit::visit_item_fn(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_impl_item_fn(&mut self, node: &'ast ImplItemFn) {
        let asserts = count_block_assertions(&node.block);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        let (eff_opt, eff_test) = self.record_fn_metrics(&node.sig.ident, &node.attrs, asserts);
        self.in_opt_out_scope = eff_opt;
        self.in_test_scope = eff_test;
        visit::visit_impl_item_fn(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }

    fn visit_trait_item_fn(&mut self, node: &'ast TraitItemFn) {
        let name = node.sig.ident.to_string();
        let Some(ref block) = node.default else {
            let fn_in_test = self.in_test_scope || attrs_mark_test_fn(&node.attrs);
            if !fn_in_test && name.ends_with("_legacy") {
                self.legacy_funcs.push(name);
            }
            self.opt_outs
                .extend(extract_opt_outs_filtered(&node.attrs, false, true));
            visit::visit_trait_item_fn(self, node);
            return;
        };
        let asserts = count_block_assertions(block);
        let prev_opt = self.in_opt_out_scope;
        let prev_test = self.in_test_scope;
        let (eff_opt, eff_test) = self.record_fn_metrics(&node.sig.ident, &node.attrs, asserts);
        self.in_opt_out_scope = eff_opt;
        self.in_test_scope = eff_test;
        visit::visit_trait_item_fn(self, node);
        self.in_opt_out_scope = prev_opt;
        self.in_test_scope = prev_test;
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "ast_visitor_tests.rs"]
mod pr_tests;
