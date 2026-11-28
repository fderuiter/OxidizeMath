//! Qualification gates verifying AST evidence for `Experimental`, `Validated`, and `Certified` tiers.

#[path = "qualification_proof.rs"]
mod proof_gate;

use super::AstCache;
use super::analysis::extract_citations_with_lines;
use super::maturity::{EXPECTED_WORKSPACE_CRATES, MaturityRegistry};
use crate::ast_visitor::AstVisitor;
use scientific_metadata::theory::MaturityTier;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// Minimum assertion-to-function ratio required for `Validated` and `Certified` qualification.
pub const VALIDATED_MIN_ASSERTION_DENSITY: f64 = 0.40;

/// Aggregated static AST evidence for a single crate or registered module.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MaturityEvidence {
    /// Number of Rust files mapped to this crate or module.
    pub files_scanned: usize,
    /// Number of files failing `syn::parse_file`.
    pub syntax_errors: usize,
    /// Total non-opted-out functions counted across mapped files.
    pub total_funcs: usize,
    /// Total functions (including `#[verified(opt_out)]` infrastructure functions) when `total_funcs == 0`.
    pub raw_funcs: usize,
    /// Total assertions (`assert*`, `assert_relative_eq!`, `assert_abs_diff_eq!`) across mapped files.
    pub maturity_asserts: usize,
    /// Whether a non-vacuous `theory_verification!` deterministic contract exists for this target.
    pub has_deterministic_contract: bool,
    /// Whether a vacuous `theory_verification!` (`test = {}`) was found.
    pub has_vacuous_contract: bool,
    /// Whether `stochastic_signature_verification!` was used on this target.
    pub has_stochastic_contract: bool,
    /// Whether `VerifiedAllocator`, `VerifiedEngine::run_verified`, or `lock_allocations` was proven in AST.
    pub has_zero_alloc_proof: bool,
    /// Whether seeded RNG (`OxidizeRng`, `ChaCha8Rng`, `from_seed`, `seed_from_u64`) or deterministic contract is present.
    pub has_seeded_determinism: bool,
    /// Whether `thread_rng` was invoked in AST.
    pub has_unseeded_rng: bool,
    /// Whether golden reference constants, equality fixtures, or embedded test vectors exist.
    pub has_golden_fixture: bool,
    /// Whether a valid `Tier6_Formal` `formal_proof_verification!` binding to a compiled Lean theorem exists.
    pub has_formal_proof: bool,
}

impl MaturityEvidence {
    /// Returns effective function count (`total_funcs` when non-zero, otherwise `raw_funcs`).
    pub fn effective_funcs(&self) -> usize {
        if self.total_funcs > 0 {
            self.total_funcs
        } else {
            self.raw_funcs
        }
    }

    /// Returns `maturity_asserts / effective_funcs()`, or `0.0` when `effective_funcs() == 0`.
    pub fn assertion_density(&self) -> f64 {
        let denom = self.effective_funcs();
        if denom == 0 {
            0.0
        } else {
            self.maturity_asserts as f64 / denom as f64
        }
    }
}

/// Complete result of running workspace maturity qualification gates.
#[derive(Clone, Debug, Default)]
pub struct MaturityGateReport {
    /// Parsed `traceability.toml` maturity registry.
    pub registry: MaturityRegistry,
    /// Collected AST evidence per workspace crate.
    pub crate_evidence: BTreeMap<String, MaturityEvidence>,
    /// Collected AST evidence per `[links]` module.
    pub module_evidence: BTreeMap<String, MaturityEvidence>,
    /// Actionable qualification failure messages; empty when all gates pass.
    pub violations: Vec<String>,
}

impl MaturityGateReport {
    /// Returns `true` when zero qualification violations were detected.
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

fn collect_rs_files_under(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            collect_rs_files_under(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Discovers all workspace Rust files inspected by maturity qualification (`src/` and `tests/`).
pub fn discover_maturity_scan_files(workspace_root: &Path) -> Vec<PathBuf> {
    if let Ok(packages) = crate::workspace::discover_workspace_packages(workspace_root) {
        let mut files = Vec::new();
        for pkg in packages {
            if pkg.is_vendored_patch {
                continue;
            }
            let member_root = workspace_root.join(&pkg.member_path);
            if pkg.effective_tier() == Some(crate::workspace::TierKind::Apps) {
                if pkg.participates_in_traceability() {
                    collect_rs_files_under(
                        &member_root.join(pkg.traceability_subdir()),
                        &mut files,
                    );
                }
            } else {
                collect_rs_files_under(&member_root.join("src"), &mut files);
                collect_rs_files_under(&member_root.join("tests"), &mut files);
            }
        }
        files.sort();
        files.dedup();
        return files;
    }

    let mut files = Vec::new();
    let crates_dir = workspace_root.join("crates");
    if let Ok(entries) = std::fs::read_dir(&crates_dir) {
        let mut crate_dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        crate_dirs.sort();
        for crate_dir in crate_dirs {
            if !crate_dir.is_dir() {
                continue;
            }
            if crate_dir.join("Cargo.toml").is_file() {
                collect_rs_files_under(&crate_dir.join("src"), &mut files);
                collect_rs_files_under(&crate_dir.join("tests"), &mut files);
            } else if let Ok(nested_entries) = std::fs::read_dir(&crate_dir) {
                let mut nested_dirs: Vec<PathBuf> =
                    nested_entries.flatten().map(|e| e.path()).collect();
                nested_dirs.sort();
                for nested in nested_dirs {
                    if nested.is_dir() && nested.join("Cargo.toml").is_file() {
                        collect_rs_files_under(&nested.join("src"), &mut files);
                        collect_rs_files_under(&nested.join("tests"), &mut files);
                    }
                }
            }
        }
    }
    files
}

fn check_registry_completeness(registry: &MaturityRegistry, violations: &mut Vec<String>) {
    violations.extend(registry.invalid_entries.iter().cloned());
    for &expected_crate in EXPECTED_WORKSPACE_CRATES {
        if !registry.crates.contains_key(expected_crate) {
            violations.push(format!(
                "Missing `[crates].{expected_crate}` maturity entry in `traceability.toml`."
            ));
        }
    }
    for link_mod in registry.links.keys() {
        if !registry.modules.contains_key(link_mod) {
            violations.push(format!(
                "Missing `[modules].{link_mod}` maturity entry in `traceability.toml` for `[links]` key `{link_mod}`."
            ));
        }
    }
    for mod_key in registry.modules.keys() {
        if !registry.links.contains_key(mod_key) {
            violations.push(format!(
                "`[modules].{mod_key}` in `traceability.toml` has no matching key in `[links]`."
            ));
        }
    }
}

fn record_visitor_into_crate(
    ev: &mut MaturityEvidence,
    visitor: &AstVisitor,
    has_valid_formal_proof: bool,
) {
    ev.files_scanned += 1;
    ev.total_funcs += visitor.active_funcs;
    ev.raw_funcs += visitor.total_funcs;
    ev.maturity_asserts += visitor.maturity_asserts;
    ev.has_deterministic_contract |= !visitor.non_vacuous_deterministic_modules.is_empty();
    ev.has_vacuous_contract |= !visitor.vacuous_deterministic_modules.is_empty();
    ev.has_stochastic_contract |= !visitor.stochastic_modules.is_empty();
    ev.has_zero_alloc_proof |= visitor.has_zero_alloc_proof;
    ev.has_seeded_determinism |= visitor.has_seeded_rng || !visitor.has_unseeded_rng;
    ev.has_unseeded_rng |= visitor.has_unseeded_rng;
    ev.has_golden_fixture |= visitor.has_golden_fixture;
    ev.has_formal_proof |= has_valid_formal_proof;
}

fn record_visitor_into_module(
    ev: &mut MaturityEvidence,
    mod_name: &str,
    path_mods: &[String],
    visitor: &AstVisitor,
    valid_formal_modules: &BTreeSet<String>,
) {
    ev.files_scanned += 1;
    ev.total_funcs += visitor.active_funcs;
    ev.raw_funcs += visitor.total_funcs;
    ev.maturity_asserts += visitor.maturity_asserts;
    if visitor
        .non_vacuous_deterministic_modules
        .iter()
        .any(|x| x == mod_name)
    {
        ev.has_deterministic_contract = true;
    }
    if visitor
        .vacuous_deterministic_modules
        .iter()
        .any(|(x, _)| x == mod_name)
    {
        ev.has_vacuous_contract = true;
    }
    if visitor.stochastic_modules.iter().any(|x| x == mod_name) {
        ev.has_stochastic_contract = true;
    }
    if path_mods.iter().any(|pm| pm == mod_name) {
        ev.has_zero_alloc_proof |= visitor.has_zero_alloc_proof;
    }
    ev.has_seeded_determinism |= visitor.has_seeded_rng || !visitor.has_unseeded_rng;
    ev.has_unseeded_rng |= visitor.has_unseeded_rng;
    ev.has_golden_fixture |= visitor.has_golden_fixture;
    if valid_formal_modules.contains(mod_name) {
        ev.has_formal_proof = true;
    }
}

fn audit_file_items_and_proofs(
    rel_path: &str,
    visitor: &AstVisitor,
    registry: &MaturityRegistry,
    proof_catalog: &proof_gate::WorkspaceProofCatalog,
    violations: &mut Vec<String>,
) -> BTreeSet<String> {
    let mut valid_formal_modules = BTreeSet::new();
    if rel_path.contains("/tests/") {
        return valid_formal_modules;
    }
    let enclosing_tier = registry.tier_for_repo_path(rel_path);
    for (item_name, item_tier, line) in &visitor.declared_item_maturities {
        if *item_tier > enclosing_tier {
            violations.push(format!(
                "{rel_path}:{line}: `{item_name}` declares `{item_tier}` maturity inside `{enclosing_tier}` module; promote the module in `traceability.toml` with required qualification proofs first."
            ));
        }
    }
    for binding in &visitor.formal_proof_bindings {
        if proof_catalog.validate_binding(rel_path, binding, violations) {
            valid_formal_modules.insert(binding.module.clone());
        }
    }
    valid_formal_modules
}

fn accumulate_file_evidence(
    rel_path: &str,
    content: &str,
    ast: Option<&syn::File>,
    registry: &MaturityRegistry,
    proof_catalog: &proof_gate::WorkspaceProofCatalog,
    report: &mut MaturityGateReport,
) {
    let crate_name = MaturityRegistry::crate_for_repo_path(rel_path).map(str::to_string);
    let path_mods = registry.path_owned_modules(rel_path);
    let mut all_mods: BTreeSet<String> = path_mods.iter().cloned().collect();
    for (cite, _) in extract_citations_with_lines(content) {
        if registry.modules.contains_key(&cite) {
            all_mods.insert(cite);
        }
    }

    let Some(parsed) = ast else {
        if let Some(c) = &crate_name {
            let ev = report.crate_evidence.entry(c.clone()).or_default();
            ev.files_scanned += 1;
            ev.syntax_errors += 1;
        }
        for m in all_mods {
            let ev = report.module_evidence.entry(m).or_default();
            ev.files_scanned += 1;
            ev.syntax_errors += 1;
        }
        return;
    };

    let mut visitor = AstVisitor::new();
    syn::visit::Visit::visit_file(&mut visitor, parsed);
    for vm in &visitor.verified_modules {
        if registry.modules.contains_key(vm) {
            all_mods.insert(vm.clone());
        }
    }

    let valid_formal_modules = audit_file_items_and_proofs(
        rel_path,
        &visitor,
        registry,
        proof_catalog,
        &mut report.violations,
    );

    if crate_name.as_deref() == Some("math_explorer_gui")
        || rel_path.starts_with("math_explorer_gui/")
        || rel_path.starts_with("apps/math_explorer_gui/")
    {
        return;
    }
    if let Some(c) = crate_name {
        record_visitor_into_crate(
            report.crate_evidence.entry(c).or_default(),
            &visitor,
            !valid_formal_modules.is_empty(),
        );
    }
    for m in &all_mods {
        record_visitor_into_module(
            report.module_evidence.entry(m.clone()).or_default(),
            m,
            &path_mods,
            &visitor,
            &valid_formal_modules,
        );
    }
}

fn verify_target_qualification(
    kind: &str,
    name: &str,
    tier: MaturityTier,
    ev: &MaturityEvidence,
    violations: &mut Vec<String>,
) {
    if ev.syntax_errors > 0 {
        violations.push(format!(
            "[{kind} `{name}` ({tier})]: {} file(s) failed Rust AST syntax parsing.",
            ev.syntax_errors
        ));
    }
    if tier == MaturityTier::Experimental {
        return;
    }
    let funcs = ev.effective_funcs();
    if ev.files_scanned == 0 || funcs == 0 {
        violations.push(format!(
            "[{kind} `{name}` ({tier})]: no active functions scanned (files={}, funcs={funcs}); stub targets must stay `Experimental`.",
            ev.files_scanned
        ));
        return;
    }
    let density = ev.assertion_density();
    if density + 1e-9 < VALIDATED_MIN_ASSERTION_DENSITY {
        violations.push(format!(
            "[{kind} `{name}` ({tier})]: assertion density {density:.2} ({}/{funcs} asserts/fn) is below required {:.2}.",
            ev.maturity_asserts, VALIDATED_MIN_ASSERTION_DENSITY
        ));
    }
    if ev.has_unseeded_rng {
        violations.push(format!(
            "[{kind} `{name}` ({tier})]: unseeded `thread_rng` detected; `{tier}` requires seeded determinism (`OxidizeRng` / `ChaCha8Rng`)."
        ));
    }
    if !ev.has_golden_fixture {
        violations.push(format!(
            "[{kind} `{name}` ({tier})]: missing golden reference constants or equality fixtures."
        ));
    }
    if tier == MaturityTier::Certified {
        if !ev.has_deterministic_contract {
            violations.push(format!(
                "[{kind} `{name}` (Certified)]: missing non-vacuous deterministic `theory_verification!` contract."
            ));
        }
        if ev.has_stochastic_contract {
            violations.push(format!(
                "[{kind} `{name}` (Certified)]: uses `stochastic_signature_verification!`; `Certified` requires deterministic contracts."
            ));
        }
        if !ev.has_zero_alloc_proof {
            violations.push(format!(
                "[{kind} `{name}` (Certified)]: missing zero-allocation proof (`VerifiedAllocator` / `VerifiedEngine::run_verified`)."
            ));
        }
        if !ev.has_formal_proof {
            violations.push(format!(
                "[{kind} `{name}` (Certified)]: missing valid `Tier6_Formal` `formal_proof_verification!` binding to a compiled Lean 4 theorem."
            ));
        }
    }
}

/// Evaluates maturity qualification gates against `workspace_root` using `registry` and shared `cache`.
pub fn evaluate_maturity_with_registry_and_cache(
    workspace_root: &Path,
    registry: MaturityRegistry,
    cache: &mut AstCache,
) -> MaturityGateReport {
    let mut report = MaturityGateReport {
        registry,
        ..MaturityGateReport::default()
    };
    check_registry_completeness(&report.registry, &mut report.violations);

    let proof_catalog = proof_gate::WorkspaceProofCatalog::load(workspace_root);
    let files = discover_maturity_scan_files(workspace_root);
    cache.preload_sources(&files);
    let registry_snapshot = report.registry.clone();
    for abs_path in &files {
        let rel = abs_path
            .strip_prefix(workspace_root)
            .unwrap_or(abs_path.as_path());
        let rel_norm = crate::path_utils::normalize_path(rel);
        if let Some(entry) = cache.get_or_load_fs(abs_path) {
            let content = entry.content.clone();
            let ast_clone = entry.ast.clone();
            accumulate_file_evidence(
                &rel_norm,
                &content,
                ast_clone.as_ref(),
                &registry_snapshot,
                &proof_catalog,
                &mut report,
            );
        }
    }

    for (crate_name, &tier) in &registry_snapshot.crates {
        let ev = report
            .crate_evidence
            .get(crate_name)
            .cloned()
            .unwrap_or_default();
        verify_target_qualification("crate", crate_name, tier, &ev, &mut report.violations);
    }
    for (mod_name, &tier) in &registry_snapshot.modules {
        let ev = report
            .module_evidence
            .get(mod_name)
            .cloned()
            .unwrap_or_default();
        verify_target_qualification("module", mod_name, tier, &ev, &mut report.violations);
    }
    report
}

/// Evaluates workspace maturity qualification gates loading `traceability.toml` from `workspace_root` and reusing `cache`.
pub fn evaluate_workspace_maturity_with_cache(
    workspace_root: &Path,
    cache: &mut AstCache,
) -> MaturityGateReport {
    let registry = MaturityRegistry::load_or_embedded(Some(workspace_root));
    evaluate_maturity_with_registry_and_cache(workspace_root, registry, cache)
}

/// Evaluates workspace maturity qualification gates with a fresh [`AstCache`].
pub fn evaluate_workspace_maturity(workspace_root: &Path) -> MaturityGateReport {
    let mut cache = AstCache::new();
    evaluate_workspace_maturity_with_cache(workspace_root, &mut cache)
}
