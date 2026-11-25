pub mod certificate;
mod coverage;
mod dashboard;

pub use coverage::{collect_llvm_cov_json, parse_coverage};
pub(crate) use dashboard::{print_dashboard, print_gate_failures};

use crate::whitelist::{
    VerificationDebtConfig, VerifySuiteConfig, WORKSPACE_WHITELIST_CANDIDATES,
    load_whitelist_strict,
};
use oxidize_core::ast_visitor::{AstVisitor, is_dedicated_test_path};
use oxidize_core::traceability::AstCache;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use syn::visit::Visit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportMode {
    Check,
    StrictParity,
    Write,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IntegrityReport {
    pub native_execution_coverage_pct: f64,
    pub wasm_path_coverage_pct: f64,
    pub total_assertion_density: f64,
    pub verified_modules_density: f64,
    pub unverified_modules_density: f64,
    pub total_production_assertion_density: f64,
    pub verified_production_assertion_density: f64,
    pub test_assertion_density: f64,
    pub passed: bool,
    #[serde(default)]
    pub certified_solvers: usize,
    #[serde(default)]
    pub max_interval_width: f64,
    #[serde(default)]
    pub tier_counts: BTreeMap<String, usize>,
    #[serde(default)]
    pub interval_bounds: BTreeMap<String, [f64; 2]>,
    #[serde(default)]
    pub lean4_proof_digests: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default)]
pub struct FileMetrics {
    pub wasm_paths: usize,
    pub wasm_covered: usize,
    pub total_funcs: f64,
    pub total_asserts: f64,
    pub verified_funcs: f64,
    pub verified_asserts: f64,
    pub unverified_funcs: f64,
    pub unverified_asserts: f64,
    pub prod_funcs: f64,
    pub prod_asserts: f64,
    pub verified_prod_funcs: f64,
    pub verified_prod_asserts: f64,
    pub test_funcs: f64,
    pub test_asserts: f64,
    pub opt_outs: Vec<(String, String)>,
    pub legacy_funcs: Vec<(String, String)>,
}

pub struct DebtSources<'a> {
    pub unverified_crates: &'a [String],
    pub file_length_debt: &'a [String],
    pub entropy_debt: &'a [String],
    pub extra_debt: &'a [String],
}

pub struct GateEvaluation {
    pub report: IntegrityReport,
    pub unauthorized_debt: Vec<String>,
    pub gate_errors: Vec<String>,
    pub authorized_opt_outs: usize,
    pub authorized_unverified_crates: usize,
}

pub(crate) fn load_suite_and_debt_configs_strict(
    explicit_whitelist: Option<&Path>,
) -> Result<(VerifySuiteConfig, Option<VerificationDebtConfig>), String> {
    load_whitelist_strict(explicit_whitelist, WORKSPACE_WHITELIST_CANDIDATES)
        .map(|wl| (wl.verify_suite.unwrap_or_default(), wl.verification_debt))
}

pub const VERIFY_SUITE_WRITE_ENV: &str = "OXIDIZE_VERIFY_SUITE_WRITE";

pub fn collect_rs_files() -> Vec<PathBuf> {
    crate::utils::collect_rs_files_filtered(&[Path::new(".")], &["target", ".git"])
}

fn normalize_report_path(filepath: &Path) -> String {
    let norm = crate::whitelist::lossy_slash_path(filepath);
    norm.strip_prefix("./").unwrap_or(&norm).to_string()
}

pub(crate) fn analyze_file_ast_into(
    filepath: &Path,
    ast: &syn::File,
    wasm_counts: (usize, usize),
    m: &mut FileMetrics,
) {
    let mut visitor = AstVisitor::with_test_scope(is_dedicated_test_path(filepath));
    visitor.visit_file(ast);

    let norm = normalize_report_path(filepath);
    for reason in visitor.opt_outs {
        m.opt_outs.push((norm.clone(), reason));
    }
    for fn_name in visitor.legacy_funcs {
        m.legacy_funcs.push((norm.clone(), fn_name));
    }

    let unverified_funcs = visitor.total_funcs.saturating_sub(visitor.verified_funcs);
    let unverified_asserts = visitor
        .total_asserts
        .saturating_sub(visitor.verified_asserts);

    m.total_funcs += visitor.total_funcs as f64;
    m.total_asserts += visitor.total_asserts as f64;
    m.verified_funcs += visitor.verified_funcs as f64;
    m.verified_asserts += visitor.verified_asserts as f64;
    m.unverified_funcs += unverified_funcs as f64;
    m.unverified_asserts += unverified_asserts as f64;
    m.prod_funcs += visitor.prod_funcs as f64;
    m.prod_asserts += visitor.prod_asserts as f64;
    m.verified_prod_funcs += visitor.verified_prod_funcs as f64;
    m.verified_prod_asserts += visitor.verified_prod_asserts as f64;
    m.test_funcs += visitor.test_funcs as f64;
    m.test_asserts += visitor.test_asserts as f64;
    m.wasm_paths += wasm_counts.0;
    m.wasm_covered += wasm_counts.1;
}

pub fn analyze_files_with_cache(rs_files: &[PathBuf], cache: &mut AstCache) -> FileMetrics {
    cache.preload_sources(rs_files);
    let mut m = FileMetrics::default();
    for filepath in rs_files {
        let wasm_counts =
            crate::wasm_coverage::count_file_wasm_coverage_with_cache(filepath, cache);
        if let Some(ast) = cache.get_or_load_fs(filepath).and_then(|e| e.ast.as_ref()) {
            analyze_file_ast_into(filepath, ast, wasm_counts, &mut m);
        }
    }
    m.opt_outs.sort();
    m.legacy_funcs.sort();
    m
}

pub fn find_unverified_crates_with_cache(members: &[String], cache: &mut AstCache) -> Vec<String> {
    let mut unverified = Vec::new();
    for fm in members {
        if fm == "egui_plot"
            || fm.ends_with("/egui_plot")
            || fm.ends_with("\\egui_plot")
            || fm == "crates/scientific_verification"
            || fm.ends_with("/crates/scientific_verification")
            || fm.ends_with("\\crates\\scientific_verification")
        {
            continue;
        }
        let mod_dir = Path::new(fm).join("src");
        if !mod_dir.exists() {
            continue;
        }
        let has_theory = crate::utils::collect_rs_files_filtered(&[&mod_dir], &["target"])
            .into_iter()
            .any(|path| {
                if is_dedicated_test_path(&path) {
                    return false;
                }
                let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
                    return false;
                };
                if oxidize_core::ast_visitor::attrs_require_cfg_test(&ast.attrs) {
                    return false;
                }
                let mut visitor = AstVisitor::new();
                visitor.visit_file(ast);
                visitor.has_verification_macro
            });
        if !has_theory {
            unverified.push(fm.clone());
        }
    }
    unverified.sort();
    unverified
}

#[cfg(test)]
pub fn check_unverified_modules(members: &[String]) -> Vec<String> {
    let mut cache = AstCache::new();
    find_unverified_crates_with_cache(members, &mut cache)
        .into_iter()
        .map(|fm| format!("Unverified Module: {fm}"))
        .collect()
}

fn ratio_or_zero(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn collect_debt_violations(
    m: &FileMetrics,
    debt: &DebtSources<'_>,
    config: &VerifySuiteConfig,
) -> (Vec<String>, usize, usize) {
    let mut unauthorized = Vec::new();
    let mut auth_opt_outs = 0usize;
    let mut auth_crates = 0usize;

    for (file, reason) in &m.opt_outs {
        if config.is_opt_out_allowed(file, reason) {
            auth_opt_outs += 1;
        } else {
            unauthorized.push(format!("Unauthorized opt_out in {file}: '{reason}'"));
        }
    }
    for (file, fn_name) in &m.legacy_funcs {
        unauthorized.push(format!("Legacy function in {file}: {fn_name}"));
    }
    for fm in debt.unverified_crates {
        if config.is_unverified_module_allowed(fm) {
            auth_crates += 1;
        } else {
            unauthorized.push(format!("Unwhitelisted unverified module: {fm}"));
        }
    }
    unauthorized.extend(debt.file_length_debt.iter().cloned());
    unauthorized.extend(debt.entropy_debt.iter().cloned());
    unauthorized.extend(debt.extra_debt.iter().cloned());
    (unauthorized, auth_opt_outs, auth_crates)
}

fn check_density_floor(label: &str, actual: f64, floor: f64, kind: &str, errors: &mut Vec<String>) {
    if !actual.is_finite() || actual < floor {
        errors.push(format!(
            "{label} {actual:.4} is below {kind} {floor:.4} asserts/fn"
        ));
    }
}

fn check_coverage_floor(label: &str, actual: f64, floor: f64, errors: &mut Vec<String>) {
    if !actual.is_finite() || !(0.0..=100.0).contains(&actual) {
        errors.push(format!(
            "{label} {actual:.2}% must be finite within 0.0%..=100.0%"
        ));
    } else if actual < floor {
        errors.push(format!(
            "{label} {actual:.2}% is below ratcheted floor {floor:.2}%"
        ));
    }
}

fn build_gate_errors(report: &IntegrityReport, config: &VerifySuiteConfig) -> Vec<String> {
    let mut errors = Vec::new();
    check_density_floor(
        "Verified modules density",
        report.verified_modules_density,
        config.min_verified_assertion_density,
        "required",
        &mut errors,
    );
    check_density_floor(
        "Total assertion density",
        report.total_assertion_density,
        config.min_total_assertion_density,
        "ratcheted floor",
        &mut errors,
    );
    check_density_floor(
        "Total production density",
        report.total_production_assertion_density,
        config.min_total_production_density,
        "ratcheted floor",
        &mut errors,
    );
    check_density_floor(
        "Verified production density",
        report.verified_production_assertion_density,
        config.min_verified_production_density,
        "ratcheted floor",
        &mut errors,
    );
    check_density_floor(
        "Test assertion density",
        report.test_assertion_density,
        config.min_test_assertion_density,
        "ratcheted floor",
        &mut errors,
    );
    check_coverage_floor(
        "WASM path coverage",
        report.wasm_path_coverage_pct,
        config.min_wasm_path_coverage_pct,
        &mut errors,
    );
    check_coverage_floor(
        "Native execution coverage",
        report.native_execution_coverage_pct,
        config.min_native_execution_coverage_pct,
        &mut errors,
    );
    errors
}

fn evaluate_pyramid_gate(gate_errors: &mut Vec<String>) -> crate::pyramid_cli::PyramidTelemetry {
    match crate::pyramid_cli::cached_pyramid_telemetry() {
        Ok(summary) => {
            let min_solvers = oxidize_core::traceability::CERTIFIED_SOLVER_COUNT;
            if summary.certified_solvers < min_solvers {
                gate_errors.push(format!(
                    "Certified solvers {} is below required {}",
                    summary.certified_solvers, min_solvers
                ));
            }
            if summary.tier_counts.len() != 6 || summary.tier_counts.values().any(|&c| c == 0) {
                gate_errors.push("Scientific verification pyramid missing active tiers".into());
            }
            summary
        }
        Err(err) => {
            gate_errors.push(format!("Scientific Verification Pyramid Failure: {err}"));
            crate::pyramid_cli::PyramidTelemetry::default()
        }
    }
}

pub fn evaluate_integrity_gates(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt: &DebtSources<'_>,
    config: &VerifySuiteConfig,
) -> GateEvaluation {
    let mut report = IntegrityReport {
        native_execution_coverage_pct: native_cov_pct,
        wasm_path_coverage_pct: wasm_cov_pct,
        total_assertion_density: ratio_or_zero(m.total_asserts, m.total_funcs),
        verified_modules_density: ratio_or_zero(m.verified_asserts, m.verified_funcs),
        unverified_modules_density: ratio_or_zero(m.unverified_asserts, m.unverified_funcs),
        total_production_assertion_density: ratio_or_zero(m.prod_asserts, m.prod_funcs),
        verified_production_assertion_density: ratio_or_zero(
            m.verified_prod_asserts,
            m.verified_prod_funcs,
        ),
        test_assertion_density: ratio_or_zero(m.test_asserts, m.test_funcs),
        passed: false,
        certified_solvers: 0,
        max_interval_width: 0.0,
        tier_counts: BTreeMap::new(),
        interval_bounds: BTreeMap::new(),
        lean4_proof_digests: BTreeMap::new(),
    };

    let (unauthorized_debt, authorized_opt_outs, authorized_unverified_crates) =
        collect_debt_violations(m, debt, config);
    let mut gate_errors = build_gate_errors(&report, config);
    let pyramid = evaluate_pyramid_gate(&mut gate_errors);
    report.certified_solvers = pyramid.certified_solvers;
    report.max_interval_width = pyramid.max_interval_width;
    report.tier_counts = pyramid.tier_counts;
    report.interval_bounds = pyramid.interval_bounds;
    report.lean4_proof_digests = pyramid.lean4_proof_digests;
    report.passed = gate_errors.is_empty() && unauthorized_debt.is_empty();

    GateEvaluation {
        report,
        unauthorized_debt,
        gate_errors,
        authorized_opt_outs,
        authorized_unverified_crates,
    }
}

fn apply_report_mode_io(
    report: &IntegrityReport,
    native_cov_pct: f64,
    tolerance_pct: f64,
    mode: ReportMode,
    output_cert: Option<&Path>,
) -> bool {
    if let Some(target_cert) = output_cert
        && let Err(err) = certificate::write_certificate_only(report, target_cert)
    {
        eprintln!("[!] {err}");
        return false;
    }

    let cert_path = Path::new(certificate::CERT_FILE_NAME);
    let readme_path = Path::new(certificate::README_FILE_NAME);
    match mode {
        ReportMode::Write => {
            if let Err(err) =
                certificate::write_certificate_and_readme(report, cert_path, readme_path)
            {
                eprintln!("[!] {err}");
                return false;
            }
            true
        }
        ReportMode::Check => certificate::run_check_certificate_io(
            report,
            cert_path,
            readme_path,
            "Verified Verification_Certificate.json schema and README.md badge in read-only --check mode.",
        ),
        ReportMode::StrictParity => certificate::run_strict_parity_io(
            report,
            Some(native_cov_pct),
            tolerance_pct,
            cert_path,
            readme_path,
        ),
    }
}

pub fn run_report_with_options(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt: &DebtSources<'_>,
    mode: ReportMode,
    output_cert: Option<&Path>,
) -> bool {
    let (config, debt_caps) = match load_suite_and_debt_configs_strict(None) {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };
    let mut combined_extra: Vec<String> = debt.extra_debt.to_vec();
    if let Some(cfg) = &debt_caps {
        combined_extra.extend(crate::whitelist::debt::evaluate_opt_out_caps(
            cfg,
            &m.opt_outs,
        ));
    }
    let merged_debt = DebtSources {
        unverified_crates: debt.unverified_crates,
        file_length_debt: debt.file_length_debt,
        entropy_debt: debt.entropy_debt,
        extra_debt: &combined_extra,
    };
    let eval = evaluate_integrity_gates(native_cov_pct, wasm_cov_pct, m, &merged_debt, &config);
    print_dashboard(&eval);
    if !eval.report.passed {
        return false;
    }
    apply_report_mode_io(
        &eval.report,
        native_cov_pct,
        config.coverage_tolerance_pct,
        mode,
        output_cert,
    )
}

#[cfg(test)]
pub fn print_report(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt_items: &[String],
) -> bool {
    let cfg = VerifySuiteConfig::default();
    let debt = DebtSources {
        unverified_crates: &[],
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: debt_items,
    };
    let eval = evaluate_integrity_gates(native_cov_pct, wasm_cov_pct, m, &debt, &cfg);
    print_dashboard(&eval);
    eval.report.passed
}

#[cfg(test)]
mod tests;
