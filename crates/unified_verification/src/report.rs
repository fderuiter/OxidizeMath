pub mod certificate;

use crate::whitelist::{VerifySuiteConfig, WORKSPACE_WHITELIST_CANDIDATES, load_whitelist_strict};
use oxidize_core::ast_visitor::{AstVisitor, is_dedicated_test_path};
use oxidize_core::traceability::AstCache;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use syn::visit::Visit;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReportMode {
    Check,
    Write,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IntegrityReport {
    pub native_execution_coverage_pct: f64,
    pub wasm_path_coverage_pct: f64,
    pub total_assertion_density: f64,
    pub verified_modules_density: f64,
    pub unverified_modules_density: f64,
    #[serde(default)]
    pub total_production_assertion_density: f64,
    #[serde(default)]
    pub verified_production_assertion_density: f64,
    #[serde(default)]
    pub test_assertion_density: f64,
    pub passed: bool,
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

pub fn load_verify_suite_config_strict(
    explicit_whitelist: Option<&Path>,
) -> Result<VerifySuiteConfig, String> {
    load_whitelist_strict(explicit_whitelist, WORKSPACE_WHITELIST_CANDIDATES)
        .map(|wl| wl.verify_suite.unwrap_or_default())
}

pub const VERIFY_SUITE_WRITE_ENV: &str = "OXIDIZE_VERIFY_SUITE_WRITE";

pub fn get_llvm_cov_output(mode: ReportMode) -> std::process::Output {
    let mut cmd = Command::new("cargo");
    cmd.args(["llvm-cov", "--all-features", "--workspace", "--json"]);
    if mode == ReportMode::Write {
        cmd.env(VERIFY_SUITE_WRITE_ENV, "1");
    }
    cmd.output().expect("Failed to run cargo llvm-cov")
}

pub fn collect_rs_files() -> Vec<PathBuf> {
    crate::utils::collect_rs_files_filtered(&[Path::new(".")], &["target", ".git"])
}

pub fn parse_coverage(cov_json: &serde_json::Value) -> (f64, f64) {
    let mut native_lines_total = 0.0;
    let mut native_lines_covered = 0.0;

    if let Some(files) = cov_json
        .get("data")
        .and_then(|d| d.as_array())
        .and_then(|a| a.first())
        .and_then(|data| data.get("files"))
        .and_then(|f| f.as_array())
    {
        for f in files {
            if let Some(summary) = f.get("summary").and_then(|s| s.get("lines")) {
                native_lines_total += summary.get("count").and_then(|c| c.as_f64()).unwrap_or(0.0);
                native_lines_covered += summary
                    .get("covered")
                    .and_then(|c| c.as_f64())
                    .unwrap_or(0.0);
            }
        }
    }
    (native_lines_total, native_lines_covered)
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
        if fm == "egui_plot" {
            continue;
        }
        let mod_dir = Path::new(fm).join("src");
        if !mod_dir.exists() {
            continue;
        }
        let has_theory = crate::utils::collect_rs_files_filtered(&[&mod_dir], &["target"])
            .into_iter()
            .any(|path| {
                let Some(ast) = cache.get_or_load_fs(&path).and_then(|e| e.ast.as_ref()) else {
                    return false;
                };
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

fn build_gate_errors(
    native_cov_pct: f64,
    verified_density: f64,
    total_prod_density: f64,
    verified_prod_density: f64,
    config: &VerifySuiteConfig,
) -> Vec<String> {
    let mut errors = Vec::new();
    if verified_density < config.min_verified_assertion_density {
        errors.push(format!(
            "Verified modules density {:.4} is below required {:.4} asserts/fn",
            verified_density, config.min_verified_assertion_density
        ));
    }
    if total_prod_density < config.min_total_production_density {
        errors.push(format!(
            "Total production density {:.4} is below ratcheted floor {:.4} asserts/fn",
            total_prod_density, config.min_total_production_density
        ));
    }
    if verified_prod_density < config.min_verified_production_density {
        errors.push(format!(
            "Verified production density {:.4} is below ratcheted floor {:.4} asserts/fn",
            verified_prod_density, config.min_verified_production_density
        ));
    }
    if native_cov_pct < 50.0 {
        errors.push(format!(
            "Native execution coverage {:.2}% is below threshold of 50.0%",
            native_cov_pct
        ));
    }
    errors
}

pub fn evaluate_integrity_gates(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt: &DebtSources<'_>,
    config: &VerifySuiteConfig,
) -> GateEvaluation {
    let total_density = ratio_or_zero(m.total_asserts, m.total_funcs);
    let verified_density = ratio_or_zero(m.verified_asserts, m.verified_funcs);
    let unverified_density = ratio_or_zero(m.unverified_asserts, m.unverified_funcs);
    let total_prod_density = ratio_or_zero(m.prod_asserts, m.prod_funcs);
    let verified_prod_density = ratio_or_zero(m.verified_prod_asserts, m.verified_prod_funcs);
    let test_density = ratio_or_zero(m.test_asserts, m.test_funcs);

    let (unauthorized_debt, authorized_opt_outs, authorized_unverified_crates) =
        collect_debt_violations(m, debt, config);
    let gate_errors = build_gate_errors(
        native_cov_pct,
        verified_density,
        total_prod_density,
        verified_prod_density,
        config,
    );
    let passed = gate_errors.is_empty() && unauthorized_debt.is_empty();

    GateEvaluation {
        report: IntegrityReport {
            native_execution_coverage_pct: native_cov_pct,
            wasm_path_coverage_pct: wasm_cov_pct,
            total_assertion_density: total_density,
            verified_modules_density: verified_density,
            unverified_modules_density: unverified_density,
            total_production_assertion_density: total_prod_density,
            verified_production_assertion_density: verified_prod_density,
            test_assertion_density: test_density,
            passed,
        },
        unauthorized_debt,
        gate_errors,
        authorized_opt_outs,
        authorized_unverified_crates,
    }
}

pub(crate) fn print_gate_failures(eval: &GateEvaluation) {
    for err in &eval.gate_errors {
        println!("\n[!] {err}");
    }
    if !eval.unauthorized_debt.is_empty() {
        println!("\n--- Unauthorized Integrity Debt (FAIL) ---");
        for (idx, item) in eval.unauthorized_debt.iter().enumerate() {
            println!("[{}] {}", idx + 1, item);
        }
    }
}

fn print_dashboard(eval: &GateEvaluation) {
    let r = &eval.report;
    println!("\n--- Integrity Report ---");
    println!(
        "Native Execution Coverage: {:.2}%",
        r.native_execution_coverage_pct
    );
    println!("WASM Path Coverage: {:.2}%", r.wasm_path_coverage_pct);
    println!("\n--- High-Integrity Dashboard ---");
    println!(
        "Total Assertion Density: {:.2} asserts/fn",
        r.total_assertion_density
    );
    println!(
        "Verified Modules Density: {:.2} asserts/fn",
        r.verified_modules_density
    );
    println!(
        "Unverified Modules Density: {:.2} asserts/fn",
        r.unverified_modules_density
    );
    println!(
        "Total Production Density: {:.4} asserts/fn",
        r.total_production_assertion_density
    );
    println!(
        "Verified Production Density: {:.4} asserts/fn",
        r.verified_production_assertion_density
    );
    println!(
        "Test Assertion Density: {:.2} asserts/fn",
        r.test_assertion_density
    );
    println!(
        "Authorized whitelist exemptions: {} opt-outs, {} non-domain crates",
        eval.authorized_opt_outs, eval.authorized_unverified_crates
    );
    print_gate_failures(eval);
}

pub fn run_report_with_mode(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt: &DebtSources<'_>,
    mode: ReportMode,
) -> bool {
    let config = match load_verify_suite_config_strict(None) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };
    let eval = evaluate_integrity_gates(native_cov_pct, wasm_cov_pct, m, debt, &config);
    print_dashboard(&eval);
    if !eval.report.passed {
        return false;
    }

    let cert_path = Path::new(certificate::CERT_FILE_NAME);
    let readme_path = Path::new(certificate::README_FILE_NAME);
    match mode {
        ReportMode::Write => {
            if let Err(err) =
                certificate::write_certificate_and_readme(&eval.report, cert_path, readme_path)
            {
                eprintln!("[!] {err}");
                return false;
            }
            true
        }
        ReportMode::Check => match certificate::verify_certificate_against_report(
            &eval.report,
            Some(native_cov_pct),
            config.coverage_tolerance_pct,
            cert_path,
            readme_path,
        ) {
            Ok(()) => {
                println!(
                    "Verified {} and {} parity in read-only --check mode.",
                    certificate::CERT_FILE_NAME,
                    certificate::README_FILE_NAME
                );
                true
            }
            Err(diffs) => {
                eprintln!("\n[!] Certificate parity check failed (--check):");
                for diff in diffs {
                    eprintln!("  - {diff}");
                }
                false
            }
        },
    }
}

#[cfg(test)]
pub fn print_report(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt_items: &[String],
) -> bool {
    let config = VerifySuiteConfig::default();
    let debt = DebtSources {
        unverified_crates: &[],
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: debt_items,
    };
    let eval = evaluate_integrity_gates(native_cov_pct, wasm_cov_pct, m, &debt, &config);
    print_dashboard(&eval);
    eval.report.passed
}

#[cfg(test)]
mod tests;
