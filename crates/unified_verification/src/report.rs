use crate::whitelist::VerificationDebtConfig;
use oxidize_core::traceability::AstCache;
use regex::Regex;
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Serialize)]
pub struct IntegrityReport {
    pub native_execution_coverage_pct: f64,
    pub wasm_path_coverage_pct: f64,
    pub total_assertion_density: f64,
    pub verified_modules_density: f64,
    pub unverified_modules_density: f64,
    pub passed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ReportWriteOptions {
    pub output_cert: Option<PathBuf>,
    pub write_tracked: bool,
}

/// Executes `cargo llvm-cov --all-features --workspace --json` and returns process output.
pub fn get_llvm_cov_output() -> std::process::Output {
    let mut cmd = Command::new("cargo");
    cmd.args(["llvm-cov", "--all-features", "--workspace", "--json"]);
    cmd.output().expect("Failed to run cargo llvm-cov")
}

/// Removes instrumented `cargo llvm-cov` build artifacts from the workspace target directory.
pub fn clean_llvm_cov_workspace() -> Result<(), String> {
    let output = Command::new("cargo")
        .args(["llvm-cov", "clean", "--workspace"])
        .output()
        .map_err(|err| format!("Failed to spawn `cargo llvm-cov clean --workspace`: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "Error running cargo llvm-cov clean --workspace:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

/// Collects and parses workspace coverage JSON, then cleans `cargo llvm-cov` build artifacts.
pub fn collect_llvm_cov_json() -> Result<serde_json::Value, String> {
    collect_llvm_cov_json_with(get_llvm_cov_output, clean_llvm_cov_workspace)
}

fn collect_llvm_cov_json_with<F, C>(run_cov: F, clean_cov: C) -> Result<serde_json::Value, String>
where
    F: FnOnce() -> std::process::Output,
    C: FnOnce() -> Result<(), String>,
{
    let output = run_cov();
    if !output.status.success() {
        return Err(format!(
            "Error running cargo llvm-cov:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let cov_json: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|err| format!("Failed to parse coverage JSON: {err}"))?;
    clean_cov()?;
    Ok(cov_json)
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

pub struct FileMetrics {
    pub wasm_paths: usize,
    pub wasm_covered: usize,
    pub total_funcs: f64,
    pub total_asserts: f64,
    pub verified_funcs: f64,
    pub verified_asserts: f64,
    pub unverified_funcs: f64,
    pub unverified_asserts: f64,
    pub opt_outs: Vec<(String, String)>,
}

impl FileMetrics {
    pub(crate) fn empty() -> Self {
        Self {
            wasm_paths: 0,
            wasm_covered: 0,
            total_funcs: 0.0,
            total_asserts: 0.0,
            verified_funcs: 0.0,
            verified_asserts: 0.0,
            unverified_funcs: 0.0,
            unverified_asserts: 0.0,
            opt_outs: Vec::new(),
        }
    }
}

fn normalize_report_path(filepath: &Path) -> String {
    let norm = crate::whitelist::lossy_slash_path(filepath);
    norm.strip_prefix("./").unwrap_or(&norm).to_string()
}

fn accumulate_ast_for_file(
    filepath: &Path,
    cache: &mut AstCache,
    include_prod_metrics: bool,
    m: &mut FileMetrics,
) {
    let norm = normalize_report_path(filepath);
    let key = AstCache::normalize_key(filepath);
    if cache.get(&key).is_none() && cache.get_or_load_fs(filepath).is_none() {
        return;
    }

    if let Some(ast) = cache.get(&key).and_then(|entry| entry.ast.as_ref()) {
        for reason in oxidize_core::ast_visitor::collect_ast_file_debt(ast) {
            m.opt_outs.push((norm.clone(), reason));
        }
        if include_prod_metrics
            && !crate::wasm_coverage::is_dedicated_test_source_path(filepath)
            && !oxidize_core::ast_visitor::attrs_require_cfg_test(&ast.attrs)
        {
            let mut visitor = oxidize_core::ast_visitor::AstVisitor::new();
            syn::visit::Visit::visit_file(&mut visitor, ast);
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
        }
    }

    let (file_wasm_paths, file_wasm_covered) =
        crate::wasm_coverage::count_file_wasm_coverage_cached(filepath, cache);
    m.wasm_paths += file_wasm_paths;
    m.wasm_covered += file_wasm_covered;
}

#[cfg(test)]
fn analyze_file_content(filepath: &Path, content: &str, cache: &mut AstCache, m: &mut FileMetrics) {
    let key = AstCache::normalize_key(filepath);
    cache.insert_source(key, content.to_string());
    accumulate_ast_for_file(filepath, cache, true, m);
}

#[cfg(test)]
pub(crate) fn analyze_files(rs_files: &[PathBuf]) -> FileMetrics {
    let mut cache = AstCache::new();
    let mut m = FileMetrics::empty();
    cache.preload_sources(rs_files);
    for filepath in rs_files {
        accumulate_ast_for_file(filepath, &mut cache, true, &mut m);
    }
    m.opt_outs.sort();
    m
}

pub(crate) fn analyze_workspace_with_cache(
    rs_files: &[PathBuf],
    cache: &mut AstCache,
) -> Result<FileMetrics, String> {
    let mut m = FileMetrics::empty();
    cache.preload_sources(rs_files);
    let ws = oxidize_core::traceability::collect_workspace_ast_metrics_with_cache(cache)
        .map_err(|err| format!("Failed to collect workspace AST metrics: {err}"))?;
    let unverified_funcs = ws.total_funcs.saturating_sub(ws.verified_funcs);
    let unverified_asserts = ws.total_asserts.saturating_sub(ws.verified_asserts);
    m.total_funcs = ws.total_funcs as f64;
    m.total_asserts = ws.total_asserts as f64;
    m.verified_funcs = ws.verified_funcs as f64;
    m.verified_asserts = ws.verified_asserts as f64;
    m.unverified_funcs = unverified_funcs as f64;
    m.unverified_asserts = unverified_asserts as f64;

    for filepath in rs_files {
        accumulate_ast_for_file(filepath, cache, false, &mut m);
    }
    m.opt_outs.sort();
    Ok(m)
}

pub(crate) fn check_unverified_modules_with_cache(
    members: &[String],
    exempt_crates: &[String],
    cache: &mut AstCache,
) -> Vec<String> {
    let mut unverified_modules = Vec::new();
    for fm in members {
        if crate::whitelist::debt::is_exempt_crate(fm, exempt_crates) {
            continue;
        }
        let mod_dir = Path::new(fm).join("src");
        if !mod_dir.exists() {
            continue;
        }
        let files = crate::utils::collect_rs_files_filtered(&[&mod_dir], &["target"]);
        let has_theory = files.into_iter().any(|path| {
            if crate::wasm_coverage::is_dedicated_test_source_path(&path) {
                return false;
            }
            let Some(ast) = cache
                .get_or_load_fs(&path)
                .and_then(|entry| entry.ast.as_ref())
            else {
                return false;
            };
            if oxidize_core::ast_visitor::attrs_require_cfg_test(&ast.attrs) {
                return false;
            }
            let mut visitor = oxidize_core::ast_visitor::AstVisitor::new();
            syn::visit::Visit::visit_file(&mut visitor, ast);
            !visitor.module_tiers.is_empty()
        });
        if !has_theory {
            unverified_modules.push(format!("Unverified Module: {fm}"));
        }
    }
    unverified_modules.sort();
    unverified_modules
}

#[cfg(test)]
pub fn check_unverified_modules(members: &[String]) -> Vec<String> {
    let whitelist = crate::whitelist::load_whitelist_or_default(
        None,
        crate::whitelist::WORKSPACE_WHITELIST_CANDIDATES,
    );
    let exempt = whitelist
        .verification_debt
        .map(|d| d.exempt_crates)
        .unwrap_or_else(|| vec!["egui_plot".to_string()]);
    let mut cache = AstCache::new();
    check_unverified_modules_with_cache(members, &exempt, &mut cache)
}

fn write_pretty_certificate(path: &Path, json_report: &str) -> Result<(), String> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create certificate directory {}: {err}",
                parent.display()
            )
        })?;
    }
    fs::write(path, format!("{json_report}\n"))
        .map_err(|err| format!("Failed to write {}: {err}", path.display()))?;
    println!("Generated {}", path.display());
    Ok(())
}

pub(crate) fn save_report_in_dir(
    base_dir: &Path,
    report: &IntegrityReport,
    native_cov: f64,
    write_opts: &ReportWriteOptions,
) -> Result<(), String> {
    if write_opts.output_cert.is_none() && !write_opts.write_tracked {
        return Ok(());
    }
    let json_report = serde_json::to_string_pretty(report)
        .map_err(|err| format!("Failed to serialize IntegrityReport: {err}"))?;

    if let Some(cert_path) = &write_opts.output_cert {
        let target = if cert_path.is_absolute() || base_dir == Path::new(".") {
            cert_path.clone()
        } else {
            base_dir.join(cert_path)
        };
        write_pretty_certificate(&target, &json_report)?;
    }

    if write_opts.write_tracked {
        let tracked_cert = if base_dir == Path::new(".") {
            PathBuf::from("Verification_Certificate.json")
        } else {
            base_dir.join("Verification_Certificate.json")
        };
        write_pretty_certificate(&tracked_cert, &json_report)?;

        let readme_path = base_dir.join("README.md");
        if let Ok(readme) = fs::read_to_string(&readme_path) {
            let badge_color = if native_cov >= 90.0 {
                "brightgreen"
            } else {
                "yellow"
            };
            let new_badge = format!(
                "![Coverage](https://img.shields.io/badge/coverage-{:.1}%25-{})",
                native_cov, badge_color
            );
            let badge_re = Regex::new(r"!\[Coverage\]\(.*?\)").expect("Valid coverage badge regex");
            let updated = badge_re.replace(&readme, new_badge.as_str()).to_string();
            fs::write(&readme_path, updated)
                .map_err(|err| format!("Failed to write {}: {err}", readme_path.display()))?;
            println!("README static badge updated with dynamic metric.");
        }
    }
    Ok(())
}

fn ratio_or_zero(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

fn check_density_floors(
    total_density: f64,
    verified_density: f64,
    unverified_density: f64,
    m: &FileMetrics,
    debt_cfg: Option<&VerificationDebtConfig>,
) -> bool {
    let mut passed = true;
    let min_verified = debt_cfg.map_or(0.4, |c| c.min_verified_density);
    if verified_density < min_verified {
        println!(
            "\n[!] Assertion Density Failure: Verified modules have a density of {:.3} asserts/fn, which is below the minimum required {:.3} asserts/fn.",
            verified_density, min_verified
        );
        passed = false;
    }

    if let Some(cfg) = debt_cfg {
        if m.total_funcs > 0.0 && total_density < cfg.min_total_density {
            println!(
                "\n[!] Assertion Density Failure: Total assertion density is {:.3} asserts/fn, which is below the minimum required {:.3} asserts/fn.",
                total_density, cfg.min_total_density
            );
            passed = false;
        }
        if let Some(min_unverified) = cfg.min_unverified_density
            && m.unverified_funcs > 0.0
            && unverified_density < min_unverified
        {
            println!(
                "\n[!] Assertion Density Failure: Unverified modules have a density of {:.3} asserts/fn, which is below the minimum required {:.3} asserts/fn.",
                unverified_density, min_unverified
            );
            passed = false;
        }
    }
    passed
}

fn evaluate_and_print_debt(
    m: &FileMetrics,
    debt_items: &[String],
    debt_cfg: Option<&VerificationDebtConfig>,
) -> bool {
    let mut violations = Vec::new();
    if let Some(cfg) = debt_cfg {
        violations.extend(crate::whitelist::debt::evaluate_opt_out_caps(
            cfg,
            &m.opt_outs,
        ));
        if violations.is_empty() && !m.opt_outs.is_empty() {
            println!(
                "Authorized opt_out attributes: {} (within [verification_debt] caps)",
                m.opt_outs.len()
            );
        }
    } else {
        for (file, reason) in &m.opt_outs {
            violations.push(format!("{file} bypassed: '{reason}'"));
        }
    }
    violations.extend(debt_items.iter().cloned());

    if !violations.is_empty() {
        println!("\n--- Integrity Debt ---");
        for (idx, item) in violations.iter().enumerate() {
            println!("[{}] {}", idx + 1, item);
        }
        return false;
    }
    true
}

pub(crate) fn print_report_with_options(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt_items: &[String],
    debt_cfg: Option<&VerificationDebtConfig>,
    write_opts: &ReportWriteOptions,
) -> Result<bool, String> {
    let total_density = ratio_or_zero(m.total_asserts, m.total_funcs);
    let verified_density = ratio_or_zero(m.verified_asserts, m.verified_funcs);
    let unverified_density = ratio_or_zero(m.unverified_asserts, m.unverified_funcs);

    println!("\n--- Integrity Report ---");
    println!("Native Execution Coverage: {:.2}%", native_cov_pct);
    println!("WASM Path Coverage: {:.2}%", wasm_cov_pct);

    println!("\n--- High-Integrity Dashboard ---");
    println!("Total Assertion Density: {:.2} asserts/fn", total_density);
    println!(
        "Verified Modules Density: {:.2} asserts/fn",
        verified_density
    );
    println!(
        "Unverified Modules Density: {:.2} asserts/fn",
        unverified_density
    );

    let mut passed = check_density_floors(
        total_density,
        verified_density,
        unverified_density,
        m,
        debt_cfg,
    );
    if !evaluate_and_print_debt(m, debt_items, debt_cfg) {
        passed = false;
    }
    if native_cov_pct < 50.0 {
        println!(
            "Coverage {:.2}% is below threshold of 50.0%",
            native_cov_pct
        );
        passed = false;
    }

    let report = IntegrityReport {
        native_execution_coverage_pct: native_cov_pct,
        wasm_path_coverage_pct: wasm_cov_pct,
        total_assertion_density: total_density,
        verified_modules_density: verified_density,
        unverified_modules_density: unverified_density,
        passed,
    };

    save_report_in_dir(Path::new("."), &report, native_cov_pct, write_opts)?;
    Ok(passed)
}

#[cfg(test)]
pub fn print_report(
    native_cov_pct: f64,
    wasm_cov_pct: f64,
    m: &FileMetrics,
    debt_items: &[String],
) -> bool {
    print_report_with_options(
        native_cov_pct,
        wasm_cov_pct,
        m,
        debt_items,
        None,
        &ReportWriteOptions::default(),
    )
    .unwrap_or(false)
}

#[cfg(test)]
mod tests;
