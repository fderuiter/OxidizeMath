use super::{
    DebtSources, IntegrityReport, analyze_files_with_cache, collect_rs_files,
    evaluate_integrity_gates, find_unverified_crates_with_cache, load_verify_suite_config_strict,
};
use oxidize_core::traceability::AstCache;
use std::fs;
use std::path::Path;

pub const CERT_FILE_NAME: &str = "Verification_Certificate.json";
pub const README_FILE_NAME: &str = "README.md";
pub const CERT_FLOAT_EPSILON: f64 = 1e-6;

pub fn format_coverage_badge(native_cov: f64) -> String {
    let badge_color = if native_cov >= 90.0 {
        "brightgreen"
    } else {
        "yellow"
    };
    format!(
        "![Coverage](https://img.shields.io/badge/coverage-{:.1}%25-{})",
        native_cov, badge_color
    )
}

fn replace_coverage_badge(readme: &str, new_badge: &str) -> String {
    let Some(start) = readme.find("![Coverage](") else {
        return readme.to_string();
    };
    let rest = &readme[start..];
    let Some(close_rel) = rest.find(')') else {
        return readme.to_string();
    };
    let end = start + close_rel + 1;
    format!("{}{}{}", &readme[..start], new_badge, &readme[end..])
}

pub fn write_certificate_and_readme(
    report: &IntegrityReport,
    cert_path: &Path,
    readme_path: &Path,
) -> Result<(), String> {
    let mut json_report = serde_json::to_string_pretty(report)
        .map_err(|err| format!("Failed to serialize certificate: {err}"))?;
    json_report.push('\n');
    fs::write(cert_path, json_report)
        .map_err(|err| format!("Failed to write {}: {err}", cert_path.display()))?;
    println!("Generated {}", cert_path.display());

    if readme_path.exists() {
        let readme = fs::read_to_string(readme_path)
            .map_err(|err| format!("Failed to read {}: {err}", readme_path.display()))?;
        let updated = replace_coverage_badge(
            &readme,
            &format_coverage_badge(report.native_execution_coverage_pct),
        );
        if updated != readme {
            fs::write(readme_path, updated)
                .map_err(|err| format!("Failed to write {}: {err}", readme_path.display()))?;
            println!("README static badge updated with dynamic metric.");
        }
    }
    Ok(())
}

fn check_float_eq(field: &str, actual: f64, cert_val: f64, eps: f64, diffs: &mut Vec<String>) {
    if !actual.is_finite() || !cert_val.is_finite() || (actual - cert_val).abs() > eps {
        diffs.push(format!(
            "{field} drift: workspace={actual:.6}, certificate={cert_val:.6} (tolerance={eps})"
        ));
    }
}

pub fn verify_certificate_against_report(
    expected: &IntegrityReport,
    measured_native_cov: Option<f64>,
    coverage_tolerance_pct: f64,
    cert_path: &Path,
    readme_path: &Path,
) -> Result<(), Vec<String>> {
    let raw = fs::read_to_string(cert_path).map_err(|err| {
        vec![format!(
            "Missing or unreadable {}: {err} (run `cargo run -p unified_verification -- verify-suite --write` locally to refresh)",
            cert_path.display()
        )]
    })?;
    let cert: IntegrityReport = serde_json::from_str(&raw)
        .map_err(|err| vec![format!("Invalid JSON in {}: {err}", cert_path.display())])?;

    let mut diffs = Vec::new();
    if !cert.passed {
        diffs.push(format!("{} records passed=false", cert_path.display()));
    }
    if !expected.passed {
        diffs.push("Current workspace failed integrity gates".to_string());
    }
    if cert.native_execution_coverage_pct < 50.0 {
        diffs.push(format!(
            "Certificate native_execution_coverage_pct {:.2}% is below 50.0%",
            cert.native_execution_coverage_pct
        ));
    }
    if let Some(measured_cov) = measured_native_cov {
        check_float_eq(
            "native_execution_coverage_pct",
            measured_cov,
            cert.native_execution_coverage_pct,
            coverage_tolerance_pct,
            &mut diffs,
        );
    }

    check_deterministic_fields(expected, &cert, &mut diffs);
    check_readme_badge(&cert, readme_path, &mut diffs);

    if diffs.is_empty() { Ok(()) } else { Err(diffs) }
}

fn check_deterministic_fields(
    expected: &IntegrityReport,
    cert: &IntegrityReport,
    diffs: &mut Vec<String>,
) {
    let pairs = [
        (
            "wasm_path_coverage_pct",
            expected.wasm_path_coverage_pct,
            cert.wasm_path_coverage_pct,
        ),
        (
            "total_assertion_density",
            expected.total_assertion_density,
            cert.total_assertion_density,
        ),
        (
            "verified_modules_density",
            expected.verified_modules_density,
            cert.verified_modules_density,
        ),
        (
            "unverified_modules_density",
            expected.unverified_modules_density,
            cert.unverified_modules_density,
        ),
        (
            "total_production_assertion_density",
            expected.total_production_assertion_density,
            cert.total_production_assertion_density,
        ),
        (
            "verified_production_assertion_density",
            expected.verified_production_assertion_density,
            cert.verified_production_assertion_density,
        ),
        (
            "test_assertion_density",
            expected.test_assertion_density,
            cert.test_assertion_density,
        ),
    ];
    for (name, actual, recorded) in pairs {
        check_float_eq(name, actual, recorded, CERT_FLOAT_EPSILON, diffs);
    }
}

fn check_readme_badge(cert: &IntegrityReport, readme_path: &Path, diffs: &mut Vec<String>) {
    let Ok(readme) = fs::read_to_string(readme_path) else {
        return;
    };
    if !readme.contains("![Coverage](") {
        return;
    }
    let expected_badge = format_coverage_badge(cert.native_execution_coverage_pct);
    if !readme.contains(&expected_badge) {
        diffs.push(format!(
            "{} coverage badge does not match {} (expected `{expected_badge}`)",
            readme_path.display(),
            CERT_FILE_NAME
        ));
    }
}

pub fn check_ast_certificate_parity_with_cache(members: &[String], cache: &mut AstCache) -> bool {
    let config = match load_verify_suite_config_strict(None) {
        Ok(cfg) => cfg,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };
    let cert_path = Path::new(CERT_FILE_NAME);
    let readme_path = Path::new(README_FILE_NAME);
    let cert_cov = fs::read_to_string(cert_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<IntegrityReport>(&raw).ok())
        .map(|c| c.native_execution_coverage_pct)
        .unwrap_or(100.0);

    let rs_files = collect_rs_files();
    let metrics = analyze_files_with_cache(&rs_files, cache);
    let unverified = find_unverified_crates_with_cache(members, cache);
    let file_len_debt = crate::utils::check_file_lengths(members);
    let entropy_debt = crate::entropy_guard::check_entropy_with_cache(&[], cache);
    let wasm_cov_pct = if metrics.wasm_paths > 0 {
        (metrics.wasm_covered as f64 / metrics.wasm_paths as f64) * 100.0
    } else {
        100.0
    };

    let debt = DebtSources {
        unverified_crates: &unverified,
        file_length_debt: &file_len_debt,
        entropy_debt: &entropy_debt,
        extra_debt: &[],
    };
    let eval = evaluate_integrity_gates(cert_cov, wasm_cov_pct, &metrics, &debt, &config);
    if !eval.report.passed {
        super::print_gate_failures(&eval);
        return false;
    }
    match verify_certificate_against_report(
        &eval.report,
        None,
        config.coverage_tolerance_pct,
        cert_path,
        readme_path,
    ) {
        Ok(()) => {
            println!("Verification_Certificate.json AST parity verified.");
            true
        }
        Err(diffs) => {
            eprintln!("\n[!] Verification_Certificate.json parity check failed:");
            for diff in diffs {
                eprintln!("  - {diff}");
            }
            false
        }
    }
}
