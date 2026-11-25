use super::{
    DebtSources, IntegrityReport, ReportMode, analyze_files_with_cache, collect_rs_files,
    evaluate_integrity_gates, find_unverified_crates_with_cache,
    load_suite_and_debt_configs_strict,
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

pub fn write_certificate_only(report: &IntegrityReport, cert_path: &Path) -> Result<(), String> {
    if let Some(parent) = cert_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "Failed to create certificate directory {}: {err}",
                parent.display()
            )
        })?;
    }
    let mut json_report = serde_json::to_string_pretty(report)
        .map_err(|err| format!("Failed to serialize certificate: {err}"))?;
    json_report.push('\n');
    fs::write(cert_path, json_report)
        .map_err(|err| format!("Failed to write {}: {err}", cert_path.display()))?;
    println!("Generated {}", cert_path.display());
    Ok(())
}

pub fn write_certificate_and_readme(
    report: &IntegrityReport,
    cert_path: &Path,
    readme_path: &Path,
) -> Result<(), String> {
    write_certificate_only(report, cert_path)?;

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

const REQUIRED_CERT_KEYS: [&str; 9] = [
    "native_execution_coverage_pct",
    "passed",
    "test_assertion_density",
    "total_assertion_density",
    "total_production_assertion_density",
    "unverified_modules_density",
    "verified_modules_density",
    "verified_production_assertion_density",
    "wasm_path_coverage_pct",
];

fn parse_certificate_strict_9_keys(
    raw: &str,
    cert_path: &Path,
) -> Result<IntegrityReport, Vec<String>> {
    let val: serde_json::Value = serde_json::from_str(raw)
        .map_err(|err| vec![format!("Invalid JSON in {}: {err}", cert_path.display())])?;
    let Some(obj) = val.as_object() else {
        return Err(vec![format!(
            "Invalid schema in {}: expected top-level JSON object",
            cert_path.display()
        )]);
    };
    let mut keys: Vec<&str> = obj.keys().map(String::as_str).collect();
    keys.sort_unstable();
    if keys != REQUIRED_CERT_KEYS {
        return Err(vec![format!(
            "Invalid 9-key schema in {}: expected keys {:?}, found {:?}",
            cert_path.display(),
            REQUIRED_CERT_KEYS,
            keys
        )]);
    }
    serde_json::from_str::<IntegrityReport>(raw)
        .map_err(|err| vec![format!("Invalid JSON in {}: {err}", cert_path.display())])
}

fn validate_certificate_bounds(cert: &IntegrityReport, cert_path: &Path, diffs: &mut Vec<String>) {
    if !cert.passed {
        diffs.push(format!("{} records passed=false", cert_path.display()));
    }
    for (field, val) in [
        (
            "native_execution_coverage_pct",
            cert.native_execution_coverage_pct,
        ),
        ("wasm_path_coverage_pct", cert.wasm_path_coverage_pct),
    ] {
        if !val.is_finite() || !(0.0..=100.0).contains(&val) {
            diffs.push(format!(
                "Certificate {field} must be finite within 0.0..=100.0 (got {val})"
            ));
        }
    }
    for (field, val) in [
        ("total_assertion_density", cert.total_assertion_density),
        ("verified_modules_density", cert.verified_modules_density),
        (
            "unverified_modules_density",
            cert.unverified_modules_density,
        ),
        (
            "total_production_assertion_density",
            cert.total_production_assertion_density,
        ),
        (
            "verified_production_assertion_density",
            cert.verified_production_assertion_density,
        ),
        ("test_assertion_density", cert.test_assertion_density),
    ] {
        if !val.is_finite() || val < 0.0 {
            diffs.push(format!(
                "Certificate {field} must be finite and >= 0.0 (got {val})"
            ));
        }
    }
}

fn load_and_validate_certificate(
    expected: &IntegrityReport,
    cert_path: &Path,
    readme_path: &Path,
) -> Result<IntegrityReport, Vec<String>> {
    let raw = fs::read_to_string(cert_path).map_err(|err| {
        vec![format!(
            "Missing or unreadable {}: {err} (run `cargo run -p unified_verification -- verify-suite --write` locally to refresh)",
            cert_path.display()
        )]
    })?;
    let cert = parse_certificate_strict_9_keys(&raw, cert_path)?;

    let mut diffs = Vec::new();
    validate_certificate_bounds(&cert, cert_path, &mut diffs);
    if !expected.passed {
        diffs.push("Current workspace failed integrity gates".to_string());
    }
    if cert.native_execution_coverage_pct.is_finite()
        && (0.0..=100.0).contains(&cert.native_execution_coverage_pct)
    {
        check_readme_badge(&cert, readme_path, &mut diffs);
    }

    if diffs.is_empty() {
        Ok(cert)
    } else {
        Err(diffs)
    }
}

pub fn verify_certificate_against_report(
    expected: &IntegrityReport,
    cert_path: &Path,
    readme_path: &Path,
) -> Result<(), Vec<String>> {
    load_and_validate_certificate(expected, cert_path, readme_path).map(|_| ())
}

pub fn verify_certificate_strict_parity(
    expected: &IntegrityReport,
    measured_native_cov: Option<f64>,
    coverage_tolerance_pct: f64,
    cert_path: &Path,
    readme_path: &Path,
) -> Result<(), Vec<String>> {
    let cert = load_and_validate_certificate(expected, cert_path, readme_path)?;
    let mut diffs = Vec::new();
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
    let marker = "![Coverage](";
    if !readme.contains(marker) {
        return;
    }
    let expected_badge = format_coverage_badge(cert.native_execution_coverage_pct);
    let mut rest = readme.as_str();
    while let Some(start) = rest.find(marker) {
        let from_marker = &rest[start..];
        let line_slice = from_marker.lines().next().unwrap_or(from_marker);
        let Some(close_rel) = line_slice.find(')') else {
            diffs.push(format!(
                "{} contains malformed coverage badge (expected `{expected_badge}`)",
                readme_path.display()
            ));
            return;
        };
        let actual_badge = &line_slice[..=close_rel];
        if actual_badge != expected_badge {
            diffs.push(format!(
                "{} coverage badge does not match {} (expected `{expected_badge}`, found `{actual_badge}`)",
                readme_path.display(),
                CERT_FILE_NAME
            ));
            return;
        }
        rest = &from_marker[close_rel + 1..];
    }
}

pub(super) fn run_check_certificate_io(
    expected: &IntegrityReport,
    cert_path: &Path,
    readme_path: &Path,
    ok_message: &str,
) -> bool {
    match verify_certificate_against_report(expected, cert_path, readme_path) {
        Ok(()) => {
            println!("{ok_message}");
            true
        }
        Err(diffs) => {
            eprintln!("\n[!] Verification_Certificate.json validation failed (--check):");
            for diff in diffs {
                eprintln!("  - {diff}");
            }
            false
        }
    }
}

pub(super) fn run_strict_parity_io(
    expected: &IntegrityReport,
    measured_native_cov: Option<f64>,
    coverage_tolerance_pct: f64,
    cert_path: &Path,
    readme_path: &Path,
) -> bool {
    match verify_certificate_strict_parity(
        expected,
        measured_native_cov,
        coverage_tolerance_pct,
        cert_path,
        readme_path,
    ) {
        Ok(()) => {
            println!(
                "Verified {} and {} strict parity.",
                CERT_FILE_NAME, README_FILE_NAME
            );
            true
        }
        Err(diffs) => {
            eprintln!("\n[!] Certificate strict parity check failed (--strict-parity):");
            for diff in diffs {
                eprintln!("  - {diff}");
            }
            false
        }
    }
}

pub fn check_ast_certificate_parity_with_cache(members: &[String], cache: &mut AstCache) -> bool {
    sync_ast_certificate_with_cache(members, ReportMode::Check, cache)
}

fn resolve_static_cert_coverage(cert_path: &Path, mode: ReportMode) -> f64 {
    if mode != ReportMode::Write && std::env::var_os(super::VERIFY_SUITE_WRITE_ENV).is_some() {
        return 100.0;
    }
    fs::read_to_string(cert_path)
        .ok()
        .and_then(|raw| serde_json::from_str::<IntegrityReport>(&raw).ok())
        .map(|c| c.native_execution_coverage_pct)
        .unwrap_or(100.0)
}

fn apply_static_certificate_mode(
    report: &IntegrityReport,
    mode: ReportMode,
    tolerance_pct: f64,
    cert_path: &Path,
    readme_path: &Path,
) -> bool {
    if mode != ReportMode::Write && std::env::var_os(super::VERIFY_SUITE_WRITE_ENV).is_some() {
        return true;
    }
    match mode {
        ReportMode::Check => run_check_certificate_io(
            report,
            cert_path,
            readme_path,
            "Verification_Certificate.json schema and [verify_suite] quality floors verified.",
        ),
        ReportMode::StrictParity => {
            run_strict_parity_io(report, None, tolerance_pct, cert_path, readme_path)
        }
        ReportMode::Write => match write_certificate_and_readme(report, cert_path, readme_path) {
            Ok(()) => true,
            Err(err) => {
                eprintln!("[!] {err}");
                false
            }
        },
    }
}

pub fn sync_ast_certificate_with_cache(
    members: &[String],
    mode: ReportMode,
    cache: &mut AstCache,
) -> bool {
    let (config, debt_caps) = match load_suite_and_debt_configs_strict(None) {
        Ok(pair) => pair,
        Err(err) => {
            eprintln!("[!] {err}");
            return false;
        }
    };
    let cert_path = Path::new(CERT_FILE_NAME);
    let readme_path = Path::new(README_FILE_NAME);
    let cert_cov = resolve_static_cert_coverage(cert_path, mode);

    let rs_files = collect_rs_files();
    let metrics = analyze_files_with_cache(&rs_files, cache);
    let unverified = find_unverified_crates_with_cache(members, cache);
    let file_len_debt = crate::utils::check_file_lengths(members);
    let entropy_debt = crate::entropy_guard::check_entropy_with_cache(&[], cache);
    let cap_violations = debt_caps
        .as_ref()
        .map(|cfg| crate::whitelist::debt::evaluate_opt_out_caps(cfg, &metrics.opt_outs))
        .unwrap_or_default();
    let wasm_cov_pct = if metrics.wasm_paths > 0 {
        (metrics.wasm_covered as f64 / metrics.wasm_paths as f64) * 100.0
    } else {
        100.0
    };

    let debt = DebtSources {
        unverified_crates: &unverified,
        file_length_debt: &file_len_debt,
        entropy_debt: &entropy_debt,
        extra_debt: &cap_violations,
    };
    let eval = evaluate_integrity_gates(cert_cov, wasm_cov_pct, &metrics, &debt, &config);
    if !eval.report.passed {
        super::print_gate_failures(&eval);
        return false;
    }
    apply_static_certificate_mode(
        &eval.report,
        mode,
        config.coverage_tolerance_pct,
        cert_path,
        readme_path,
    )
}
