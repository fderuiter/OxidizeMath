use super::certificate::{
    format_coverage_badge, verify_certificate_against_report, verify_certificate_strict_parity,
    write_certificate_and_readme, write_certificate_only,
};
use super::tests::baseline_metrics;
use super::*;
use crate::utils::test_utils::TempTestDir;
use crate::whitelist::VerifySuiteConfig;
use std::fs;
use std::path::{Path, PathBuf};

fn assert_deterministic_and_pyramid_drift_rejected(
    base: &IntegrityReport,
    cert_path: &Path,
    readme_path: &Path,
) {
    let mut density_drifted = base.clone();
    density_drifted.total_assertion_density += 0.25;
    assert!(
        verify_certificate_against_report(&density_drifted, cert_path, readme_path).is_err(),
        "Default --check must reject deterministic AST density drift"
    );

    let mut pyramid_drifted = base.clone();
    pyramid_drifted.certified_solvers += 1;
    assert!(
        verify_certificate_against_report(&pyramid_drifted, cert_path, readme_path).is_err(),
        "Default --check must reject pyramid certified_solvers drift"
    );

    let mut digest_drifted = base.clone();
    digest_drifted
        .lean4_proof_digests
        .insert("IntervalSoundness.lean".to_string(), "0".repeat(64));
    assert!(
        verify_certificate_against_report(&digest_drifted, cert_path, readme_path).is_err(),
        "Default --check must reject tampered lean4_proof_digests"
    );

    let mut binding_drifted = base.clone();
    if let Some(binding) = binding_drifted
        .lean4_proof_bindings
        .get_mut("IntervalSoundness.lean")
    {
        binding.theorems.push("tampered_theorem".to_string());
    }
    assert!(
        verify_certificate_against_report(&binding_drifted, cert_path, readme_path).is_err(),
        "Default --check must reject tampered lean4_proof_bindings"
    );
}

#[test]
fn test_certificate_check_mode_enforces_deterministic_and_pyramid_parity() {
    let dir = TempTestDir::new("cert_parity");
    let cert_path = dir.join("Verification_Certificate.json");
    let readme_path = dir.join("README.md");
    fs::write(
        &readme_path,
        "# Demo\n![Coverage](https://img.shields.io/badge/coverage-10.0%25-yellow)\n",
    )
    .unwrap();

    let empty_debt = DebtSources {
        unverified_crates: &[],
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: &[],
    };
    let eval = evaluate_integrity_gates(
        84.5,
        50.0,
        &baseline_metrics(),
        &empty_debt,
        &VerifySuiteConfig::default(),
    );
    write_certificate_and_readme(&eval.report, &cert_path, &readme_path).unwrap();

    let cert_before = fs::read_to_string(&cert_path).unwrap();
    let readme_before = fs::read_to_string(&readme_path).unwrap();
    assert!(readme_before.contains(&format_coverage_badge(84.5)));

    let mut cov_shifted = eval.report.clone();
    cov_shifted.native_execution_coverage_pct = 93.2;
    assert!(verify_certificate_against_report(&cov_shifted, &cert_path, &readme_path).is_ok());
    assert!(
        verify_certificate_strict_parity(&eval.report, Some(85.2), 1.5, &cert_path, &readme_path)
            .is_ok()
    );
    assert!(
        verify_certificate_strict_parity(&eval.report, Some(87.0), 1.5, &cert_path, &readme_path)
            .is_err()
    );

    assert_deterministic_and_pyramid_drift_rejected(&eval.report, &cert_path, &readme_path);
    assert_eq!(fs::read_to_string(&cert_path).unwrap(), cert_before);
    assert_eq!(fs::read_to_string(&readme_path).unwrap(), readme_before);
}

#[test]
fn test_certificate_check_rejects_missing_malformed_failed_and_bad_badge() {
    let dir = TempTestDir::new("cert_schema_rejections");
    let cert_path = dir.join("Verification_Certificate.json");
    let readme_path = dir.join("README.md");
    let eval = evaluate_integrity_gates(
        84.5,
        75.0,
        &baseline_metrics(),
        &DebtSources {
            unverified_crates: &[],
            file_length_debt: &[],
            entropy_debt: &[],
            extra_debt: &[],
        },
        &VerifySuiteConfig::default(),
    );

    assert!(verify_certificate_against_report(&eval.report, &cert_path, &readme_path).is_err());

    for invalid_payload in [
        "not-json",
        "[84.5, 75.0, 1.0, 1.0, 1.0, 0.5, 0.5, 1.75, true]",
        r#"{"native_execution_coverage_pct":84.5,"passed":true}"#,
        r#"{"native_execution_coverage_pct":84.5,"wasm_path_coverage_pct":75.0,"total_assertion_density":1.0,"verified_modules_density":1.0,"unverified_modules_density":1.0,"total_production_assertion_density":0.5,"verified_production_assertion_density":0.5,"test_assertion_density":1.75,"passed":false}"#,
        r#"{"native_execution_coverage_pct":105.0,"wasm_path_coverage_pct":75.0,"total_assertion_density":1.0,"verified_modules_density":1.0,"unverified_modules_density":1.0,"total_production_assertion_density":0.5,"verified_production_assertion_density":0.5,"test_assertion_density":1.75,"passed":true}"#,
        r#"{"native_execution_coverage_pct":84.5,"wasm_path_coverage_pct":75.0,"total_assertion_density":-0.1,"verified_modules_density":1.0,"unverified_modules_density":1.0,"total_production_assertion_density":0.5,"verified_production_assertion_density":0.5,"test_assertion_density":1.75,"passed":true}"#,
        r#"{"native_execution_coverage_pct":84.5,"wasm_path_coverage_pct":75.0,"total_assertion_density":1.0,"verified_modules_density":1.0,"unverified_modules_density":1.0,"total_production_assertion_density":0.5,"verified_production_assertion_density":0.5,"test_assertion_density":1.75,"passed":true,"extra":1}"#,
    ] {
        fs::write(&cert_path, invalid_payload).unwrap();
        assert!(
            verify_certificate_against_report(&eval.report, &cert_path, &readme_path).is_err(),
            "Payload must fail schema validation: {invalid_payload}"
        );
    }

    let mut unverified_binding = eval.report.clone();
    if let Some(b) = unverified_binding
        .lean4_proof_bindings
        .get_mut("IntervalSoundness.lean")
    {
        b.kernel_verified = false;
    }
    write_certificate_and_readme(&unverified_binding, &cert_path, &readme_path).unwrap();
    assert!(
        verify_certificate_against_report(&eval.report, &cert_path, &readme_path).is_err(),
        "Certificate with kernel_verified=false must fail bounds validation"
    );

    write_certificate_and_readme(&eval.report, &cert_path, &readme_path).unwrap();
    fs::write(&readme_path, "# Demo\n![Coverage](broken_unclosed\n").unwrap();
    assert!(verify_certificate_against_report(&eval.report, &cert_path, &readme_path).is_err());
    fs::write(
        &readme_path,
        "# Demo\n![Coverage](https://img.shields.io/badge/coverage-10.0%25-yellow)\n",
    )
    .unwrap();
    assert!(verify_certificate_against_report(&eval.report, &cert_path, &readme_path).is_err());
}

#[test]
fn test_write_certificate_only_emits_15_key_schema_without_touching_tracked_files() {
    let tmp = TempTestDir::new("report_output_cert_only");
    let readme_path = tmp.join("README.md");
    let orig_readme = "# Demo\n![Coverage](https://img.shields.io/badge/coverage-10.0%25-yellow)\n";
    fs::write(&readme_path, orig_readme).unwrap();

    let metrics = baseline_metrics();
    let cfg = VerifySuiteConfig::default();
    let debt = DebtSources {
        unverified_crates: &[],
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: &[],
    };
    let eval = evaluate_integrity_gates(92.4, 81.0, &metrics, &debt, &cfg);
    assert_eq!(eval.report.certified_solvers, 55);
    assert_eq!(eval.report.tier_counts.len(), 6);
    assert_eq!(eval.report.lean4_proof_bindings.len(), 6);
    let artifact_abs = tmp.join(PathBuf::from(
        "target/verification/Verification_Certificate.json",
    ));
    write_certificate_only(&eval.report, &artifact_abs).unwrap();

    assert!(artifact_abs.is_file() && !tmp.join("Verification_Certificate.json").exists());
    assert_eq!(fs::read_to_string(&readme_path).unwrap(), orig_readme);

    let parsed: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&artifact_abs).unwrap()).unwrap();
    let mut keys: Vec<&str> = parsed
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "certified_solvers",
            "interval_bounds",
            "lean4_proof_bindings",
            "lean4_proof_digests",
            "max_interval_width",
            "native_execution_coverage_pct",
            "passed",
            "test_assertion_density",
            "tier_counts",
            "total_assertion_density",
            "total_production_assertion_density",
            "unverified_modules_density",
            "verified_modules_density",
            "verified_production_assertion_density",
            "wasm_path_coverage_pct",
        ]
    );
}
