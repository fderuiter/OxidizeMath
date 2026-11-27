use super::*;
use crate::report::ReportMode;
use crate::utils::test_utils::{CurrentDirGuard, TempTestDir};
use std::fs;
use std::path::PathBuf;

fn parse(args: &[&str]) -> Result<VerifySuiteOptions, String> {
    let owned: Vec<String> = args.iter().map(|s| (*s).to_string()).collect();
    parse_verify_suite_args(&owned)
}

fn sample_cov_json(count: u64, covered: u64) -> serde_json::Value {
    serde_json::json!({
        "data": [{
            "files": [{
                "summary": {
                    "lines": {
                        "count": count,
                        "covered": covered
                    }
                }
            }]
        }]
    })
}

fn seed_temp_verify_suite_workspace(dir: &TempTestDir) {
    fs::create_dir_all(dir.join("crates/demo/src")).unwrap();
    fs::write(
        dir.join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/demo\"]\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/demo/Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    fs::write(
        dir.join("crates/demo/src/lib.rs"),
        concat!(
            "#[verified_engine::verified]\n",
            "pub fn step(x: f64) -> f64 { assert!(x.is_finite()); x }\n",
            "theory_verification! { module = \"demo\", test = { assert_eq!(step(1.0), 1.0); } }\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("verification_whitelist.toml"),
        concat!(
            "[verify_suite]\n",
            "min_verified_assertion_density = 0.40\n",
            "min_total_production_density = 0.20\n",
            "min_verified_production_density = 0.20\n",
            "coverage_tolerance_pct = 1.5\n",
        ),
    )
    .unwrap();
    fs::write(
        dir.join("README.md"),
        "# Demo\n![Coverage](https://img.shields.io/badge/coverage-10.0%25-yellow)\n",
    )
    .unwrap();
}

#[test]
fn verify_suite_defaults_to_read_only_check_and_validates_flags() {
    assert_eq!(
        parse(&[]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Check,
            auto_fix: false,
            coverage_only: false,
            output_cert: None,
        }
    );
    assert_eq!(
        parse(&["--check", "--coverage-only"]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Check,
            auto_fix: false,
            coverage_only: true,
            output_cert: None,
        }
    );
    assert_eq!(
        parse(&[
            "--coverage-only",
            "--output-cert",
            "target/verification/Verification_Certificate.json",
        ])
        .unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Check,
            auto_fix: false,
            coverage_only: true,
            output_cert: Some(PathBuf::from(
                "target/verification/Verification_Certificate.json"
            )),
        }
    );
    assert_eq!(
        parse(&["--strict-parity", "--coverage-only"]).unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::StrictParity,
            auto_fix: false,
            coverage_only: true,
            output_cert: None,
        }
    );
    assert_eq!(
        parse(&[
            "--write-tracked",
            "--auto-fix",
            "--coverage-only",
            "--output-cert=target/cert.json",
        ])
        .unwrap(),
        VerifySuiteOptions {
            mode: ReportMode::Write,
            auto_fix: true,
            coverage_only: true,
            output_cert: Some(PathBuf::from("target/cert.json")),
        }
    );
}

#[test]
fn verify_suite_and_check_certificate_reject_invalid_flag_combinations() {
    assert!(parse(&["--check", "--write"]).is_err());
    assert!(parse(&["--strict-parity", "--write"]).is_err());
    assert!(parse(&["--strict-certificate", "--write-tracked"]).is_err());
    assert!(parse(&["--check", "--write-tracked"]).is_err());
    assert!(parse(&["--auto-fix"]).is_err());
    assert!(parse(&["--check", "--auto-fix"]).is_err());
    assert!(parse(&["--strict-parity", "--auto-fix"]).is_err());
    assert!(parse(&["--coverage-only", "--auto-fix"]).is_err());
    assert!(parse(&["--output-cert"]).is_err());
    assert!(parse(&["--output-cert="]).is_err());
    assert!(parse(&["--invalid-flag"]).is_err());
    assert_eq!(parse_check_certificate_mode(&[]), Ok(ReportMode::Check));
    assert_eq!(
        parse_check_certificate_mode(&["--check".to_string()]),
        Ok(ReportMode::Check)
    );
    assert_eq!(
        parse_check_certificate_mode(&["--strict-parity".to_string()]),
        Ok(ReportMode::StrictParity)
    );
    assert_eq!(
        parse_check_certificate_mode(&["--write".to_string()]),
        Ok(ReportMode::Write)
    );
    assert!(
        parse_check_certificate_mode(&["--write".to_string(), "--strict-parity".to_string()])
            .is_err()
    );
    assert!(parse_check_certificate_mode(&["--bad".to_string()]).is_err());
}

#[test]
fn test_verify_suite_coverage_only_skips_static_gates_and_enforces_thresholds() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let temp = TempTestDir::new("verify_suite_cov_only");
    seed_temp_verify_suite_workspace(&temp);
    env::set_current_dir(&*temp).unwrap();

    let members = vec!["crates/demo".to_string()];
    let mut static_calls = 0usize;

    let write_cov = parse(&["--write", "--coverage-only"]).unwrap();
    let mut write_cache = oxidize_core::traceability::AstCache::new();
    let seeded = run_verify_suite_with(
        write_cov,
        &members,
        &mut write_cache,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        |mode| {
            assert_eq!(mode, ReportMode::Write);
            Ok(sample_cov_json(1000, 500))
        },
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(seeded && static_calls == 0);
    assert_eq!(write_cache.parse_calls(), 1);

    let cov_only = parse(&["--coverage-only"]).unwrap();
    let mut low_cache = oxidize_core::traceability::AstCache::new();
    let low = run_verify_suite_with(
        cov_only.clone(),
        &members,
        &mut low_cache,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        |_| Ok(sample_cov_json(1000, 499)),
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(!low && static_calls == 0, "49.9% coverage must fail 50.0%");

    let out_artifact = "target/verification/Verification_Certificate.json";
    let check_with_artifact =
        parse(&["--coverage-only", "--check", "--output-cert", out_artifact]).unwrap();
    let mut ok_cache = oxidize_core::traceability::AstCache::new();
    let ok_cov = run_verify_suite_with(
        check_with_artifact,
        &members,
        &mut ok_cache,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        |_| Ok(sample_cov_json(1000, 920)),
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(
        ok_cov && static_calls == 0,
        "92.0% coverage above ratcheted floor must pass --check even when checked-in cert has 50.0%"
    );
    assert!(std::path::Path::new(out_artifact).is_file());
}

fn run_cov_only_suite(args: &[&str], members: &[String], covered: u64) -> bool {
    run_verify_suite_with(
        parse(args).unwrap(),
        members,
        &mut oxidize_core::traceability::AstCache::new(),
        |_, _, _| Vec::new(),
        |_| Ok(sample_cov_json(1000, covered)),
    )
    .unwrap()
}

#[test]
fn test_verify_suite_check_allows_coverage_shifts_enforces_deterministic_parity_and_emits_15_keys()
{
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let temp = TempTestDir::new("verify_suite_ratchet_check");
    seed_temp_verify_suite_workspace(&temp);
    env::set_current_dir(&*temp).unwrap();

    let members = vec!["crates/demo".to_string()];
    let write_args = ["--write", "--coverage-only"];
    assert!(run_cov_only_suite(&write_args, &members, 820));
    let cert_before = fs::read_to_string("Verification_Certificate.json").unwrap();
    let readme_before = fs::read_to_string("README.md").unwrap();

    let wl = "[verify_suite]\nmin_native_execution_coverage_pct = 80.0\nmin_total_assertion_density = 0.80\nmin_verified_assertion_density = 0.40\nmin_total_production_density = 0.20\nmin_verified_production_density = 0.20\nmin_test_assertion_density = 0.80\ncoverage_tolerance_pct = 1.5\n";
    fs::write("verification_whitelist.toml", wl).unwrap();

    let out_cert = "target/verification/Verification_Certificate.json";
    let check_out = ["--coverage-only", "--check", "--output-cert", out_cert];
    assert!(run_cov_only_suite(&check_out, &members, 910));
    assert_eq!(
        fs::read_to_string("Verification_Certificate.json").unwrap(),
        cert_before
    );
    assert_eq!(fs::read_to_string("README.md").unwrap(), readme_before);
    let live_val: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out_cert).unwrap()).unwrap();
    assert_eq!(live_val.as_object().unwrap().len(), 15);
    assert_eq!(live_val["native_execution_coverage_pct"], 91.0);
    let strict_args = ["--coverage-only", "--strict-parity"];
    assert!(!run_cov_only_suite(&strict_args, &members, 910));

    let drifted_rs = "#[verified_engine::verified]\npub fn step(x: f64) -> f64 { assert!(x.is_finite()); assert!(x >= 0.0); x }\ntheory_verification! { module = \"demo\", test = { assert_eq!(step(1.0), 1.0); assert_eq!(step(2.0), 2.0); } }\n";
    fs::write("crates/demo/src/lib.rs", drifted_rs).unwrap();
    let check_only = ["--coverage-only", "--check"];
    assert!(!run_cov_only_suite(&check_only, &members, 910));
}

#[test]
fn test_verify_suite_check_fails_violated_floors_and_invalid_certificates() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let temp = TempTestDir::new("verify_suite_ratchet_failures");
    seed_temp_verify_suite_workspace(&temp);
    env::set_current_dir(&*temp).unwrap();

    let members = vec!["crates/demo".to_string()];
    let check = ["--coverage-only", "--check"];
    assert!(run_cov_only_suite(
        &["--write", "--coverage-only"],
        &members,
        850
    ));
    let valid_cert = fs::read_to_string("Verification_Certificate.json").unwrap();

    fs::write(
        "verification_whitelist.toml",
        "[verify_suite]\nmin_native_execution_coverage_pct = 80.0\nmin_total_assertion_density = 0.80\n",
    )
    .unwrap();
    assert!(!run_cov_only_suite(&check, &members, 790));

    fs::write(
        "verification_whitelist.toml",
        "[verify_suite]\nmin_native_execution_coverage_pct = 80.0\nmin_total_assertion_density = 2.50\n",
    )
    .unwrap();
    assert!(!run_cov_only_suite(&check, &members, 900));

    fs::write(
        "verification_whitelist.toml",
        "[verify_suite]\nmin_native_execution_coverage_pct = 80.0\n",
    )
    .unwrap();
    let failed_cert = valid_cert.replace("\"passed\": true", "\"passed\": false");
    fs::write("Verification_Certificate.json", failed_cert).unwrap();
    assert!(!run_cov_only_suite(&check, &members, 900));

    fs::write("Verification_Certificate.json", "{malformed-json").unwrap();
    assert!(!run_cov_only_suite(&check, &members, 900));

    fs::remove_file("Verification_Certificate.json").unwrap();
    assert!(!run_cov_only_suite(&check, &members, 900));
}

#[test]
fn test_verify_suite_shares_ast_cache_and_propagates_coverage_errors() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let temp = TempTestDir::new("verify_suite_shared_ast");
    seed_temp_verify_suite_workspace(&temp);
    env::set_current_dir(&*temp).unwrap();

    let members = vec!["crates/demo".to_string()];
    assert!(run_cov_only_suite(
        &["--write", "--coverage-only"],
        &members,
        500
    ));

    let mut static_calls = 0usize;
    let mut shared_cache = oxidize_core::traceability::AstCache::new();
    let full = run_verify_suite_with(
        parse(&[]).unwrap(),
        &members,
        &mut shared_cache,
        |_, _, cache| {
            static_calls += 1;
            let _ = cache.get_or_load_fs(std::path::Path::new("crates/demo/src/lib.rs"));
            Vec::new()
        },
        |_| Ok(sample_cov_json(1000, 510)),
    )
    .expect("default verify-suite should evaluate cleanly");
    assert!(full && static_calls == 1);
    assert_eq!(shared_cache.parse_calls(), 1);
    assert!(
        run_verify_suite_with(
            parse(&["--coverage-only"]).unwrap(),
            &members,
            &mut shared_cache,
            |_, _, _| Vec::new(),
            |_| Err("llvm-cov failed".to_string()),
        )
        .is_err()
    );
}

#[test]
fn test_verify_suite_security_checks_parse_each_member_manifest_once() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, members) = manifest_cache::discover_workspace();
    let members_for_job = members.clone();
    let (passed, entropy_debt, manifests) = crate::with_workspace_ast_cache(move |ast_cache| {
        let member_refs: Vec<&str> = members_for_job.iter().map(String::as_str).collect();
        let mut mc = manifest_cache::ManifestCache::new();
        let (ok, debt) = run_security_checks_with_hooks(
            &member_refs,
            false,
            ast_cache,
            &mut mc,
            || true,
            || true,
        );
        (ok, debt, mc)
    });
    assert!(passed);
    assert!(entropy_debt.is_empty());
    assert_eq!(manifests.lint_check_calls(), members.len());
    assert_eq!(manifests.parse_calls(), members.len() + 1);
    for member in &members {
        let manifest = std::path::Path::new(member).join("Cargo.toml");
        assert_eq!(manifests.parse_count_for(&manifest), 1);
    }
}

#[test]
fn test_workspace_opt_out_caps_enforce_reason_crate_and_file_ceilings() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let rs_files = report::collect_rs_files();
    let suite_metrics = crate::with_workspace_ast_cache(move |cache| {
        report::analyze_files_with_cache(&rs_files, cache)
    });
    let pure_dir = oxidize_core::workspace::find_package_member_path_std(
        &manifest_cache::workspace_root_dir(),
        "pure_math",
    )
    .expect("pure_math member path");
    let expected_evol = format!("{pure_dir}/src/pure_math/analysis/evolution.rs");
    assert!(
        suite_metrics
            .opt_outs
            .iter()
            .any(|(f, r)| f == &expected_evol && r == "infrastructure"),
        "expected cfg_attr opt_out in evolution.rs to be detected"
    );

    let loaded = whitelist::load_whitelist_strict(None, whitelist::WORKSPACE_WHITELIST_CANDIDATES)
        .expect("load workspace whitelist");
    let base_debt = loaded
        .verification_debt
        .expect("[verification_debt] present");
    assert!(
        whitelist::debt::evaluate_opt_out_caps(&base_debt, &suite_metrics.opt_outs).is_empty(),
        "calibrated workspace opt_out caps must pass cleanly"
    );

    let mut tighter_reason = base_debt.clone();
    *tighter_reason.opt_out_reasons.get_mut("gui_tool").unwrap() -= 1;
    assert!(
        !whitelist::debt::evaluate_opt_out_caps(&tighter_reason, &suite_metrics.opt_outs)
            .is_empty()
    );

    let mut tighter_crate = base_debt.clone();
    *tighter_crate
        .crate_opt_out_caps
        .get_mut("math_explorer_gui")
        .unwrap() -= 1;
    assert!(
        !whitelist::debt::evaluate_opt_out_caps(&tighter_crate, &suite_metrics.opt_outs).is_empty()
    );

    let mut tighter_file = base_debt;
    *tighter_file
        .file_opt_out_caps
        .get_mut("math_explorer_gui:src/tabs/paradox_lab/mod.rs")
        .unwrap() = 2;
    assert!(
        !whitelist::debt::evaluate_opt_out_caps(&tighter_file, &suite_metrics.opt_outs).is_empty()
    );
}
