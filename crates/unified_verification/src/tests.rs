use super::*;
use crate::utils::test_utils::CurrentDirGuard;

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

#[test]
fn test_parse_verify_suite_args_flags_and_unknown_rejection() {
    assert_eq!(
        parse_verify_suite_args(&[]),
        Ok(VerifySuiteOptions {
            auto_fix: false,
            coverage_only: false,
            output_cert: None,
            write_tracked: false,
        })
    );
    assert_eq!(
        parse_verify_suite_args(&["--coverage-only".to_string()]),
        Ok(VerifySuiteOptions {
            auto_fix: false,
            coverage_only: true,
            output_cert: None,
            write_tracked: false,
        })
    );
    assert_eq!(
        parse_verify_suite_args(&["--auto-fix".to_string(), "--coverage-only".to_string()]),
        Ok(VerifySuiteOptions {
            auto_fix: true,
            coverage_only: true,
            output_cert: None,
            write_tracked: false,
        })
    );
    assert_eq!(
        parse_verify_suite_args(&[
            "--coverage-only".to_string(),
            "--output-cert".to_string(),
            "target/verification/Verification_Certificate.json".to_string(),
            "--write-tracked".to_string(),
        ]),
        Ok(VerifySuiteOptions {
            auto_fix: false,
            coverage_only: true,
            output_cert: Some(std::path::PathBuf::from(
                "target/verification/Verification_Certificate.json"
            )),
            write_tracked: true,
        })
    );
    assert!(parse_verify_suite_args(&["--output-cert".to_string()]).is_err());
    assert!(parse_verify_suite_args(&["--output-cert=".to_string()]).is_err());
    assert!(parse_verify_suite_args(&["--invalid-flag".to_string()]).is_err());
}

#[test]
fn test_verify_suite_coverage_only_skips_static_gates_and_enforces_50_pct_threshold() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, members) = manifest_cache::discover_workspace();
    let cov_only = parse_verify_suite_args(&["--coverage-only".to_string()]).unwrap();
    let mut static_calls = 0usize;

    let low = run_verify_suite_with(
        cov_only.clone(),
        &members,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        || Ok(sample_cov_json(1000, 499)),
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(!low, "49.9% coverage must fail 50.0% gate");
    assert_eq!(
        static_calls, 0,
        "--coverage-only must skip static/OSV gates"
    );

    let ok_cov = run_verify_suite_with(
        cov_only.clone(),
        &members,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        || Ok(sample_cov_json(1000, 500)),
    )
    .expect("coverage JSON should evaluate cleanly");
    assert!(ok_cov && static_calls == 0, "50.0% coverage must pass");

    let default_opts = parse_verify_suite_args(&[]).unwrap();
    let full = run_verify_suite_with(
        default_opts,
        &members,
        |_, _, _| {
            static_calls += 1;
            Vec::new()
        },
        || Ok(sample_cov_json(1000, 750)),
    )
    .expect("default verify-suite should evaluate cleanly");
    assert!(full && static_calls == 1);
    assert!(
        run_verify_suite_with(
            cov_only,
            &members,
            |_, _, _| Vec::new(),
            || Err("llvm-cov failed".to_string()),
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
    let member_refs: Vec<&str> = members.iter().map(String::as_str).collect();
    let mut ast_cache = oxidize_core::traceability::AstCache::new();
    let mut manifests = manifest_cache::ManifestCache::new();

    let (passed, entropy_debt) = run_security_checks_with_hooks(
        &member_refs,
        false,
        &mut ast_cache,
        &mut manifests,
        || true,
        || true,
    );
    assert!(passed);
    assert!(entropy_debt.is_empty());
    assert_eq!(manifests.lint_check_calls(), members.len());
    assert_eq!(manifests.parse_calls(), members.len() + 1);
    for member in &members {
        let manifest = std::path::Path::new(member).join("Cargo.toml");
        assert_eq!(manifests.parse_count_for(&manifest), 1);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_verify_suite_and_traceability_report_identical_ast_metrics() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let mut cache = oxidize_core::traceability::AstCache::new();
    let code_dirs = oxidize_core::traceability::discover_workspace_code_dirs();
    let dir_refs: Vec<&str> = code_dirs.iter().map(String::as_str).collect();
    let engine = oxidize_core::traceability::TraceabilityEngine::new(oxidize_core::vfs::DefaultVfs);
    let trace_report = {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        let fut = std::pin::pin!(
            engine.scan_repository_with_cache(&dir_refs, "papers", false, &mut cache)
        );
        let std::task::Poll::Ready(res) = std::future::Future::poll(fut, &mut cx) else {
            panic!("DefaultVfs future must resolve immediately");
        };
        res.unwrap_or_else(|err| panic!("traceability scan should succeed: {err}"))
    };

    let rs_files = report::collect_rs_files();
    let suite_metrics = report::analyze_workspace_with_cache(&rs_files, &mut cache)
        .expect("workspace AST report metrics");

    assert_eq!(
        suite_metrics.verified_funcs as usize,
        trace_report.verified_funcs
    );
    assert_eq!(
        suite_metrics.verified_asserts as usize,
        trace_report.verified_asserts
    );
    assert_eq!(suite_metrics.total_funcs as usize, trace_report.total_funcs);
    assert_eq!(
        suite_metrics.total_asserts as usize,
        trace_report.total_asserts
    );
    assert!(
        suite_metrics.opt_outs.len() >= 605,
        "expected all cfg_attr opt-outs (>= 605, currently 753) to be detected, got {}",
        suite_metrics.opt_outs.len()
    );
    assert!(
        suite_metrics.opt_outs.iter().any(|(f, r)| {
            f == "crates/pure_math/src/pure_math/analysis/evolution.rs" && r == "infrastructure"
        }),
        "expected cfg_attr opt_out in evolution.rs to be detected"
    );
}

#[test]
fn test_verify_suite_enforces_reason_crate_and_file_opt_out_caps() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let mut cache = oxidize_core::traceability::AstCache::new();
    let rs_files = report::collect_rs_files();
    let suite_metrics = report::analyze_workspace_with_cache(&rs_files, &mut cache)
        .expect("workspace AST report metrics");
    let loaded = whitelist::load_whitelist_strict(None, whitelist::WORKSPACE_WHITELIST_CANDIDATES)
        .expect("load workspace whitelist");
    let base_debt = loaded
        .verification_debt
        .expect("[verification_debt] present");

    let mut tighter_reason = base_debt.clone();
    *tighter_reason.opt_out_reasons.get_mut("gui_tool").unwrap() -= 1;
    assert!(
        !report::print_report_with_options(
            80.0,
            100.0,
            &suite_metrics,
            &[],
            Some(&tighter_reason),
            &report::ReportWriteOptions::default(),
        )
        .unwrap()
    );

    let mut tighter_crate = base_debt.clone();
    *tighter_crate
        .crate_opt_out_caps
        .get_mut("math_explorer_gui")
        .unwrap() -= 1;
    assert!(
        !report::print_report_with_options(
            80.0,
            100.0,
            &suite_metrics,
            &[],
            Some(&tighter_crate),
            &report::ReportWriteOptions::default(),
        )
        .unwrap()
    );

    let mut tighter_file = base_debt;
    *tighter_file
        .file_opt_out_caps
        .get_mut("math_explorer_gui/src/tabs/paradox_lab/mod.rs")
        .unwrap() = 2;
    assert!(
        !report::print_report_with_options(
            80.0,
            100.0,
            &suite_metrics,
            &[],
            Some(&tighter_file),
            &report::ReportWriteOptions::default(),
        )
        .unwrap()
    );
}

#[test]
fn test_verify_suite_default_and_coverage_only_both_fail_on_unwhitelisted_debt() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, mut members_with_fake) = manifest_cache::discover_workspace();
    let fake_dir = crate::utils::test_utils::TempTestDir::new("unwhitelisted_crate_probe");
    std::fs::create_dir_all(fake_dir.join("src")).unwrap();
    std::fs::write(
        fake_dir.join("src/lib.rs"),
        "// theory_verification!\npub fn unverified_fn() -> u32 { 1 }\n",
    )
    .unwrap();
    members_with_fake.push(fake_dir.to_string_lossy().into_owned());

    let cov_only = parse_verify_suite_args(&["--coverage-only".to_string()]).unwrap();
    let cov_res = run_verify_suite_with(
        cov_only,
        &members_with_fake,
        |_, _, _| Vec::new(),
        || Ok(sample_cov_json(1000, 850)),
    )
    .unwrap();
    assert!(
        !cov_res,
        "verify-suite --coverage-only must return passed = false on unwhitelisted crate debt"
    );

    let default_opts = parse_verify_suite_args(&[]).unwrap();
    let full_res = run_verify_suite_with(
        default_opts,
        &members_with_fake,
        |_, _, _| Vec::new(),
        || Ok(sample_cov_json(1000, 850)),
    )
    .unwrap();
    assert!(
        !full_res,
        "default verify-suite must return passed = false on unwhitelisted crate debt"
    );
}

#[test]
fn test_verify_suite_output_cert_writes_valid_live_certificate_schema() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let _guard = CurrentDirGuard(env::current_dir().unwrap());
    let root = manifest_cache::workspace_root_dir();
    if root != std::path::Path::new(".") {
        env::set_current_dir(root).unwrap();
    }

    let (_, members) = manifest_cache::discover_workspace();
    let out_path = "target/verification/test_ci_Verification_Certificate.json";
    let _ = std::fs::remove_file(out_path);
    let opts = parse_verify_suite_args(&[
        "--coverage-only".to_string(),
        "--output-cert".to_string(),
        out_path.to_string(),
    ])
    .unwrap();
    let passed = run_verify_suite_with(
        opts,
        &members,
        |_, _, _| Vec::new(),
        || Ok(sample_cov_json(1000, 850)),
    )
    .unwrap();
    assert!(passed);
    let parsed: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(out_path).unwrap()).unwrap();
    assert_eq!(parsed["passed"], serde_json::Value::Bool(true));
    assert!(parsed["verified_modules_density"].as_f64().unwrap() >= 0.060);
    assert!(parsed["total_assertion_density"].as_f64().unwrap() >= 0.020);
}
