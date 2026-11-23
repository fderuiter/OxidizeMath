use super::*;

#[test]
fn test_parse_coverage_empty_and_valid() {
    let (tot, cov) = parse_coverage(&serde_json::json!({}));
    assert_eq!(tot, 0.0);
    assert_eq!(cov, 0.0);

    let data = serde_json::json!({
        "data": [{
            "files": [
                {
                    "summary": {
                        "lines": {
                            "count": 100,
                            "covered": 80
                        }
                    }
                },
                {
                    "summary": {
                        "lines": {
                            "count": 50,
                            "covered": 20
                        }
                    }
                }
            ]
        }]
    });
    let (tot2, cov2) = parse_coverage(&data);
    assert_eq!(tot2, 150.0);
    assert_eq!(cov2, 100.0);
}

#[test]
fn test_print_report_threshold_enforcement() {
    let metrics = FileMetrics {
        wasm_paths: 0,
        wasm_covered: 0,
        total_funcs: 10.0,
        total_asserts: 10.0,
        verified_funcs: 5.0,
        verified_asserts: 5.0,
        unverified_funcs: 5.0,
        unverified_asserts: 5.0,
        opt_outs: Vec::new(),
        orphaned_wasm_tests: Vec::new(),
    };

    // Coverage below 50.0% should fail
    let passed_low = print_report(49.9, 100.0, &metrics, &[]);
    assert!(!passed_low, "Coverage below 50.0% must fail");

    // Coverage at or above 50.0% with valid density should pass
    let passed_high = print_report(55.0, 100.0, &metrics, &[]);
    assert!(passed_high, "Coverage at or above 50.0% should pass");
    let _ = fs::remove_file("Verification_Certificate.json");
}

#[test]
fn test_collect_rs_files_and_unverified_modules() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let files = collect_rs_files();
    assert!(!files.is_empty());
    let mut saw_egui_plot = false;
    for f in &files {
        assert!(f.extension().and_then(|s| s.to_str()) == Some("rs"));
        let p_str = crate::whitelist::lossy_slash_path(f);
        assert!(!p_str.contains("/target/"));
        if p_str.contains("egui_plot/src/file_io.rs") {
            saw_egui_plot = true;
        }
    }
    if Path::new("egui_plot/src/file_io.rs").is_file() {
        assert!(
            saw_egui_plot,
            "collect_rs_files must include egui_plot sources for WASM coverage"
        );
    }

    let unverified =
        check_unverified_modules(&["egui_plot".to_string(), "crates/pure_math".to_string()]);
    assert!(
        !unverified
            .iter()
            .any(|u| u.contains("egui_plot") || u.contains("crates/pure_math"))
    );
}

fn empty_file_metrics() -> FileMetrics {
    FileMetrics {
        wasm_paths: 0,
        wasm_covered: 0,
        total_funcs: 0.0,
        total_asserts: 0.0,
        verified_funcs: 0.0,
        verified_asserts: 0.0,
        unverified_funcs: 0.0,
        unverified_asserts: 0.0,
        opt_outs: Vec::new(),
        orphaned_wasm_tests: Vec::new(),
    }
}

#[test]
fn test_wasm_coverage_ignores_theory_and_attribute_comments() {
    let patterns = ReportPatterns::new();
    let comment_only_src = r#"
        #[cfg(target_arch = "wasm32")]
        // theory_verification!
        // stochastic_signature_verification!
        // #[wasm_bindgen_test]
        fn mark_boot_initialized() {
            let _marker = "theory_verification! #[wasm_bindgen_test]";
        }

        #[cfg(target_arch = "wasm32")]
        // theory_verification!
        fn handle_webgl_failure() {}
    "#;
    let mut m = empty_file_metrics();
    analyze_file_content(
        Path::new("math_explorer_gui/src/comment_only_probe.rs"),
        comment_only_src,
        &patterns,
        &mut m,
    );
    assert_eq!(m.wasm_paths, 2);
    assert_eq!(
        m.wasm_covered, 0,
        "// theory_verification! comments must never count as covered WASM paths"
    );
}

#[test]
fn test_wasm_coverage_counts_real_wasm_bindgen_tests() {
    let patterns = ReportPatterns::new();
    let tested_src = r#"
        #[cfg(target_arch = "wasm32")]
        pub fn browser_helper() -> u32 { 42 }

        #[cfg(all(test, target_arch = "wasm32"))]
        mod wasm_tests {
            use wasm_bindgen_test::wasm_bindgen_test;

            #[wasm_bindgen_test]
            fn verifies_browser_helper() {
                assert_eq!(super::browser_helper(), 42);
            }
        }
    "#;
    let mut m = empty_file_metrics();
    analyze_file_content(
        Path::new("math_explorer_gui/src/tested_probe.rs"),
        tested_src,
        &patterns,
        &mut m,
    );
    assert_eq!(m.wasm_paths, 1);
    assert_eq!(m.wasm_covered, 1);
}

#[test]
fn test_collect_llvm_cov_json_cleans_only_after_parsing() {
    let ok_status = Command::new("cargo")
        .arg("--version")
        .status()
        .expect("cargo --version must succeed");
    let mut cleaned = false;
    let parsed = collect_llvm_cov_json_with(
        || std::process::Output {
            status: ok_status,
            stdout: br#"{"data":[{"files":[{"summary":{"lines":{"count":80,"covered":60}}}]}]}"#
                .to_vec(),
            stderr: Vec::new(),
        },
        || {
            cleaned = true;
            Ok(())
        },
    )
    .expect("valid coverage JSON should parse and clean");
    assert!(cleaned, "llvm-cov clean must run after JSON parsing");
    assert_eq!(parse_coverage(&parsed), (80.0, 60.0));

    let mut cleaned_on_bad_json = false;
    let bad_res = collect_llvm_cov_json_with(
        || std::process::Output {
            status: ok_status,
            stdout: b"not-json".to_vec(),
            stderr: Vec::new(),
        },
        || {
            cleaned_on_bad_json = true;
            Ok(())
        },
    );
    assert!(bad_res.is_err());
    assert!(!cleaned_on_bad_json);
}

#[test]
fn test_report_rejects_orphaned_wasm32_unit_tests() {
    let patterns = ReportPatterns::new();
    let bad_src = r#"
        #[cfg(target_arch = "wasm32")]
        #[test]
        fn orphaned_browser_test() {
            assert_eq!(1, 1);
        }
    "#;
    let mut m = empty_file_metrics();
    m.verified_funcs = 2.0;
    m.verified_asserts = 2.0;
    analyze_file_content(
        Path::new("egui_plot/src/probe.rs"),
        bad_src,
        &patterns,
        &mut m,
    );
    assert_eq!(m.orphaned_wasm_tests.len(), 1);
    assert!(!print_report(85.0, 100.0, &m, &[]));
    let _ = fs::remove_file("Verification_Certificate.json");
}
