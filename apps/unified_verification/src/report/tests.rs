use super::coverage::collect_llvm_cov_json_with;
use super::*;
use crate::whitelist::{OptOutAllowEntry, VerifySuiteConfig};
use oxidize_core::traceability::AstCache;
use std::path::Path;
use std::process::Command;

fn print_report(
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

pub(super) fn baseline_metrics() -> FileMetrics {
    FileMetrics {
        wasm_paths: 2,
        wasm_covered: 1,
        total_funcs: 10.0,
        total_asserts: 10.0,
        verified_funcs: 5.0,
        verified_asserts: 5.0,
        unverified_funcs: 5.0,
        unverified_asserts: 5.0,
        prod_funcs: 6.0,
        prod_asserts: 3.0,
        verified_prod_funcs: 4.0,
        verified_prod_asserts: 2.0,
        test_funcs: 4.0,
        test_asserts: 7.0,
        opt_outs: Vec::new(),
        legacy_funcs: Vec::new(),
    }
}

fn analyze_snippet(rel_path: &str, content: &str) -> FileMetrics {
    let mut cache = AstCache::new();
    let key = AstCache::normalize_key(rel_path);
    cache.insert_source(key, content.to_string());
    analyze_files_with_cache(&[Path::new(rel_path).to_path_buf()], &mut cache)
}

#[test]
fn test_parse_coverage_empty_and_valid() {
    let (tot, cov) = parse_coverage(&serde_json::json!({}));
    assert_eq!(tot, 0.0);
    assert_eq!(cov, 0.0);

    let data = serde_json::json!({
        "data": [{
            "files": [
                { "summary": { "lines": { "count": 100, "covered": 80 } } },
                { "summary": { "lines": { "count": 50, "covered": 20 } } }
            ]
        }]
    });
    let (tot2, cov2) = parse_coverage(&data);
    assert_eq!(tot2, 150.0);
    assert_eq!(cov2, 100.0);
}

#[test]
fn test_print_report_threshold_and_fail_closed_debt() {
    let metrics = baseline_metrics();
    assert!(!print_report(49.9, 100.0, &metrics, &[]));
    assert!(print_report(55.0, 100.0, &metrics, &[]));
    assert!(
        !print_report(
            55.0,
            100.0,
            &metrics,
            &["File too long: foo.rs".to_string()]
        ),
        "Any unauthorized debt item must fail print_report"
    );
}

#[test]
fn test_fail_closed_gates_reject_unwhitelisted_modules_opt_outs_and_legacy_fns() {
    let eval = evaluate_integrity_gates;
    let cfg = VerifySuiteConfig {
        min_total_production_density: 0.20,
        min_verified_production_density: 0.25,
        unverified_modules: vec!["crates/oxidize_core".into()],
        opt_outs: vec![OptOutAllowEntry {
            path: "math_explorer_gui/src/".into(),
            reason: "gui_tool".into(),
        }],
        ..VerifySuiteConfig::default()
    };
    let mut m = baseline_metrics();
    m.opt_outs
        .push(("math_explorer_gui/src/app.rs".into(), "gui_tool".into()));
    let allowed = vec!["crates/oxidize_core".to_string()];
    let debt = DebtSources {
        unverified_crates: &allowed,
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: &[],
    };
    assert!(eval(80.0, 50.0, &m, &debt, &cfg).report.passed);

    let rogue = vec!["crates/domain_rogue".to_string()];
    let rogue_debt = DebtSources {
        unverified_crates: &rogue,
        ..debt
    };
    assert!(!eval(80.0, 50.0, &m, &rogue_debt, &cfg).report.passed);

    let mut bad_opt = m.clone();
    bad_opt
        .opt_outs
        .push(("crates/pure_math/src/lib.rs".into(), "x".into()));
    assert!(!eval(80.0, 50.0, &bad_opt, &debt, &cfg).report.passed);

    let mut leg = m.clone();
    leg.legacy_funcs
        .push(("crates/pure_math/src/lib.rs".into(), "f_legacy".into()));
    assert!(!eval(80.0, 50.0, &leg, &debt, &cfg).report.passed);
}

#[test]
fn test_evaluate_integrity_gates_enforces_ratcheted_floors() {
    let m = baseline_metrics();
    let cfg = VerifySuiteConfig {
        min_native_execution_coverage_pct: 75.0,
        min_wasm_path_coverage_pct: 45.0,
        min_total_assertion_density: 0.90,
        min_total_production_density: 0.20,
        min_verified_production_density: 0.25,
        min_test_assertion_density: 1.50,
        ..VerifySuiteConfig::default()
    };
    let debt = DebtSources {
        unverified_crates: &[],
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: &[],
    };
    assert!(
        evaluate_integrity_gates(80.0, 50.0, &m, &debt, &cfg)
            .report
            .passed
    );

    for bad_floor_cfg in [
        VerifySuiteConfig {
            min_verified_production_density: 0.90,
            ..cfg.clone()
        },
        VerifySuiteConfig {
            min_native_execution_coverage_pct: 85.0,
            ..cfg.clone()
        },
        VerifySuiteConfig {
            min_wasm_path_coverage_pct: 70.0,
            ..cfg.clone()
        },
        VerifySuiteConfig {
            min_total_assertion_density: 1.50,
            ..cfg.clone()
        },
        VerifySuiteConfig {
            min_test_assertion_density: 2.50,
            ..cfg
        },
    ] {
        assert!(
            !evaluate_integrity_gates(80.0, 50.0, &m, &debt, &bad_floor_cfg)
                .report
                .passed
        );
    }
}

#[test]
fn test_collect_rs_files_and_unverified_modules() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let orig = std::env::current_dir().unwrap();
    let _guard = crate::utils::test_utils::CurrentDirGuard(orig);
    if !Path::new("crates").is_dir() && Path::new("../../crates").is_dir() {
        std::env::set_current_dir("../..").unwrap();
    }
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

    let pure_dir =
        oxidize_core::workspace::find_package_member_path_std(Path::new("."), "pure_math").unwrap();
    let core_dir =
        oxidize_core::workspace::find_package_member_path_std(Path::new("."), "oxidize_core")
            .unwrap();
    let unverified =
        check_unverified_modules(&["egui_plot".to_string(), pure_dir.clone(), core_dir.clone()]);
    assert!(
        !unverified
            .iter()
            .any(|u| u.contains("egui_plot") || u.contains(&pure_dir))
    );
    assert!(
        unverified.iter().any(|u| u.contains(&core_dir)),
        "oxidize_core has no real verification macro once comment spoofs are ignored"
    );
}

#[test]
fn test_ast_metrics_ignore_comments_and_empty_macro_floors() {
    let spoof_src = concat!(
        "#[cfg(target_arch = \"wasm32\")]\n",
        "// theory_",
        "verification!\n",
        "// #[wasm_bindgen_test]\n",
        "fn mark_boot_initialized() {\n",
        "    let _s = \"#[verified] opt_out = \\\"spoof\\\" fn foo_legacy()\";\n",
        "}\n",
        "crate::theory_verification! { module = \"empty_mod\", test = {} }\n"
    );
    let m = analyze_snippet("math_explorer_gui/src/comment_only_probe.rs", spoof_src);
    assert_eq!(m.wasm_paths, 1);
    assert_eq!(m.wasm_covered, 0);
    assert_eq!(m.verified_funcs, 1.0);
    assert_eq!(m.verified_asserts, 0.0);
    assert!(m.opt_outs.is_empty());
    assert!(m.legacy_funcs.is_empty());
}

#[test]
fn test_wasm_coverage_counts_real_wasm_bindgen_tests() {
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
    let m = analyze_snippet("math_explorer_gui/src/tested_probe.rs", tested_src);
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
fn test_comment_markers_in_vfs_and_accessibility_never_mark_files_or_crates_verified() {
    let _lock = crate::DIR_LOCK.lock().unwrap();
    let ws_root = crate::manifest_cache::workspace_root_dir();
    let core_rel = oxidize_core::workspace::find_package_member_path_std(&ws_root, "oxidize_core")
        .expect("oxidize_core member path");
    let gui_rel =
        oxidize_core::workspace::find_package_member_path_std(&ws_root, "math_explorer_gui")
            .expect("math_explorer_gui member path");
    let vfs_path = ws_root.join(&core_rel).join("src/vfs.rs");
    let acc_path = ws_root.join(&gui_rel).join("src/accessibility.rs");

    let mut cache = AstCache::new();
    let file_metrics = analyze_files_with_cache(&[vfs_path, acc_path], &mut cache);
    assert_eq!(
        file_metrics.verified_funcs, 0.0,
        "comment-only markers in vfs.rs and accessibility.rs must not mark functions verified"
    );

    let crates_to_probe = vec![
        ws_root.join(&core_rel).to_string_lossy().into_owned(),
        ws_root.join(&gui_rel).to_string_lossy().into_owned(),
    ];
    let unverified = find_unverified_crates_with_cache(&crates_to_probe, &mut cache);
    assert_eq!(
        unverified.len(),
        2,
        "// theory_verification! comments in vfs.rs and accessibility.rs must not spoof crate verification: {unverified:?}"
    );
}
