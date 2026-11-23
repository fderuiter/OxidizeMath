use super::certificate::{
    format_coverage_badge, verify_certificate_against_report, write_certificate_and_readme,
};
use super::*;
use crate::utils::test_utils::TempTestDir;
use crate::whitelist::{OptOutAllowEntry, VerifySuiteConfig};
use oxidize_core::traceability::AstCache;
use std::fs;
use std::path::Path;

fn analyze_snippet(rel_path: &str, content: &str) -> FileMetrics {
    let mut cache = AstCache::new();
    let key = AstCache::normalize_key(rel_path);
    cache.insert_source(key, content.to_string());
    analyze_files_with_cache(&[Path::new(rel_path).to_path_buf()], &mut cache)
}

fn baseline_metrics() -> FileMetrics {
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

    let strict = VerifySuiteConfig {
        min_verified_production_density: 0.90,
        ..cfg
    };
    assert!(!eval(80.0, 50.0, &m, &debt, &strict).report.passed);
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

    let unverified = check_unverified_modules(&[
        "egui_plot".to_string(),
        "crates/pure_math".to_string(),
        "crates/oxidize_core".to_string(),
    ]);
    assert!(
        !unverified
            .iter()
            .any(|u| u.contains("egui_plot") || u.contains("crates/pure_math"))
    );
    assert!(
        unverified.iter().any(|u| u.contains("crates/oxidize_core")),
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
fn test_certificate_check_mode_is_read_only_and_enforces_parity() {
    let dir = TempTestDir::new("cert_parity");
    let cert_path = dir.join("Verification_Certificate.json");
    let readme_path = dir.join("README.md");
    fs::write(
        &readme_path,
        "# Demo\n![Coverage](https://img.shields.io/badge/coverage-10.0%25-yellow)\n",
    )
    .unwrap();

    let metrics = baseline_metrics();
    let cfg = VerifySuiteConfig::default();
    let empty_debt = DebtSources {
        unverified_crates: &[],
        file_length_debt: &[],
        entropy_debt: &[],
        extra_debt: &[],
    };
    let eval = evaluate_integrity_gates(84.5, 50.0, &metrics, &empty_debt, &cfg);
    write_certificate_and_readme(&eval.report, &cert_path, &readme_path).unwrap();

    let cert_before = fs::read_to_string(&cert_path).unwrap();
    let readme_before = fs::read_to_string(&readme_path).unwrap();
    assert!(readme_before.contains(&format_coverage_badge(84.5)));

    assert!(
        verify_certificate_against_report(&eval.report, Some(85.2), 1.5, &cert_path, &readme_path)
            .is_ok(),
        "Coverage jitter within 1.5% tolerance must pass"
    );
    assert!(
        verify_certificate_against_report(&eval.report, Some(87.0), 1.5, &cert_path, &readme_path)
            .is_err(),
        "Coverage drift beyond 1.5% tolerance must fail"
    );

    let mut drifted = eval.report.clone();
    drifted.verified_modules_density += 0.01;
    assert!(
        verify_certificate_against_report(&drifted, None, 1.5, &cert_path, &readme_path).is_err(),
        "Deterministic AST density drift must fail check mode"
    );
    assert_eq!(fs::read_to_string(&cert_path).unwrap(), cert_before);
    assert_eq!(fs::read_to_string(&readme_path).unwrap(), readme_before);
}
