#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_web_manifest_relative_urls_and_tex_fallback_links() {
    assert_eq!(
        web_snapshot::MANIFEST_REL_URL,
        "./traceability_manifest.json"
    );
    assert!(web_snapshot::is_relative_same_origin_url(
        web_snapshot::MANIFEST_REL_URL
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        "/traceability.toml"
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        "./../traceability.toml"
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        "https://example.com/traceability_manifest.json"
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        ".\\traceability_manifest.json"
    ));

    assert_eq!(
        web_snapshot::paper_tex_url("partitions_implementation.tex"),
        Some(
            "https://github.com/fderuiter/OxidizeMath/blob/main/papers/partitions_implementation.tex"
                .to_string()
        )
    );
    assert_eq!(
        web_snapshot::paper_tex_url("papers/cera_framework.tex"),
        Some(
            "https://github.com/fderuiter/OxidizeMath/blob/main/papers/cera_framework.tex"
                .to_string()
        )
    );
    assert_eq!(
        web_snapshot::paper_tex_url("win_ratio.pdf"),
        Some("https://github.com/fderuiter/OxidizeMath/blob/main/papers/win_ratio.tex".to_string())
    );
    assert_eq!(web_snapshot::paper_tex_url("spec:transformer_v1"), None);
    assert_eq!(web_snapshot::paper_tex_url("registry:internal"), None);

    const SW_JS: &str = include_str!("../../../sw.js");
    assert!(
        !SW_JS.contains("traceability_manifest.json")
            && !SW_JS.contains("Verification_Certificate.json"),
        "sw.js must keep traceability JSON out of install precache so startup makes zero traceability requests"
    );
}

#[test]
fn test_invalid_manifest_payload_errors_and_bare_report_deserializes() {
    assert!(web_snapshot::parse_manifest_json("not valid json").is_err());

    let mut report = oxidize_core::traceability::TraceabilityReport {
        scanned_files: 42,
        ..Default::default()
    };
    report
        .paper_coverage
        .insert("essay.tex".to_string(), vec!["math_commons".to_string()]);
    let raw = serde_json::to_string(&report).expect("serialize TraceabilityReport");
    let (parsed, cert) = web_snapshot::parse_manifest_json(&raw).expect("parse bare report");
    assert_eq!(parsed, report);
    assert!(cert.is_none());
}

#[test]
fn test_app_startup_does_not_scan_when_traceability_tab_unopened() {
    use eframe::App;

    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut app = crate::app::MathExplorerApp::default();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| app.update(ctx, &mut frame));

    let state = app.get_app_state();
    assert_ne!(
        state.selected_tab_name.as_deref(),
        Some("Traceability Portal"),
        "Default landing tab should not eagerly open Traceability Portal"
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_generated_manifest_matches_live_workspace_scan_for_all_28_papers() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let _cwd_guard = super::workspace_root_tests::CwdRestoreGuard::capture();
    let ws_root = TraceabilityTab::canonical_workspace_root().expect("workspace root");
    let ws_norm = oxidize_core::path_utils::normalize_path(&ws_root);
    std::env::set_current_dir(&ws_norm).expect("chdir workspace root");

    let embedded_json = include_str!(concat!(env!("OUT_DIR"), "/traceability_manifest.json"));
    let direct_report: oxidize_core::traceability::TraceabilityReport =
        serde_json::from_str(embedded_json).expect("deserialize TraceabilityReport directly");
    let (manifest_report, embedded_cert) =
        web_snapshot::parse_manifest_json(embedded_json).expect("valid manifest JSON");
    assert_eq!(direct_report, manifest_report);

    let live_report = futures::executor::block_on(execute_workspace_scan(
        oxidize_core::vfs::DefaultVfs,
        String::new(),
    ))
    .expect("live workspace scan");

    drop(_cwd_guard);

    let tex_paper_count = manifest_report
        .paper_coverage
        .keys()
        .filter(|k| k.ends_with(".tex"))
        .count();
    assert_eq!(
        tex_paper_count, 29,
        "Manifest must cover all 29 .tex papers"
    );
    assert!(manifest_report.orphaned_papers.is_empty());
    assert!(manifest_report.unlinked_code.is_empty());
    assert!(manifest_report.invalid_links.is_empty());
    assert!(manifest_report.total_funcs > 0);
    assert!(manifest_report.total_asserts > 0);
    assert!(manifest_report.verified_funcs > 0);
    assert!(manifest_report.verified_asserts > 0);
    assert_eq!(manifest_report, live_report);

    let cert = embedded_cert.expect("embedded Verification_Certificate.json");
    assert!(cert.passed);
    assert!(cert.wasm_path_coverage_pct > 0.0);
}
