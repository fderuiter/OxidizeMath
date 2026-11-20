#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_web_snapshot_relative_urls_and_tex_fallback_links() {
    assert!(web_snapshot::is_relative_same_origin_url(
        web_snapshot::SNAPSHOT_REL_URL
    ));
    assert!(web_snapshot::is_relative_same_origin_url(
        web_snapshot::CERTIFICATE_REL_URL
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        "/traceability.toml"
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        "./../traceability.toml"
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        "https://example.com/traceability_snapshot.json"
    ));
    assert!(!web_snapshot::is_relative_same_origin_url(
        ".\\traceability_snapshot.json"
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
        !SW_JS.contains("traceability_snapshot.json")
            && !SW_JS.contains("Verification_Certificate.json"),
        "sw.js must keep traceability JSON out of install precache so startup makes zero traceability requests"
    );
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
fn test_generated_snapshot_matches_live_workspace_scan_and_certificate_parses() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ws_root = TraceabilityTab::canonical_workspace_root().expect("workspace root");
    let orig_cwd = std::env::current_dir().expect("cwd");
    std::env::set_current_dir(&ws_root).expect("chdir workspace root");

    let embedded_json = include_str!(concat!(env!("OUT_DIR"), "/traceability_snapshot.json"));
    let snap_report =
        web_snapshot::parse_snapshot_json(embedded_json).expect("valid snapshot JSON");

    let live_report = futures::executor::block_on(execute_workspace_scan(
        oxidize_core::vfs::DefaultVfs,
        String::new(),
    ))
    .expect("live workspace scan");

    std::env::set_current_dir(orig_cwd).expect("restore cwd");

    assert_eq!(snap_report.scanned_files, live_report.scanned_files);
    assert_eq!(snap_report.orphaned_papers, live_report.orphaned_papers);
    assert_eq!(snap_report.unlinked_code, live_report.unlinked_code);
    assert_eq!(snap_report.invalid_links, live_report.invalid_links);
    assert_eq!(
        snap_report.paper_coverage.len(),
        live_report.paper_coverage.len()
    );
    for (paper, mut live_mods) in live_report.paper_coverage {
        live_mods.sort();
        live_mods.dedup();
        let snap_mods = snap_report
            .paper_coverage
            .get(&paper)
            .unwrap_or_else(|| panic!("Missing paper {paper} in snapshot"));
        assert_eq!(snap_mods, &live_mods, "Coverage mismatch for {paper}");
    }

    let cert_raw = std::fs::read_to_string(ws_root.join("Verification_Certificate.json"))
        .expect("read Verification_Certificate.json");
    let cert = web_snapshot::parse_certificate_json(&cert_raw).expect("valid certificate JSON");
    assert!(cert.passed);
    assert!(cert.wasm_path_coverage_pct > 0.0);
}
