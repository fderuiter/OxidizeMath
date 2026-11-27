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

    const SW_JS: &str = include_str!("../../../../sw.js");
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
        tex_paper_count, 32,
        "Manifest must cover all 32 .tex papers"
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
    assert_eq!(cert.certified_solvers, 55);
    assert_eq!(cert.tier_counts.len(), 6);
    assert!(cert.tier_counts.values().all(|&c| c > 0));
    assert!(!cert.interval_bounds.is_empty());
    assert!(cert.lean4_proof_digests.len() >= 6);
    assert!(cert.lean4_proof_bindings.len() >= 6);
    assert!(cert
        .lean4_proof_bindings
        .values()
        .all(|b| b.kernel_verified && !b.rust_modules.is_empty() && !b.theorems.is_empty()));
    assert_eq!(manifest_report.tier_counts.len(), 6);
    assert!(manifest_report.tier_counts.values().all(|&c| c > 0));
    assert_eq!(manifest_report.lean4_proof_bindings.len(), 6);
    assert_eq!(
        web_snapshot::lean_proof_url("IntervalSoundness.lean"),
        Some(
            "https://github.com/fderuiter/OxidizeMath/blob/main/proofs/OxidizeMath/IntervalSoundness.lean"
                .to_string()
        )
    );
}

fn append_shape_text(shape: &egui::epaint::Shape, dst: &mut String) {
    match shape {
        egui::epaint::Shape::Text(t) => {
            dst.push_str(t.galley.text());
            dst.push('\n');
        }
        egui::epaint::Shape::Vec(items) => {
            for s in items {
                append_shape_text(s, dst);
            }
        }
        _ => {}
    }
}

fn collect_certificate_panel_text(cert: &web_snapshot::VerificationCertificate) -> String {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let live_bindings = std::collections::BTreeMap::new();
    let out = ctx.run(raw, |ctx| {
        ctx.enable_accesskit();
        egui::CentralPanel::default().show(ctx, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            web_snapshot::render_certificate_metrics(ui, Some(cert), &live_bindings, &palette);
        });
    });
    let mut rendered = String::new();
    for cs in &out.shapes {
        append_shape_text(&cs.shape, &mut rendered);
    }
    rendered
}

#[test]
fn test_proof_panel_renders_kernel_verified_vs_unverified_and_expandable_theorems() {
    use oxidize_core::traceability::LeanProofBinding;

    let embedded_json = include_str!(concat!(env!("OUT_DIR"), "/traceability_manifest.json"));
    let mut cert = web_snapshot::parse_manifest_json(embedded_json)
        .expect("valid manifest JSON")
        .1
        .expect("embedded certificate");
    cert.lean4_proof_digests.clear();
    cert.lean4_proof_bindings.clear();
    cert.lean4_proof_digests
        .insert("IntervalSoundness.lean".to_string(), "a".repeat(64));
    cert.lean4_proof_digests
        .insert("UnprovenDraft.lean".to_string(), "b".repeat(64));
    cert.lean4_proof_bindings.insert(
        "IntervalSoundness.lean".to_string(),
        LeanProofBinding {
            kernel_verified: true,
            rust_modules: vec!["math_commons".to_string()],
            theorems: vec![
                "add_outward_sound".to_string(),
                "sub_outward_sound".to_string(),
            ],
        },
    );

    assert!(super::proof_panel::is_proof_kernel_verified(
        &cert,
        "IntervalSoundness.lean"
    ));
    assert!(!super::proof_panel::is_proof_kernel_verified(
        &cert,
        "UnprovenDraft.lean"
    ));

    let rendered = collect_certificate_panel_text(&cert);
    for expected in [
        "[Kernel Verified]",
        "[Unverified]",
        "math_commons",
        "Theorem Bindings (2)",
        "add_outward_sound",
        "sub_outward_sound",
    ] {
        assert!(rendered.contains(expected), "missing `{expected}`");
    }
    assert!(!rendered.contains("sorry-free"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn generate_traceability_proof_verification_screenshots() {
    use crate::app::theme::ThemeMode;

    let mut tab = TraceabilityTab::default();
    let (report, cert) = web_snapshot::embedded_traceability().expect("embedded traceability");
    tab.apply_scan_report(report);
    tab.certificate = cert;
    tab.has_initial_scan = true;
    assert!(matches!(tab.state, TraceabilityScanState::Ready));

    for (theme, out_path) in [
        (
            ThemeMode::Dark,
            "/tmp/traceability_kernel_verified_dark.png",
        ),
        (
            ThemeMode::Light,
            "/tmp/traceability_kernel_verified_light.png",
        ),
    ] {
        let mut frame = eframe::Frame::_new_kittest();
        crate::test_rasterizer::render_ui_to_png(theme, [1280, 1080], out_path, |ui| {
            tab.show_inside(ui, &mut frame);
        });
    }
}
