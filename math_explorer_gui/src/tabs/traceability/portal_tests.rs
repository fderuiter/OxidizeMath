#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

fn shape_contains(shape: &egui::epaint::Shape, needle: &str) -> bool {
    match shape {
        egui::epaint::Shape::Text(t) => t.galley.text().contains(needle),
        egui::epaint::Shape::Vec(items) => items.iter().any(|s| shape_contains(s, needle)),
        _ => false,
    }
}

fn render_tab_text_hit(tab: &mut TraceabilityTab, needle: &str) -> bool {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
    out.shapes
        .iter()
        .any(|cs| shape_contains(&cs.shape, needle))
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_packaged_desktop_outside_workspace_renders_embedded_28_papers() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let _cwd_guard = super::workspace_root_tests::CwdRestoreGuard::capture();

    std::env::set_current_dir("/").expect("set_current_dir to /");
    assert_eq!(TraceabilityTab::canonical_workspace_root(), None);

    let mut tab = TraceabilityTab::default();
    assert_eq!(tab.state, TraceabilityScanState::Idle);
    assert!(!tab.has_initial_scan);

    assert!(render_tab_text_hit(
        &mut tab,
        "Theory-to-Code Traceability Portal"
    ));
    super::wait_for_scan_completion(&mut tab);
    assert_eq!(tab.state, TraceabilityScanState::Ready);
    assert_eq!(tab.papers.len(), 28);
    assert!(tab.orphaned_papers.is_empty());
    assert!(tab.certificate.as_ref().is_some_and(|c| c.passed));
    assert!(render_tab_text_hit(&mut tab, "100% Parity Achieved"));
    assert!(render_tab_text_hit(&mut tab, "Total Papers: 28"));
    assert!(render_tab_text_hit(
        &mut tab,
        "Verification Certificate: Passed"
    ));
    assert!(tab.online_document_url("cera_framework.tex").is_some());
}

#[test]
fn test_zero_paper_live_scan_transitions_to_failed_instead_of_vacuous_parity() {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut tab = TraceabilityTab {
        report_rx: Some(rx),
        has_initial_scan: true,
        state: TraceabilityScanState::Scanning,
        ..TraceabilityTab::default()
    };

    let mut empty_report = oxidize_core::traceability::TraceabilityReport::default();
    empty_report
        .paper_coverage
        .insert("spec:transformer_v1".to_string(), vec!["ai".to_string()]);
    tx.send(Ok(empty_report)).unwrap();

    assert!(render_tab_text_hit(&mut tab, "Scan Failed"));
    assert!(tab.state.is_failed());
    assert!(!render_tab_text_hit(&mut tab, "100% Parity Achieved"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_external_clone_picker_scans_live_while_load_state_stays_sandboxed() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let _cwd_guard = super::workspace_root_tests::CwdRestoreGuard::capture();

    let pid = std::process::id();
    let clone_dir = std::env::temp_dir().join(format!("oxidize_external_clone_{pid}"));
    let _ = std::fs::remove_dir_all(&clone_dir);
    std::fs::create_dir_all(clone_dir.join("papers")).unwrap();
    std::fs::create_dir_all(clone_dir.join("crates").join("ext_math").join("src")).unwrap();
    std::fs::write(
        clone_dir.join("traceability.toml"),
        "[links]\next_math = \"external_manuscript.tex\"\n",
    )
    .unwrap();
    std::fs::write(
        clone_dir.join("papers").join("external_manuscript.tex"),
        "\\documentclass{article}\\begin{document}External\\end{document}",
    )
    .unwrap();
    std::fs::write(
        clone_dir
            .join("crates")
            .join("ext_math")
            .join("src")
            .join("lib.rs"),
        concat!(
            "// [cite",
            ":ext_math]\npub fn check() -> usize { let x = 1; assert_eq!(x, 1); x }\n"
        ),
    )
    .unwrap();

    let mut tab = TraceabilityTab::default();
    let untrusted_payload = serde_json::json!({ "repo_path": clone_dir }).to_string();
    tab.load_state(&untrusted_payload);
    assert_eq!(
        tab.repo_path, None,
        "Untrusted load_state must reject external clone outside workspace"
    );

    std::env::set_current_dir("/").expect("set_current_dir to /");
    tab.select_directory(&clone_dir);
    assert!(tab.repo_path.is_some());
    assert!(tab.state.is_scanning());

    super::wait_for_scan_completion(&mut tab);
    assert_eq!(tab.state, TraceabilityScanState::Ready);
    assert_eq!(tab.papers.len(), 1);
    assert!(tab.papers.contains_key("external_manuscript.tex"));
    assert!(tab.orphaned_papers.is_empty());
    assert!(tab.online_document_url("external_manuscript.tex").is_some());

    drop(_cwd_guard);
    let _ = std::fs::remove_dir_all(&clone_dir);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_invalid_external_folders_show_inline_error_feedback() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let pid = std::process::id();
    let invalid_base = std::env::temp_dir().join(format!("oxidize_invalid_pick_{pid}"));
    let _ = std::fs::remove_dir_all(&invalid_base);

    let empty_dir = invalid_base.join("empty");
    let no_toml_dir = invalid_base.join("only_papers");
    let no_papers_dir = invalid_base.join("only_toml");
    std::fs::create_dir_all(&empty_dir).unwrap();
    std::fs::create_dir_all(no_toml_dir.join("papers")).unwrap();
    std::fs::create_dir_all(&no_papers_dir).unwrap();
    std::fs::write(no_papers_dir.join("traceability.toml"), "[links]\n").unwrap();

    let mut tab = TraceabilityTab::default();
    for (candidate, expected_fragment) in [
        (&empty_dir, "missing both traceability.toml and papers/"),
        (&no_toml_dir, "missing traceability.toml"),
        (&no_papers_dir, "missing papers/"),
    ] {
        tab.select_directory(candidate);
        assert!(tab.state.is_failed());
        assert_eq!(tab.repo_path, None);
        assert!(tab
            .error_message()
            .is_some_and(|msg| msg.contains(expected_fragment)));
        assert!(render_tab_text_hit(&mut tab, expected_fragment));
    }

    let empty_papers_repo = invalid_base.join("zero_tex_repo");
    std::fs::create_dir_all(empty_papers_repo.join("papers")).unwrap();
    std::fs::write(empty_papers_repo.join("traceability.toml"), "[links]\n").unwrap();
    tab.select_directory(&empty_papers_repo);
    super::wait_for_scan_completion(&mut tab);
    assert!(tab.state.is_failed());

    let _ = std::fs::remove_dir_all(&invalid_base);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_ci_pages_workflow_verifies_all_28_paper_pdfs_and_sw_shell_scope() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ws_root = TraceabilityTab::canonical_workspace_root().expect("workspace root");
    let ci_yaml = std::fs::read_to_string(ws_root.join(".github").join("workflows").join("ci.yml"))
        .expect("read ci.yml");

    assert!(ci_yaml.contains("name: wasm_web_dist"));
    assert!(ci_yaml.contains("name: compiled_papers"));
    assert!(ci_yaml.contains("needs: [pre-flight, build-and-test, compile-papers, feature-matrix]"));
    assert!(ci_yaml.contains("pages_dist/${stem}.pdf"));
    assert!(ci_yaml.contains("pages_dist/papers/${stem}.pdf"));

    let stage_idx = ci_yaml
        .find("Stage Compiled PDFs and Refresh SHA-256 Manifest")
        .expect("PDF staging step in deploy-pages");
    let upload_idx = ci_yaml[stage_idx..]
        .find("actions/upload-pages-artifact@")
        .expect("upload-pages-artifact after PDF staging")
        + stage_idx;
    let deploy_idx = ci_yaml[upload_idx..]
        .find("actions/deploy-pages@")
        .expect("deploy-pages after upload-pages-artifact")
        + upload_idx;
    assert!(stage_idx < upload_idx && upload_idx < deploy_idx);

    let tex_count = std::fs::read_dir(ws_root.join("papers"))
        .expect("read papers/")
        .flatten()
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("tex"))
        .count();
    assert_eq!(tex_count, 28);

    let sw_js = std::fs::read_to_string(ws_root.join("math_explorer_gui").join("sw.js"))
        .expect("read sw.js");
    assert!(
        sw_js.contains("isShellNavigationPath") && sw_js.contains(r"/\.pdf$/i.test(relPath)"),
        "sw.js must keep staged paper PDF navigations out of the app-shell index.html cache"
    );
}
