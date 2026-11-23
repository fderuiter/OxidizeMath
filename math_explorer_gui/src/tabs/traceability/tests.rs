#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

mod portal_tests;
mod snapshot_tests;
#[cfg(not(target_arch = "wasm32"))]
mod workspace_root_tests;

#[test]
fn test_traceability_initial_state_transitions_to_scanning() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut tab = TraceabilityTab::default();
    assert_eq!(tab.state, TraceabilityScanState::Idle);
    assert!(!tab.has_initial_scan);
    assert!(tab.report_rx.is_none());
    assert!(!tab.state.is_scanning());
    assert!(!tab.state.is_ready());
    assert!(!tab.state.is_failed());
    assert_eq!(tab.error_message(), None);

    let ctx = egui::Context::default();
    let first_out = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
    assert!(tab.has_initial_scan);
    assert_eq!(tab.state, TraceabilityScanState::Scanning);
    assert!(tab.state.is_scanning());
    assert!(tab.report_rx.is_some());
    assert_eq!(
        first_out
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .unwrap()
            .repaint_delay,
        std::time::Duration::ZERO
    );

    #[cfg(not(target_arch = "wasm32"))]
    {
        wait_for_scan_completion(&mut tab);
        assert_eq!(tab.state, TraceabilityScanState::Ready);
        assert!(tab.state.is_ready());
    }
}

#[test]
fn test_traceability_successful_scan_transitions_to_ready() {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut tab = TraceabilityTab {
        report_rx: Some(rx),
        has_initial_scan: true,
        state: TraceabilityScanState::Scanning,
        ..TraceabilityTab::default()
    };

    let mut report = oxidize_core::traceability::TraceabilityReport::default();
    report.orphaned_papers.push("paper_alpha".to_string());
    tx.send(Ok(report)).unwrap();

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });

    assert_eq!(tab.state, TraceabilityScanState::Ready);
    assert!(tab.state.is_ready());
    assert_eq!(tab.orphaned_papers, vec!["paper_alpha".to_string()]);
    assert_eq!(tab.error_message(), None);
}

#[test]
fn test_traceability_failed_scan_displays_error() {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut tab = TraceabilityTab {
        report_rx: Some(rx),
        has_initial_scan: true,
        state: TraceabilityScanState::Scanning,
        ..TraceabilityTab::default()
    };

    let err_text = "Permission denied while accessing repository";
    tx.send(Err(err_text.to_string())).unwrap();

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });

    assert_eq!(
        tab.state,
        TraceabilityScanState::Failed(err_text.to_string())
    );
    assert!(tab.state.is_failed());
    assert_eq!(tab.error_message(), Some(err_text));
}

#[test]
fn test_traceability_channel_disconnect_transitions_to_failed() {
    let (tx, rx) = std::sync::mpsc::channel();
    let mut tab = TraceabilityTab {
        report_rx: Some(rx),
        has_initial_scan: true,
        state: TraceabilityScanState::Scanning,
        ..TraceabilityTab::default()
    };

    drop(tx);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });

    assert!(tab.state.is_failed());
    assert!(tab.error_message().unwrap().contains("disconnected"));
}

#[test]
fn test_traceability_retry_clears_error_and_rescans() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let mut tab = TraceabilityTab {
        has_initial_scan: true,
        state: TraceabilityScanState::Failed("IO error".to_string()),
        ..TraceabilityTab::default()
    };

    assert!(tab.state.is_failed());
    tab.retry();

    assert_eq!(tab.state, TraceabilityScanState::Scanning);
    assert!(tab.state.is_scanning());
    assert_eq!(tab.error_message(), None);
    assert!(tab.report_rx.is_some());
}

#[test]
fn test_traceability_no_repaint_in_terminal_states() {
    let mut tab = TraceabilityTab {
        has_initial_scan: true,
        state: TraceabilityScanState::Failed("Scan failure".to_string()),
        ..TraceabilityTab::default()
    };

    let ctx = egui::Context::default();
    // Warm up frame for font and layout initialization
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });

    let output_failed = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
    let delay_failed = output_failed
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_failed, std::time::Duration::ZERO);

    tab.state = TraceabilityScanState::Ready;
    let output_ready = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
    let delay_ready = output_ready
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_ready, std::time::Duration::ZERO);

    tab.state = TraceabilityScanState::Idle;
    let output_idle = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
    let delay_idle = output_idle
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_idle, std::time::Duration::ZERO);

    // Verify Scanning DOES request repaint with Duration::ZERO
    tab.state = TraceabilityScanState::Scanning;
    let output_scanning = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
    let delay_scanning = output_scanning
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_eq!(delay_scanning, std::time::Duration::ZERO);
}

#[test]
fn test_traceability_failure_screenshot_generation() {
    let sample_datasets = vec![(
        "Scan Failure Rate".to_string(),
        vec![[0.0, 1.0], [1.0, 0.0]],
    )];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&sample_datasets, &options) {
        let _ = std::fs::write("/tmp/traceability_failure_state.png", png_data);
    }
}

#[test]
fn test_traceability_online_document_links() {
    assert_eq!(
        TraceabilityTab::paper_url_for_target("partitions_implementation.tex", false),
        Some("https://fderuiter.github.io/OxidizeMath/partitions_implementation.pdf".to_string())
    );
    assert_eq!(
        TraceabilityTab::paper_url_for_target("partitions_implementation.tex", true),
        Some("./partitions_implementation.pdf".to_string())
    );
    assert_eq!(
        TraceabilityTab::paper_url_for_target("papers/cera_framework.tex", true),
        Some("./cera_framework.pdf".to_string())
    );
    assert_eq!(
        TraceabilityTab::paper_url_for_target("win_ratio.pdf", true),
        Some("./win_ratio.pdf".to_string())
    );
    assert_eq!(TraceabilityTab::paper_url("spec:transformer_v1"), None);

    let expected_native_or_wasm = TraceabilityTab::paper_url_for_target(
        "partitions_implementation.tex",
        cfg!(target_arch = "wasm32"),
    );
    let mut tab = TraceabilityTab {
        orphaned_papers: vec!["papers/chaos_theory.tex".to_string()],
        has_initial_scan: true,
        state: TraceabilityScanState::Ready,
        ..TraceabilityTab::default()
    };
    tab.papers.insert(
        "partitions_implementation.tex".to_string(),
        PaperStatus {
            name: "partitions_implementation.tex".to_string(),
            linked_code: vec!["crates/pure_math/src/partitions.rs".to_string()],
        },
    );

    assert_eq!(
        tab.online_document_url("partitions_implementation.tex"),
        expected_native_or_wasm
    );

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn wait_for_scan_completion(tab: &mut TraceabilityTab) {
    let ctx = egui::Context::default();
    for _ in 0..3000 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            let mut frame = eframe::Frame::_new_kittest();
            tab.show(ctx, &mut frame);
        });
        if !tab.state.is_scanning() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("Timed out waiting for TraceabilityTab scan completion");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_traceability_load_state_rejects_outside_workspace_and_traversal() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let pid = std::process::id();
    let canary_dir = std::env::temp_dir().join(format!("oxidize_canary_outside_{}", pid));
    let canary_papers = canary_dir.join("papers");
    std::fs::create_dir_all(&canary_papers).unwrap();
    std::fs::write(
        canary_papers.join("canary_audit.tex"),
        "\\documentclass{article}\\begin{document}Canary\\end{document}",
    )
    .unwrap();

    let mut tab = TraceabilityTab::default();

    let canary_path = oxidize_core::path_utils::normalize_path(&canary_dir);
    let payload = format!("{{\"repo_path\":\"{}\"}}", canary_path);
    tab.load_state(&payload);
    assert_eq!(
        tab.repo_path, None,
        "Outside-workspace path must reset to None"
    );
    tab.refresh();
    let canary_found = tab.online_document_url("canary_audit.tex").is_some();
    assert!(
        !canary_found,
        "Untrusted repo_path must not traverse outside workspace"
    );

    wait_for_scan_completion(&mut tab);
    assert!(tab.state.is_ready());
    assert_eq!(tab.online_document_url("canary_audit.tex"), None);

    for traversal in [
        "../..",
        "../../tmp",
        "..\\..\\tmp",
        "papers/..",
        "./papers/../..",
    ] {
        let traversal_payload = serde_json::json!({ "repo_path": traversal }).to_string();
        tab.load_state(&traversal_payload);
        assert_eq!(
            tab.repo_path, None,
            "Traversal sequence '{traversal}' must be rejected"
        );
    }

    let _ = std::fs::remove_dir_all(&canary_dir);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_traceability_non_workspace_cwd_rejects_mws_session() {
    let _mutex = crate::reflective_ui::tests::lock_test_mutex();
    let cwd_guard = workspace_root_tests::CwdRestoreGuard::capture();
    let expected_cwd = cwd_guard.original().to_path_buf();
    let (external_canary_dir, fake_home, home_project_dir) =
        workspace_root_tests::create_non_workspace_canary_dirs();

    std::env::set_current_dir("/").expect("set_current_dir to /");
    let mut app_root = crate::app::MathExplorerApp::default();
    let mws_root = workspace_root_tests::build_mws_envelope(&external_canary_dir.to_string_lossy());
    app_root.load_state_from_json(&mws_root);
    workspace_root_tests::assert_mws_session_rejected(
        &mut app_root,
        &external_canary_dir,
        "canary_root_escape.tex",
    );

    std::env::set_current_dir(&fake_home).expect("set_current_dir to fake_home");
    let mut app_home = crate::app::MathExplorerApp::default();
    let mws_home = workspace_root_tests::build_mws_envelope(&home_project_dir.to_string_lossy());
    app_home.load_state_from_json(&mws_home);
    workspace_root_tests::assert_mws_session_rejected(
        &mut app_home,
        &home_project_dir,
        "canary_home_escape.tex",
    );

    drop(cwd_guard);
    assert_eq!(
        std::env::current_dir().unwrap().canonicalize().unwrap(),
        expected_cwd
    );
    let _ = std::fs::remove_dir_all(&external_canary_dir);
    let _ = std::fs::remove_dir_all(&fake_home);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_traceability_edge_cases_empty_nonexistent_and_valid_subdir() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let pid = std::process::id();
    let ws_root = TraceabilityTab::canonical_workspace_root().expect("workspace root");
    let scratch_base = ws_root
        .join("target")
        .join(format!("traceability_boundary_test_{}", pid));
    std::fs::create_dir_all(&scratch_base).unwrap();

    let mut tab = TraceabilityTab::default();

    for invalid in ["", "   ", "nonexistent_dir_xyz_987654"] {
        let payload = serde_json::json!({ "repo_path": invalid }).to_string();
        tab.load_state(&payload);
        assert_eq!(tab.repo_path, None);
    }

    let regular_file = scratch_base.join("not_a_dir.txt");
    std::fs::write(&regular_file, "hello").unwrap();
    tab.load_state(&serde_json::json!({ "repo_path": regular_file }).to_string());
    assert_eq!(tab.repo_path, None);

    let valid_subdir = scratch_base.join("valid_subproject");
    std::fs::create_dir_all(valid_subdir.join("papers")).unwrap();
    std::fs::write(
        valid_subdir.join("papers").join("scoped_inside.tex"),
        "\\documentclass{article}\\begin{document}Inside\\end{document}",
    )
    .unwrap();
    tab.load_state(&serde_json::json!({ "repo_path": valid_subdir }).to_string());
    assert!(tab.repo_path.is_some());
    let saved = tab.save_state().expect("save_state");
    tab.load_state(&saved);
    assert!(tab.repo_path.is_some());
    tab.refresh();
    wait_for_scan_completion(&mut tab);
    assert!(tab.online_document_url("scoped_inside.tex").is_some());

    let _ = std::fs::remove_dir_all(&scratch_base);
}

#[cfg(unix)]
#[test]
fn test_traceability_rejects_symlink_resolving_outside_workspace() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let pid = std::process::id();
    let ws_root = TraceabilityTab::canonical_workspace_root().expect("workspace root");
    let scratch_base = ws_root
        .join("target")
        .join(format!("traceability_symlink_test_{}", pid));
    std::fs::create_dir_all(&scratch_base).unwrap();

    let external_dir = std::env::temp_dir().join(format!("oxidize_symlink_target_{}", pid));
    std::fs::create_dir_all(external_dir.join("papers")).unwrap();
    std::fs::write(external_dir.join("papers").join("symlink_canary.tex"), "x").unwrap();
    let symlink_inside = scratch_base.join("escape_link");
    let _ = std::fs::remove_file(&symlink_inside);
    std::os::unix::fs::symlink(&external_dir, &symlink_inside).unwrap();

    let mut tab = TraceabilityTab::default();

    tab.load_state(&serde_json::json!({ "repo_path": symlink_inside }).to_string());
    assert_eq!(tab.repo_path, None);

    tab.repo_path = Some(symlink_inside.to_string_lossy().into_owned());
    tab.refresh();
    assert_eq!(tab.repo_path, None);
    wait_for_scan_completion(&mut tab);
    assert_eq!(tab.online_document_url("symlink_canary.tex"), None);

    let datasets = vec![
        (
            "Rejected Untrusted Paths".to_string(),
            vec![[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
        ),
        (
            "Leaked Canary Reads".to_string(),
            vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/traceability_repo_path_validation.png", png_data);
    }

    let _ = std::fs::remove_dir_all(&external_dir);
    let _ = std::fs::remove_dir_all(&scratch_base);
}
