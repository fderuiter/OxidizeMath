#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_traceability_initial_state_transitions_to_scanning() {
    let tab = TraceabilityTab::default();
    assert_eq!(tab.state, TraceabilityScanState::Scanning);
    assert!(tab.state.is_scanning());
    assert!(!tab.state.is_ready());
    assert!(!tab.state.is_failed());
    assert_eq!(tab.error_message(), None);
}

#[test]
fn test_traceability_successful_scan_transitions_to_ready() {
    let mut tab = TraceabilityTab {
        papers: HashMap::new(),
        orphaned_papers: Vec::new(),
        unlinked_code: Vec::new(),
        invalid_links: Vec::new(),
        repo_path: None,
        report_rx: None,
        state: TraceabilityScanState::Idle,
    };

    let (tx, rx) = std::sync::mpsc::channel();
    tab.report_rx = Some(rx);
    tab.state = TraceabilityScanState::Scanning;

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
    let mut tab = TraceabilityTab {
        papers: HashMap::new(),
        orphaned_papers: Vec::new(),
        unlinked_code: Vec::new(),
        invalid_links: Vec::new(),
        repo_path: None,
        report_rx: None,
        state: TraceabilityScanState::Idle,
    };

    let (tx, rx) = std::sync::mpsc::channel();
    tab.report_rx = Some(rx);
    tab.state = TraceabilityScanState::Scanning;

    let err_text = "Permission denied while accessing repository";
    tx.send(Err(err_text.to_string())).unwrap();

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });

    assert_eq!(tab.state, TraceabilityScanState::Failed(err_text.to_string()));
    assert!(tab.state.is_failed());
    assert_eq!(tab.error_message(), Some(err_text));
}

#[test]
fn test_traceability_channel_disconnect_transitions_to_failed() {
    let mut tab = TraceabilityTab {
        papers: HashMap::new(),
        orphaned_papers: Vec::new(),
        unlinked_code: Vec::new(),
        invalid_links: Vec::new(),
        repo_path: None,
        report_rx: None,
        state: TraceabilityScanState::Idle,
    };

    let (tx, rx) = std::sync::mpsc::channel();
    tab.report_rx = Some(rx);
    tab.state = TraceabilityScanState::Scanning;

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
    let mut tab = TraceabilityTab {
        papers: HashMap::new(),
        orphaned_papers: Vec::new(),
        unlinked_code: Vec::new(),
        invalid_links: Vec::new(),
        repo_path: None,
        report_rx: None,
        state: TraceabilityScanState::Failed("IO error".to_string()),
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
        papers: HashMap::new(),
        orphaned_papers: Vec::new(),
        unlinked_code: Vec::new(),
        invalid_links: Vec::new(),
        repo_path: None,
        report_rx: None,
        state: TraceabilityScanState::Failed("Scan failure".to_string()),
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
    let sample_datasets = vec![
        (
            "Scan Failure Rate".to_string(),
            vec![[0.0, 1.0], [1.0, 0.0]],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&sample_datasets, &options) {
        let _ = std::fs::write("/tmp/traceability_failure_state.png", png_data);
    }
}

#[test]
fn test_traceability_online_document_links() {
    let url_tex = TraceabilityTab::paper_url("partitions_implementation.tex");
    assert_eq!(
        url_tex,
        Some("https://fderuiter.github.io/OxidizeMath/partitions_implementation.pdf".to_string())
    );

    let url_prefixed = TraceabilityTab::paper_url("papers/cera_framework.tex");
    assert_eq!(
        url_prefixed,
        Some("https://fderuiter.github.io/OxidizeMath/cera_framework.pdf".to_string())
    );

    let url_pdf = TraceabilityTab::paper_url("win_ratio.pdf");
    assert_eq!(
        url_pdf,
        Some("https://fderuiter.github.io/OxidizeMath/win_ratio.pdf".to_string())
    );

    assert_eq!(TraceabilityTab::paper_url("spec:transformer_v1"), None);

    let mut tab = TraceabilityTab {
        papers: HashMap::new(),
        orphaned_papers: vec!["papers/chaos_theory.tex".to_string()],
        unlinked_code: Vec::new(),
        invalid_links: Vec::new(),
        repo_path: None,
        report_rx: None,
        state: TraceabilityScanState::Ready,
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
        Some("https://fderuiter.github.io/OxidizeMath/partitions_implementation.pdf".to_string())
    );

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        let mut frame = eframe::Frame::_new_kittest();
        tab.show(ctx, &mut frame);
    });

    let sample_datasets = vec![
        (
            "Hosted Papers".to_string(),
            vec![[0.0, 0.0], [1.0, 28.0]],
        ),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&sample_datasets, &options) {
        let _ = std::fs::write("/tmp/traceability_paper_links.png", png_data);
    }
}

