#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_mri_tab_initial_state() {
    let tab = MriTab::default();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
    assert_eq!(tab.time_scale, 1.0);
    assert_eq!(tab.current_time, 0.0);
}

#[test]
fn test_mri_worker_simulation_step() {
    let mut tab = MriTab::default();
    let _ = tab.worker.send_command(SimCommand::Start);
    let mut stepped = false;
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(20));
        if let Some(snap) = tab.worker.update() {
            if snap.custom_data.first().copied().unwrap_or(0.0) > 0.0 {
                stepped = true;
                break;
            }
        }
    }
    assert!(stepped);
}

#[test]
fn test_mri_fault_halts_repaints_and_recovers() {
    let ctx = egui::Context::default();
    let mut tab = MriTab::default();
    let mut frame = eframe::Frame::_new_kittest();

    // Ready state: repaints halted
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let output_ready = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
    let delay_ready = output_ready
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_ready, std::time::Duration::ZERO);

    // Start running: repaints requested
    let _ = tab.worker.send_command(SimCommand::Start);
    let output_running = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.worker.state(), SimWorkerState::Running);
    let delay_running = output_running
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_eq!(delay_running, std::time::Duration::ZERO);

    // Pause running worker
    let _ = tab.worker.send_command(SimCommand::Pause);
    while tab.worker.update().is_some() {}

    // Simulate fault
    tab.worker.set_state_for_test(SimWorkerState::Faulted);
    assert_eq!(tab.worker.state(), SimWorkerState::Faulted);

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let output_faulted = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let delay_faulted = output_faulted
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_faulted, std::time::Duration::ZERO);

    // Retry restores Ready
    tab.retry();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_mri_save_and_load_state_compatibility() {
    let mut tab = MriTab::default();
    let _ = tab.worker.send_command(SimCommand::Start);

    let saved = tab.save_state().expect("save_state should succeed");
    assert!(saved.contains("\"state\":\"Running\""));
    assert!(!saved.contains("\"paused\""));

    let mut loaded = MriTab::default();
    loaded.load_state(&saved);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);

    // Legacy paused
    let legacy_paused = r#"{"paused":true,"time_scale":2.5}"#;
    loaded.load_state(legacy_paused);
    assert_eq!(loaded.worker.state(), SimWorkerState::Ready);
    assert_eq!(loaded.time_scale, 2.5);

    // Legacy running
    let legacy_running = r#"{"paused":false,"time_scale":3.0}"#;
    loaded.load_state(legacy_running);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);
    assert_eq!(loaded.time_scale, 3.0);
}

#[test]
fn test_mri_failure_state_screenshot() {
    let mut tab = MriTab::default();
    tab.worker.set_state_for_test(SimWorkerState::Faulted);
    let datasets = vec![
        ("Mx (Transverse)".to_string(), vec![[0.0, 1.0], [1.0, 0.0]]),
        ("Worker Fault Recovery".to_string(), vec![[0.0, 0.0], [1.0, 1.0]]),
    ];
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/simulation_worker_fault_recovery.png", png_data);
    }
}
