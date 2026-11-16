#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_ising_model_initial_state() {
    let tool = IsingModelTool::default();
    assert!(tool.dirty);
    assert!(tool.texture.is_none());
    assert!(tool.pixels.is_empty());
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_ising_model_render_lifecycle_and_dirty_tracking() {
    let ctx = egui::Context::default();
    let mut tool = IsingModelTool::default();

    // 1. Initial render pass initializes texture and pixel buffer, then clears dirty flag.
    assert!(tool.dirty);
    assert!(tool.texture.is_none());

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    assert!(!tool.dirty);
    assert!(tool.texture.is_some());
    assert_eq!(tool.pixels.len(), 100 * 100);

    // Verify pixel buffer content matches lattice spins
    for (pixel, &s) in tool.pixels.iter().zip(tool.lattice.spins().iter()) {
        let expected = if s > 0 {
            egui::Color32::from_rgb(200, 50, 50)
        } else {
            egui::Color32::from_rgb(50, 50, 200)
        };
        assert_eq!(*pixel, expected);
    }

    let initial_pixels_cap = tool.pixels.capacity();
    let initial_tex_id = tool.texture.as_ref().unwrap().id();

    // 2. Idle frame does not mark dirty and reuses persistent pixel buffer and texture.
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    assert!(!tool.dirty);
    assert_eq!(tool.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tool.texture.as_ref().unwrap().id(), initial_tex_id);

    // 3. Simulating running state sets dirty during evolution and clears it upon texture upload.
    let _ = tool.worker.send_command(SimCommand::Start);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    assert!(!tool.dirty);
    assert_eq!(tool.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tool.texture.as_ref().unwrap().id(), initial_tex_id);

    // 4. Manually setting dirty flag causes texture update and re-clearing.
    let _ = tool.worker.send_command(SimCommand::Pause);
    tool.dirty = true;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
    assert!(!tool.dirty);

    // 5. Lattice reset marks dirty and retains existing texture handle for reuse.
    tool.lattice = SpinLattice::new(100, 100, None);
    tool.dirty = true;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
    assert!(!tool.dirty);
    assert_eq!(tool.texture.as_ref().unwrap().id(), initial_tex_id);
}

#[test]
fn test_ising_fault_halts_repaints_and_recovers() {
    let ctx = egui::Context::default();
    let mut tool = IsingModelTool::default();

    // Initial pass creates texture and clears dirty flag
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    // In ready (idle) state, no immediate repaint requested
    let output = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    let delay_ready = output
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_ready, std::time::Duration::ZERO);

    // Start running: repaints should be requested
    let _ = tool.worker.send_command(SimCommand::Start);
    let output_running = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
    assert_eq!(tool.worker.state(), SimWorkerState::Running);
    let delay_running = output_running
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_eq!(delay_running, std::time::Duration::ZERO);

    // Simulate worker fault
    tool.worker.set_state_for_test(SimWorkerState::Faulted);
    assert_eq!(tool.worker.state(), SimWorkerState::Faulted);

    // In faulted state, repaint requests must be halted
    let ctx_faulted = egui::Context::default();
    let _ = ctx_faulted.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
    let output_faulted = ctx_faulted.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
    let delay_faulted = output_faulted
        .viewport_output
        .get(&egui::ViewportId::ROOT)
        .unwrap()
        .repaint_delay;
    assert_ne!(delay_faulted, std::time::Duration::ZERO);

    // Retry recovery restores communication and sets state to Ready
    tool.retry();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_ising_save_and_load_state_compatibility() {
    let mut tool = IsingModelTool {
        temperature: 3.5,
        ..Default::default()
    };
    let _ = tool.worker.send_command(SimCommand::Start);

    // Save state serializes SimWorkerState
    let saved = tool.save_state().expect("save_state should succeed");
    assert!(saved.contains("\"state\":\"Running\""));
    assert!(!saved.contains("\"running\""));

    let mut loaded = IsingModelTool::default();
    loaded.load_state(&saved);
    assert_eq!(loaded.temperature, 3.5);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);

    // Load legacy JSON with "running": false
    let legacy_paused = r#"{"temperature":1.5,"j_coupling":0.5,"h_field":0.1,"running":false,"steps_per_frame":2000}"#;
    loaded.load_state(legacy_paused);
    assert_eq!(loaded.temperature, 1.5);
    assert_eq!(loaded.worker.state(), SimWorkerState::Ready);

    // Load legacy JSON with "running": true
    let legacy_running = r#"{"temperature":2.0,"j_coupling":1.0,"h_field":0.0,"running":true,"steps_per_frame":1000}"#;
    loaded.load_state(legacy_running);
    assert_eq!(loaded.temperature, 2.0);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);
}
