#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use std::thread;
use std::time::Duration;

fn wait_for_morphogenesis_snapshot(tab: &mut MorphogenesisTab) {
    for _ in 0..600 {
        if tab.worker.update().is_some() {
            return;
        }
        thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn test_morphogenesis_initial_state() {
    let tab = MorphogenesisTab::default();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
    assert!(tab.dirty);
    assert!(tab.texture.is_none());
    assert!(tab.pixels.is_empty());
}

#[test]
fn test_morphogenesis_render_lifecycle_and_dirty_tracking() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();

    // 1. Initial state has dirty flag set and no texture
    assert!(tab.dirty);
    assert!(tab.texture.is_none());

    // Allow worker thread to generate the initial snapshot
    wait_for_morphogenesis_snapshot(&mut tab);

    // Run first render pass: uploads texture and clears dirty flag
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });

    assert!(!tab.dirty);
    assert!(tab.texture.is_some());
    assert_eq!(tab.pixels.len(), tab.width * tab.height);

    let initial_pixels_cap = tab.pixels.capacity();
    let initial_tex_id = tab.texture.as_ref().unwrap().id();

    // 2. Pause simulation to test stationary/idle frame
    let _ = tab.worker.send_command(SimCommand::Pause);
    tab.worker.wait_until_paused();

    // Drain any pending updates from the worker before pause settled
    while tab.worker.update().is_some() {}
    tab.dirty = false;

    // Run idle frame: does not mark dirty and reuses persistent buffer and texture
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });

    assert!(!tab.dirty);
    assert_eq!(tab.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tab.texture.as_ref().unwrap().id(), initial_tex_id);

    // 3. Manually setting dirty flag causes texture update and clears dirty
    tab.dirty = true;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert!(!tab.dirty);
    assert_eq!(tab.pixels.capacity(), initial_pixels_cap);
    assert_eq!(tab.texture.as_ref().unwrap().id(), initial_tex_id);

    // 4. Reset simulation sets dirty flag
    let _ = tab.worker.send_command(SimCommand::Reset);
    tab.dirty = true;
    assert!(tab.dirty);
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert!(!tab.dirty);
    assert_eq!(tab.texture.as_ref().unwrap().id(), initial_tex_id);
}

#[test]
fn test_morphogenesis_grid_resize_and_buffer_safety() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();

    // Initial render
    wait_for_morphogenesis_snapshot(&mut tab);
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.pixels.len(), 100 * 100);

    // Apply config with changed dimensions
    let mut config = tab.export_config();
    config.width = 50;
    config.height = 50;
    assert!(tab.apply_config(config).is_ok());
    assert!(tab.dirty);

    // Wait for resized runner to produce a snapshot
    wait_for_morphogenesis_snapshot(&mut tab);

    // Render after resize
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert!(!tab.dirty);
    assert_eq!(tab.pixels.len(), 50 * 50);
}

#[test]
fn test_plot_concentration_color_mapping() {
    let width = 2;
    let height = 2;
    let data = vec![0.0, 0.5, 1.0, 0.25];
    let image = MorphogenesisTab::plot_concentration(&data, width, height);

    assert_eq!(image.size, [width, height]);
    assert_eq!(image.pixels.len(), 4);

    let min_color = heatmap_color(0.0);
    assert_eq!(image.pixels[0], egui::Color32::from_rgb(min_color[0], min_color[1], min_color[2]));

    let max_color = heatmap_color(1.0);
    assert_eq!(image.pixels[2], egui::Color32::from_rgb(max_color[0], max_color[1], max_color[2]));

    // Uniform data (zero range) should not panic or divide by zero
    let uniform_data = vec![1.0; 4];
    let uniform_image = MorphogenesisTab::plot_concentration(&uniform_data, width, height);
    assert_eq!(uniform_image.pixels.len(), 4);
}

#[test]
fn test_morphogenesis_starter_gallery_lifecycle() {
    let mut tab = MorphogenesisTab::default();
    // Acceptance Criteria 1: Empty simulation tab initializes with starter gallery closed
    assert!(!tab.show_starter_gallery);

    // Acceptance Criteria 2: Applying starter preset transitions into active mode
    let spots_json = crate::starter_gallery::PRESET_TURING_SPOTS;
    assert!(tab.from_json_str(spots_json).is_ok());
    tab.show_starter_gallery = false;
    assert!(!tab.show_starter_gallery);

    // Acceptance Criteria 4: Toggle gallery on demand
    tab.show_starter_gallery = !tab.show_starter_gallery;
    assert!(tab.show_starter_gallery);

    // Acceptance Criteria 3: Loaded workspace session bypasses the gallery
    let saved_state = tab.save_state().unwrap();
    let mut loaded_tab = MorphogenesisTab::default();
    assert!(!loaded_tab.show_starter_gallery);
    loaded_tab.show_starter_gallery = true;
    loaded_tab.load_state(&saved_state);
    assert!(!loaded_tab.show_starter_gallery);
}

fn get_repaint_delay(output: &egui::FullOutput) -> std::time::Duration {
    output.viewport_output.get(&egui::ViewportId::ROOT).unwrap().repaint_delay
}

#[test]
fn test_morphogenesis_fault_halts_repaints_and_recovers() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab {
        show_starter_gallery: false,
        ..Default::default()
    };

    // Allow worker thread to generate initial snapshot
    wait_for_morphogenesis_snapshot(&mut tab);

    // Initial render pass uploads texture and clears dirty flag
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });

    // 1. Paused state: repaints halted
    let _ = tab.worker.send_command(SimCommand::Pause);
    tab.worker.wait_until_paused();
    while tab.worker.update().is_some() {}
    tab.dirty = false;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let output_paused = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_ne!(get_repaint_delay(&output_paused), std::time::Duration::ZERO);

    // 2. Running state: repaints requested
    let _ = tab.worker.send_command(SimCommand::Start);
    let output_running = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_eq!(tab.worker.state(), SimWorkerState::Running);
    assert_eq!(get_repaint_delay(&output_running), std::time::Duration::ZERO);

    // Pause running worker
    let _ = tab.worker.send_command(SimCommand::Pause);
    tab.worker.wait_until_paused();
    while tab.worker.update().is_some() {}

    // 3. Faulted state: repaints halted
    tab.worker.set_state_for_test(SimWorkerState::Faulted);
    assert_eq!(tab.worker.state(), SimWorkerState::Faulted);

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    let output_faulted = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
    assert_ne!(get_repaint_delay(&output_faulted), std::time::Duration::ZERO);

    // 4. Retry recovery restores communication and sets state to Ready
    tab.retry();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);
}

#[test]
fn test_morphogenesis_save_and_load_state_compatibility() {
    let mut tab = MorphogenesisTab::default();
    // Default stays Ready while hidden prior to first render
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);

    // Save state serializes SimWorkerState
    let saved = tab.save_state().expect("save_state should succeed");
    assert!(saved.contains("\"state\":\"Ready\""));
    assert!(!saved.contains("\"paused\""));

    let mut loaded = MorphogenesisTab::default();
    loaded.load_state(&saved);
    assert_eq!(loaded.worker.state(), SimWorkerState::Ready);

    // Running state roundtrips cleanly
    assert!(tab.worker.send_command(SimCommand::Start).is_ok());
    let saved_running = tab.save_state().expect("save_state should succeed");
    assert!(saved_running.contains("\"state\":\"Running\""));
    loaded.load_state(&saved_running);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);

    // Load legacy JSON with "paused": true
    let legacy_paused = r#"{"dt":0.02,"paused":true,"show_starter_gallery":false}"#;
    loaded.load_state(legacy_paused);
    assert_eq!(loaded.worker.state(), SimWorkerState::Ready);
    assert_eq!(loaded.params.read().unwrap().dt, 0.02);

    // Load legacy JSON with "paused": false
    let legacy_running = r#"{"dt":0.08,"paused":false,"show_starter_gallery":false}"#;
    loaded.load_state(legacy_running);
    assert_eq!(loaded.worker.state(), SimWorkerState::Running);
    assert_eq!(loaded.params.read().unwrap().dt, 0.08);
}

#[test]
fn test_morphogenesis_poisoned_lock_recovery() {
    let mut tab = MorphogenesisTab::default();
    let _ = tab.worker.send_command(SimCommand::Pause);
    tab.worker.wait_until_paused();

    // Poison shared parameter RwLock
    let params_clone = Arc::clone(&tab.params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = params_clone.write().unwrap();
        guard.a = 0.25;
        guard.dt = 0.04;
        panic!("simulated worker panic");
    }));
    assert!(tab.params.is_poisoned());

    // Runner creation and stepping recover parameters from poisoned lock
    let mut runner = MorphogenesisTab::create_runner(&tab.params, 20, 20);
    assert_eq!(runner.system.kinetics.a, 0.25);
    runner.step();
    assert!(FiniteStateGuard::is_finite_state(&runner));
    let snap = runner.get_snapshot();
    assert_eq!(snap.width, 20);
    assert_eq!(snap.height, 20);

    // FiniteStateGuard recovery actions survive poisoned lock
    {
        let mut p = tab.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 5.0;
        p.dt = 0.08;
    }
    tab.auto_scale_step_size();
    assert!((tab.params.read().unwrap_or_else(|e| e.into_inner()).dt - 0.04).abs() < 1e-12);

    tab.clamp_parameters();
    assert_eq!(tab.params.read().unwrap_or_else(|e| e.into_inner()).a, 1.0);

    tab.reset_parameters();
    assert_eq!(tab.params.read().unwrap_or_else(|e| e.into_inner()).a, 0.1);
    let _ = tab.worker.send_command(SimCommand::Pause);
    tab.worker.wait_until_paused();

    // Preset export/import and save/load state survive poisoned lock
    let mut cfg = tab.export_config();
    cfg.a = 0.18;
    cfg.dt = 0.03;
    assert!(tab.apply_config(cfg).is_ok());
    assert_eq!(tab.params.read().unwrap_or_else(|e| e.into_inner()).a, 0.18);

    let saved = tab.save_state().expect("save_state should succeed with poisoned lock");
    assert!(saved.contains("0.03"));
    tab.load_state(r#"{"dt":0.02,"state":"Ready","show_starter_gallery":false}"#);
    assert_eq!(tab.params.read().unwrap_or_else(|e| e.into_inner()).dt, 0.02);

    // UI rendering (including draw_parameter_sliders acquiring write lock) survives poisoned lock
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tab.show(ctx, &mut frame);
    });
}

#[test]
fn test_turing_stripes_syncs_kinetics_before_reinit_and_surfaces_import_errors() {
    let mut tab = MorphogenesisTab::default();
    let _ = tab.worker.send_command(SimCommand::Pause);
    tab.worker.wait_until_paused();

    // Verify runner starts with default Schnakenberg kinetics (a=0.1, b=0.9 => v0 = 0.9)
    let mut runner = MorphogenesisTab::create_runner(&tab.params, 20, 20);
    assert!((runner.system.kinetics.a - 0.1).abs() < 1e-12);
    assert!((runner.system.kinetics.b - 0.9).abs() < 1e-12);

    // Launch Turing Stripes preset (a=0.14, b=0.86, d_u=1.0, d_v=50.0, dt=0.02)
    assert!(tab
        .from_json_str(crate::starter_gallery::PRESET_TURING_STRIPES)
        .is_ok());
    {
        let p = tab.params.read().unwrap();
        assert!((p.a - 0.14).abs() < 1e-12);
        assert!((p.b - 0.86).abs() < 1e-12);
        assert!((p.d_u - 1.0).abs() < 1e-12);
        assert!((p.d_v - 50.0).abs() < 1e-12);
        assert!((p.dt - 0.02).abs() < 1e-12);
    }

    // SimCommand::Reset must sync SchnakenbergKinetics BEFORE initialize_system computes steady state
    runner.process_command(SimCommand::Reset);
    assert!((runner.system.kinetics.a - 0.14).abs() < 1e-12);
    assert!((runner.system.kinetics.b - 0.86).abs() < 1e-12);
    assert_eq!(runner.system.diffusion_coeffs, [1.0, 50.0]);

    let v_field = runner.system.v();
    let mean_v = v_field.iter().sum::<f64>() / v_field.len() as f64;
    // Steady state v_0 = b / (a + b)^2 = 0.86 / 1.0 = 0.86 (stale kinetics would produce ~0.90)
    assert!(
        (mean_v - 0.86).abs() < 0.02,
        "Expected reinitialized mean v near 0.86, got {}",
        mean_v
    );

    // Malformed preset JSON surfaces error in PresetDialogState
    let err = tab
        .from_json_str("{bad morphogenesis json")
        .expect_err("Malformed JSON must fail");
    tab.dialog_state.record_import_error(err);
    assert!(tab.dialog_state.toast_error.is_some());
}

#[test]
fn test_morphogenesis_defers_start_until_render_and_faults_on_divergence() {
    let ctx = egui::Context::default();
    let mut tab = MorphogenesisTab::default();
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);

    // Hidden tab remains idle before first render
    thread::sleep(Duration::from_millis(40));
    assert_eq!(tab.worker.state(), SimWorkerState::Ready);

    // First render starts stepping; default d_v=100, dt=0.05 diverges to Faulted
    let mut frame = eframe::Frame::_new_kittest();
    for _ in 0..400 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            tab.show(ctx, &mut frame);
        });
        if tab.worker.state() == SimWorkerState::Faulted {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(tab.worker.state(), SimWorkerState::Faulted);
    assert!(tab.worker.diagnostic_report().is_some());
}

