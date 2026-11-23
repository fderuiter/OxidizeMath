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

    // Verify pixel buffer content matches worker snapshot spins
    let spins = tool
        .worker
        .latest_snapshot()
        .and_then(|s| s.structured_data.as_ref())
        .and_then(|d| d.downcast_ref::<Vec<i8>>())
        .expect("Seeded worker snapshot must contain spin vector");
    let dark_palette = crate::app::ThemeMode::Dark.palette();
    for (pixel, &s) in tool.pixels.iter().zip(spins.iter()) {
        assert_eq!(*pixel, ising_spin_color(s, &dark_palette));
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
    tool.reset();
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

#[test]
fn test_ising_poisoned_lock_recovery() {
    let mut tool = IsingModelTool::default();

    // Poison shared parameter RwLock as if a worker thread panicked while holding it
    let params_clone = Arc::clone(&tool.params);
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut guard = params_clone.write().unwrap();
        guard.temperature = 3.25;
        guard.steps_per_frame = 200;
        panic!("simulated worker panic");
    }));
    assert!(tool.params.is_poisoned());

    // Runner step, snapshot, and speed command recover inner parameters without panicking
    let mut runner = IsingRunner {
        lattice: SpinLattice::new(20, 20, None),
        params: Arc::clone(&tool.params),
    };
    runner.step();
    let snap = runner.get_snapshot();
    assert_eq!(snap.width, 20);
    assert_eq!(snap.height, 20);
    assert_eq!(snap.custom_data.len(), 2);

    runner.process_command(SimCommand::SetSpeed(350));
    assert_eq!(
        tool.params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .steps_per_frame,
        350
    );

    // Slider-style parameter writes survive poisoned lock
    tool.temperature = 1.8;
    tool.params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .temperature = tool.temperature;
    tool.j_coupling = -1.0;
    tool.params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .j_coupling = tool.j_coupling;
    tool.h_field = 0.5;
    tool.params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .h_field = tool.h_field;
    tool.steps_per_frame = 500;
    tool.params
        .write()
        .unwrap_or_else(|e| e.into_inner())
        .steps_per_frame = tool.steps_per_frame;

    // Worker restart and UI frame execution survive poisoned lock
    tool.retry();
    assert_eq!(tool.worker.state(), SimWorkerState::Ready);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    // load_state updates poisoned params cleanly
    let saved = r#"{"temperature":4.2,"j_coupling":1.5,"h_field":-0.3,"state":"Ready","steps_per_frame":750}"#;
    tool.load_state(saved);
    let p = tool.params.read().unwrap_or_else(|e| e.into_inner());
    assert_eq!(p.temperature, 4.2);
    assert_eq!(p.j_coupling, 1.5);
    assert_eq!(p.h_field, -0.3);
    assert_eq!(p.steps_per_frame, 750);
}

#[test]
fn test_ising_spin_painting_boundary_pinning_rle_and_url_roundtrip() {
    let mut tool = IsingModelTool::default();
    let initial_rev = tool.state_revision();

    tool.brush_mode = IsingBrushMode::PinUp;
    tool.apply_brush_stroke(
        crate::framework::RasterBrushStroke {
            cells: vec![(10, 10), (11, 10)],
            secondary: false,
            via_keyboard: false,
            focus_cell: (11, 10),
        },
        None,
    );
    tool.brush_mode = IsingBrushMode::PinDown;
    tool.apply_brush_stroke(
        crate::framework::RasterBrushStroke {
            cells: vec![(80, 80)],
            secondary: false,
            via_keyboard: true,
            focus_cell: (80, 80),
        },
        None,
    );
    assert_ne!(tool.state_revision(), initial_rev);

    let saved = tool.save_state().expect("Ising save_state");
    assert!(saved.contains("pinned_rle"));
    crate::url_state::URL_JSON_BOUNDS
        .validate_json(&saved)
        .expect("Ising pinned state fits URL_JSON_BOUNDS");

    let fragment = crate::url_state::encode_share_fragment(
        "solid_state",
        Some("IsingModelTool"),
        Some(&saved),
    )
    .expect("encode Ising URL fragment");
    let decoded_url = crate::url_state::parse_share_location(&fragment, "")
        .expect("decode Ising URL fragment")
        .expect("present share params");

    let mut restored = IsingModelTool::default();
    restored
        .try_load_state(decoded_url.state_json.as_deref().unwrap())
        .expect("restore Ising pinned state");
    assert_eq!(restored.pin_cells[10 * DEFAULT_ISING_SIZE + 10], 1);
    assert_eq!(restored.pin_cells[80 * DEFAULT_ISING_SIZE + 80], 2);

    // Verify runner preserves pinned sites across thermal sweeps with zero heap allocation:
    let mut runner = IsingModelTool::build_runner(&restored.params);
    runner.step();
    {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            runner.step();
        });
    }
    assert_eq!(runner.lattice.get(10, 10), 1);
    assert!(runner.lattice.is_pinned(10, 10));
    assert_eq!(runner.lattice.get(80, 80), -1);
    assert!(runner.lattice.is_pinned(80, 80));

    restored.clear_pins(None);
    assert!(restored.params.read().unwrap().pinned_rle.is_none());
}

#[test]
fn test_ising_runner_apply_brush_clamping_and_clear_defector_sea() {
    let tool = IsingModelTool::default();
    let mut runner = IsingRunner {
        lattice: SpinLattice::new(100, 100, Some(11)),
        params: Arc::clone(&tool.params),
    };

    runner.process_command(SimCommand::ClearObstacles);
    assert_eq!(runner.lattice.magnetization(), -10_000);

    // Out-of-bounds coordinates and negative radius are safely clamped to [0, 99]
    runner.process_command(SimCommand::ApplyBrush {
        cx: -50,
        cy: 250,
        r: 3,
        is_obstacle: true,
    });
    assert_eq!(runner.lattice.get(0, 99), 1);
    assert_eq!(runner.lattice.get(3, 99), 1);
    assert_eq!(runner.lattice.get(0, 96), 1);
    assert_eq!(runner.lattice.get(50, 50), -1);

    // Center circular droplet r=4 paints positive (+1) spins inside disk
    runner.process_command(SimCommand::ApplyBrush {
        cx: 50,
        cy: 50,
        r: 4,
        is_obstacle: true,
    });
    assert_eq!(runner.lattice.get(50, 50), 1);
    assert_eq!(runner.lattice.get(54, 50), 1);
    assert_eq!(runner.lattice.get(54, 54), -1);
}

#[test]
fn test_ising_paused_brush_stroke_via_on_brush_updates_snapshot_and_ode() {
    let ctx = egui::Context::default();
    let mut tool = IsingModelTool::default();
    tool.apply_game_preset(FermiGamePreset::PrisonersDilemma, &ctx);
    tool.brush_radius = 8;
    tool.brush_cooperator = true;

    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw.clone(), |ctx| tool.show(ctx));
    let canvas_rect = tool
        .last_canvas_rect
        .expect("FermiGame mode must render interactive lattice canvas");

    assert!(tool.worker.send_command(SimCommand::ClearObstacles).is_ok());
    let mut click_raw = raw.clone();
    let center = canvas_rect.center();
    click_raw.events.push(egui::Event::PointerMoved(center));
    click_raw.events.push(egui::Event::PointerButton {
        pos: center,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run(click_raw, |ctx| tool.show(ctx));

    for _ in 0..80 {
        std::thread::sleep(std::time::Duration::from_millis(5));
        let _ = ctx.run(raw.clone(), |ctx| tool.show(ctx));
        if tool.cooperator_fraction() > 0.01 && tool.cooperator_fraction() < 0.10 {
            break;
        }
    }

    assert_eq!(tool.worker.state(), SimWorkerState::Ready);
    assert!(
        tool.cooperator_fraction() > 0.01 && tool.cooperator_fraction() < 0.10,
        "Paused droplet brush must update cooperator fraction on defector sea, got {:.4}",
        tool.cooperator_fraction()
    );
    let ode_end = tool.replicator_curve.last().expect("0D ODE curve")[1];
    assert!(
        ode_end < 1e-2,
        "Prisoner's Dilemma 0D ReplicatorDynamics curve must decay to 0, got {ode_end}"
    );
}

#[test]
fn test_ising_fermi_game_presets_and_state_roundtrip() {
    let ctx = egui::Context::default();
    let mut tool = IsingModelTool::default();
    assert_eq!(tool.mode, IsingInteractionMode::Ferromagnetic);

    for preset in [
        FermiGamePreset::PrisonersDilemma,
        FermiGamePreset::StagHunt,
        FermiGamePreset::HawkDove,
    ] {
        tool.apply_game_preset(preset, &ctx);
        assert_eq!(tool.mode, IsingInteractionMode::FermiGame);
        assert_eq!(tool.game_preset, preset);
        assert_eq!(tool.game_payoffs, preset.payoffs());
        assert!(!tool.replicator_curve.is_empty());
    }

    tool.brush_radius = 12;
    tool.brush_cooperator = false;
    let saved = tool.save_state().expect("save_state");

    let mut restored = IsingModelTool::default();
    restored
        .try_load_state(&saved)
        .expect("FermiGame IsingState must restore cleanly");
    assert_eq!(restored.mode, IsingInteractionMode::FermiGame);
    assert_eq!(restored.game_preset, FermiGamePreset::HawkDove);
    assert_eq!(restored.game_payoffs, FermiGamePreset::HawkDove.payoffs());
    assert_eq!(restored.brush_radius, 12);
    assert!(!restored.brush_cooperator);
}
