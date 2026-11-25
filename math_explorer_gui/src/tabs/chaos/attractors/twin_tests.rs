use super::*;
use crate::async_sim::{SimCommand, SimWorkerState, SimulationRunner, TimelineMetadata};
use crate::framework::{InteractiveTool, PlaybackTransport};
use eframe::egui;
use std::time::Duration;

fn wait_for_plotter(
    ctx: &egui::Context,
    plotter: &mut AttractorPlotter,
    pred: impl Fn(&AttractorPlotter) -> bool,
) {
    for _ in 0..400 {
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| plotter.show_inside(ui));
        });
        if pred(plotter) {
            return;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!(
        "Timed out waiting for AttractorPlotter condition at step={} state={}",
        plotter.step_count,
        plotter.worker.state()
    );
}

#[test]
fn test_custom_initial_coordinates_evolve_in_lorenz_and_custom_modes() {
    let mut plotter = AttractorPlotter::default();
    plotter.twin_cfg.initial_state = Vector3::new(-5.0, -4.5, 21.0);
    plotter.reset();
    assert_eq!(plotter.system.state.vec, Vector3::new(-5.0, -4.5, 21.0));

    let mut default_plotter = AttractorPlotter::default();
    plotter.step_simulation();
    default_plotter.step_simulation();
    assert_ne!(plotter.history[0], default_plotter.history[0]);
    assert!((plotter.history[0] - Vector3::new(-5.0, -4.5, 21.0)).norm() < 5.0);

    // Custom 3D AST flow mode also resets and integrates from configured (x0, y0, z0)
    plotter.mode = AttractorMode::Custom;
    plotter.twin_cfg.initial_state = Vector3::new(2.5, -1.5, 0.75);
    plotter.reset();
    assert_eq!(plotter.system.state.vec, Vector3::new(2.5, -1.5, 0.75));

    let mut runner = AttractorRunner::from_params(Arc::clone(&plotter.params));
    assert_eq!(runner.system.state.vec, Vector3::new(2.5, -1.5, 0.75));
    runner.step_once();
    let expected = CompiledFlowJob::from_bank(&plotter.flow_bank)
        .rk4_step(Vector3::new(2.5, -1.5, 0.75), plotter.dt);
    assert!((runner.system.state.vec - expected).norm() < 1e-12);
}

#[test]
fn test_lorenz_twin_1e_minus_9_tracks_8_orders_over_28_48_time_units_and_lyapunov_slope() {
    let mut plotter = AttractorPlotter::default();
    plotter.twin_cfg.enabled = true;
    plotter.twin_cfg.epsilon = 1e-9;
    plotter.dt = 0.01;
    plotter.reset();

    let mut runner = AttractorRunner::from_params(Arc::clone(&plotter.params));
    for _ in 0..2848 {
        runner.step_once();
    }
    assert_eq!(runner.step_count, 2848);
    assert_eq!(runner.twin_state.divergence_history.len(), 2849);

    let [t0, log_d0] = runner.twin_state.divergence_history[0];
    let [t_end, log_d_end] = *runner.twin_state.divergence_history.back().unwrap();
    assert!((t0 - 0.0).abs() < 1e-12);
    assert!((log_d0 - (-9.0)).abs() < 1e-6, "got initial log10={log_d0}");
    assert!((t_end - 28.48).abs() < 1e-9);

    let orders_grown = log_d_end - log_d0;
    assert!(
        (7.0..=11.5).contains(&orders_grown),
        "Expected ~8-10 orders of magnitude separation growth over 28.48 time units, got {orders_grown:.2} (end={log_d_end:.2})"
    );

    let lyap = plotter
        .current_lyapunov_exponent()
        .expect("Wolf Lyapunov exponent must compute on chaotic Lorenz");
    assert!(
        (0.65..=1.25).contains(&lyap),
        "Expected chaotic Lorenz λ₁ near 0.9, got {lyap}"
    );
    let empirical_slope = (runner.twin_state.divergence_history[2200][1]
        - runner.twin_state.divergence_history[400][1])
        / (runner.twin_state.divergence_history[2200][0]
            - runner.twin_state.divergence_history[400][0]);
    let theoretical_slope = lyap / std::f64::consts::LN_10;
    assert!(
        (empirical_slope - theoretical_slope).abs() < 0.22,
        "Empirical log10 slope {empirical_slope:.3} should track Wolf λ₁/ln(10) = {theoretical_slope:.3}"
    );

    // Render UI with twin mode active and verify both 3D trajectories + divergence plot render cleanly
    plotter.history = runner.history.clone();
    plotter.system.state.vec = runner.system.state.vec;
    plotter.twin_state = runner.twin_state.clone();
    plotter.step_count = runner.step_count;
    let ctx = egui::Context::default();
    let out = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));
    let mut texts = Vec::new();
    for cs in &out.shapes {
        crate::sensitivity::tests::collect_shape_texts(std::slice::from_ref(&cs.shape), &mut texts);
    }
    assert!(texts.iter().any(|t| t.contains("Twin Separation")));
}

#[test]
fn test_custom_ast_flow_and_lorenz_twin_zero_heap_allocations() {
    let mut plotter = AttractorPlotter {
        mode: AttractorMode::Custom,
        ..Default::default()
    };
    plotter
        .flow_bank
        .set_expressions(&["-y - z", "x + a * y", "b + z * (x - c)"]);
    let job = CompiledFlowJob::from_bank(&plotter.flow_bank);

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((primary, shadow), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        let mut p = Vector3::new(1.0, 0.5, 0.25);
        let mut s = Vector3::new(1.0 + 1e-9, 0.5, 0.25);
        for _ in 0..64 {
            let (np, ns) = twin::step_twin_pair(
                AttractorMode::Custom,
                [10.0, 28.0, 8.0 / 3.0],
                0.01,
                Some(&job),
                p,
                s,
            );
            p = np;
            s = ns;
        }
        for _ in 0..32 {
            let (np, ns) = twin::step_twin_pair(
                AttractorMode::Lorenz,
                [10.0, 28.0, 8.0 / 3.0],
                0.01,
                None,
                p,
                s,
            );
            p = np;
            s = ns;
        }
        (p, s)
    });
    assert!(primary.x.is_finite() && shadow.x.is_finite());
    assert!((shadow - primary).norm() > 0.0);
}

#[test]
fn test_transport_disables_scrubber_while_running_and_enables_when_paused() {
    let ctx = egui::Context::default();
    let meta = TimelineMetadata {
        current_step: 10,
        min_step: 0,
        max_step: 20,
        stride: 1,
    };

    let mut base_id = egui::Id::NULL;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            base_id = ui.make_persistent_id("attr_scrub_check");
            let _ = PlaybackTransport::from_worker_state(
                "attr_scrub_check",
                "Attractor Plotter",
                SimWorkerState::Running,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert!(
        !ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("timeline_scrubber_info")))
            .unwrap()
            .enabled
    );
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::from_worker_state(
                "attr_scrub_check",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert!(
        ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("timeline_scrubber_info")))
            .unwrap()
            .enabled
    );
}

#[test]
fn test_twin_seek_rewinds_histories_and_forward_step_truncates_future_frames() {
    let ctx = egui::Context::default();
    let mut plotter = AttractorPlotter::default();
    plotter.twin_cfg.enabled = true;
    plotter.twin_cfg.epsilon = 1e-6;
    plotter.reset();
    plotter.execution_state = SolverExecutionState::Paused;

    plotter.worker.send_command(SimCommand::Step(16)).unwrap();
    wait_for_plotter(&ctx, &mut plotter, |p| {
        p.worker.state() == SimWorkerState::Ready && p.step_count == 16
    });
    assert_eq!(plotter.twin_state.twin_history.len(), 16);
    assert_eq!(plotter.twin_state.divergence_history.len(), 17);

    plotter
        .worker
        .send_command(SimCommand::SeekStep(6))
        .unwrap();
    wait_for_plotter(&ctx, &mut plotter, |p| {
        p.worker.state() == SimWorkerState::Ready && p.step_count == 6
    });
    assert_eq!(plotter.history.len(), 6);
    assert_eq!(plotter.twin_state.twin_history.len(), 6);
    assert_eq!(plotter.twin_state.divergence_history.len(), 7);
    assert_eq!(plotter.worker.timeline_metadata().unwrap().max_step, 16);

    plotter.worker.send_command(SimCommand::Step(1)).unwrap();
    wait_for_plotter(&ctx, &mut plotter, |p| {
        p.worker.state() == SimWorkerState::Ready && p.step_count == 7
    });
    assert_eq!(plotter.worker.timeline_metadata().unwrap().max_step, 7);
    assert_eq!(plotter.twin_state.twin_history.len(), 7);
    assert_eq!(plotter.twin_state.divergence_history.len(), 8);

    let mut direct_7 = AttractorRunner::from_params(Arc::clone(&plotter.params));
    for _ in 0..7 {
        direct_7.step_once();
    }
    assert_eq!(
        plotter.twin_state.shadow_vec.x.to_bits(),
        direct_7.twin_state.shadow_vec.x.to_bits()
    );
}

#[test]
fn test_twin_preset_url_roundtrip_audio_coexistence_and_legacy_backward_compatibility() {
    let mut plotter = AttractorPlotter::default();
    // Default state omits twin and audio keys for compact URL payloads
    let default_json = plotter.save_state().unwrap();
    assert!(!default_json.contains("twin_enabled"));
    assert!(!default_json.contains("initial_state"));
    assert!(!default_json.contains("epsilon"));
    assert!(!default_json.contains("audio"));

    // StarterGallery butterfly twin preset loads and round-trips through #tab=chaos&tool=AttractorPlotter
    plotter
        .import_preset_json(crate::starter_gallery::PRESET_3D_ATTRACTOR_TWIN)
        .expect("Butterfly twin starter preset must import");
    assert!(plotter.twin_cfg.enabled);
    assert!((plotter.twin_cfg.epsilon - 1e-9).abs() < 1e-21);

    plotter.twin_cfg.initial_state = Vector3::new(-2.5, 3.0, 18.5);
    plotter.twin_cfg.epsilon = 1e-5;
    plotter.audio_settings.gain = 0.65;
    plotter.audio_settings.pitch_scale = 1.4;
    plotter.reset();
    let preset_json = plotter.export_preset_json().unwrap();
    assert!(preset_json.len() < crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    let fragment = crate::url_state::encode_share_fragment(
        "chaos",
        Some("AttractorPlotter"),
        Some(&preset_json),
    )
    .unwrap();
    assert!(fragment.starts_with("#tab=chaos&tool=AttractorPlotter&state="));
    let parsed = crate::url_state::parse_share_location(&fragment, "")
        .unwrap()
        .unwrap();
    let mut restored = AttractorPlotter::default();
    restored
        .import_preset_json(parsed.state_json.as_deref().unwrap())
        .unwrap();
    assert_eq!(
        restored.twin_cfg.initial_state,
        Vector3::new(-2.5, 3.0, 18.5)
    );
    assert!(restored.twin_cfg.enabled);
    assert!((restored.twin_cfg.epsilon - 1e-5).abs() < 1e-18);
    assert!((restored.audio_settings.gain - 0.65).abs() < 1e-6);
    assert!((restored.audio_settings.pitch_scale - 1.4).abs() < 1e-6);

    // Legacy preset without twin fields resets back to clean defaults
    restored
        .import_preset_json(crate::starter_gallery::PRESET_3D_ATTRACTOR)
        .unwrap();
    assert!(!restored.twin_cfg.enabled);
    assert_eq!(restored.twin_cfg.initial_state, Vector3::new(1.0, 1.0, 1.0));

    // Out-of-bounds epsilon or initial_state is rejected
    let bad_eps =
        r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"epsilon":10.0}"#;
    assert!(restored.try_load_state(bad_eps).is_err());
    let bad_coord = r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"initial_state":[999.0,1.0,1.0]}"#;
    assert!(restored.try_load_state(bad_coord).is_err());
}

#[test]
fn test_twin_export_url_state_omits_histories_and_starter_card_resolves_maturity_tier() {
    let mut plotter = AttractorPlotter::default();
    plotter.twin_cfg.enabled = true;
    plotter.twin_cfg.initial_state = Vector3::new(-3.0, 2.5, 19.0);
    plotter.twin_cfg.epsilon = 1e-7;
    plotter.reset();
    for _ in 0..150 {
        plotter.step_simulation();
    }
    let expected_steps = 150 * plotter.simulation_speed;
    assert_eq!(plotter.history.len(), expected_steps);
    assert_eq!(plotter.twin_state.twin_history.len(), expected_steps);
    assert_eq!(
        plotter.twin_state.divergence_history.len(),
        expected_steps + 1
    );

    let url_json = plotter
        .export_url_state()
        .expect("AttractorPlotter::export_url_state must return compact JSON");
    assert!(
        url_json.len() < 1200,
        "export_url_state must stay compact (< 1200 B), got {} B",
        url_json.len()
    );
    assert!(!url_json.contains("twin_history"));
    assert!(!url_json.contains("divergence_history"));

    let mut hydrated = AttractorPlotter::default();
    hydrated.try_load_recovery_state(&url_json).unwrap();
    assert!(hydrated.twin_cfg.enabled);
    assert_eq!(
        hydrated.twin_cfg.initial_state,
        Vector3::new(-3.0, 2.5, 19.0)
    );
    assert!((hydrated.twin_cfg.epsilon - 1e-7).abs() < 1e-19);

    let card = inventory::iter::<crate::starter_gallery::StarterPresetMetadata>
        .into_iter()
        .find(|p| p.id == "3d_attractor_twin")
        .expect("3d_attractor_twin StarterPresetMetadata must be registered");
    let twin_tier = crate::framework::maturity_badge::resolve_tool_maturity_tier(
        card.tool_name.unwrap_or(card.id),
        card.domain,
        None,
    );
    let base_tier = crate::framework::maturity_badge::resolve_tool_maturity_tier(
        "AttractorPlotter",
        "chaos",
        None,
    );
    assert_eq!(twin_tier, base_tier);
}

#[test]
fn test_twin_viewport_suspension_and_step_sliced_preserve_histories() {
    let mut plotter = AttractorPlotter::default();
    plotter.twin_cfg.enabled = true;
    plotter.twin_cfg.epsilon = 1e-8;
    plotter.simulation_speed = 8;
    plotter.reset();

    plotter
        .worker
        .controller_mut()
        .enable_snapshot_history(128, 1_048_576);
    plotter.set_suspended(true);
    assert!(plotter.worker.is_suspended());

    // Fork descriptor respawn preserves viewport suspension and custom SnapshotHistory limits
    plotter
        .apply_fork_descriptor(AttractorForkDescriptor {
            fork_step: 10,
            post_fork_steps: 5,
            baseline_sigma: 10.0,
            baseline_rho: 28.0,
            baseline_beta: 8.0 / 3.0,
            delta_x: 1e-5,
        })
        .expect("fork descriptor must replay cleanly while suspended");
    assert!(plotter.worker.is_suspended());
    assert_eq!(
        plotter
            .worker
            .controller()
            .snapshot_history()
            .map(crate::async_sim::SnapshotHistory::limits),
        Some((128, 1_048_576))
    );
    plotter.set_suspended(false);
    assert!(!plotter.worker.is_suspended());

    // Zero-budget cooperative WASM slice advances at least 1 lockstep twin tick
    let mut runner = AttractorRunner::from_params(Arc::clone(&plotter.params));
    runner.step_sliced(Some(Duration::ZERO));
    assert_eq!(runner.step_count, 1);
    assert_eq!(runner.history.len(), 1);
    assert_eq!(runner.twin_state.twin_history.len(), 1);
    assert_eq!(runner.twin_state.divergence_history.len(), 2);
}
