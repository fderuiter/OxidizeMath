#![cfg_attr(any(), verified(opt_out = "tests"))]

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
    plotter.twin.initial_state = Vector3::new(-5.0, -4.5, 21.0);
    plotter.reset();
    assert_eq!(plotter.system.state.vec, Vector3::new(-5.0, -4.5, 21.0));

    let mut default_plotter = AttractorPlotter::default();
    plotter.step_simulation();
    default_plotter.step_simulation();
    assert_ne!(plotter.history[0], default_plotter.history[0]);
    assert!((plotter.history[0] - Vector3::new(-5.0, -4.5, 21.0)).norm() < 5.0);

    // Custom 3D AST flow mode also resets and integrates from configured (x0, y0, z0)
    plotter.mode = AttractorMode::Custom;
    plotter.twin.initial_state = Vector3::new(2.5, -1.5, 0.75);
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
    plotter.twin.enabled = true;
    plotter.twin.epsilon = 1e-9;
    plotter.dt = 0.01;
    plotter.reset();

    let mut runner = AttractorRunner::from_params(Arc::clone(&plotter.params));
    for _ in 0..2848 {
        runner.step_once();
    }
    assert_eq!(runner.step_count, 2848);
    assert_eq!(runner.twin.divergence_history.len(), 2849);

    let [t0, log_d0] = runner.twin.divergence_history[0];
    let [t_end, log_d_end] = *runner.twin.divergence_history.back().unwrap();
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
    let empirical_slope = (runner.twin.divergence_history[2200][1]
        - runner.twin.divergence_history[400][1])
        / (runner.twin.divergence_history[2200][0] - runner.twin.divergence_history[400][0]);
    let theoretical_slope = lyap / std::f64::consts::LN_10;
    assert!(
        (empirical_slope - theoretical_slope).abs() < 0.22,
        "Empirical log10 slope {empirical_slope:.3} should track Wolf λ₁/ln(10) = {theoretical_slope:.3}"
    );

    // Render UI with twin mode active and verify both 3D trajectories + divergence plot render cleanly
    plotter.history = runner.history.clone();
    plotter.system.state.vec = runner.system.state.vec;
    plotter.twin_state = runner.twin.clone();
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
    plotter.twin.enabled = true;
    plotter.twin.epsilon = 1e-6;
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
        direct_7.twin.shadow_vec.x.to_bits()
    );
}

#[test]
fn test_twin_preset_url_roundtrip_and_legacy_backward_compatibility() {
    let mut plotter = AttractorPlotter::default();
    // Default state omits twin keys for compact URL payloads
    let default_json = plotter.save_state().unwrap();
    assert!(!default_json.contains("twin_enabled"));
    assert!(!default_json.contains("initial_state"));
    assert!(!default_json.contains("epsilon"));

    // StarterGallery butterfly twin preset loads and round-trips through #tab=chaos&tool=AttractorPlotter
    plotter
        .import_preset_json(crate::starter_gallery::PRESET_3D_ATTRACTOR_TWIN)
        .expect("Butterfly twin starter preset must import");
    assert!(plotter.twin.enabled);
    assert!((plotter.twin.epsilon - 1e-9).abs() < 1e-21);

    plotter.twin.initial_state = Vector3::new(-2.5, 3.0, 18.5);
    plotter.twin.epsilon = 1e-5;
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
    assert_eq!(restored.twin.initial_state, Vector3::new(-2.5, 3.0, 18.5));
    assert!(restored.twin.enabled);
    assert!((restored.twin.epsilon - 1e-5).abs() < 1e-18);

    // Legacy preset without twin fields resets back to clean defaults
    restored
        .import_preset_json(crate::starter_gallery::PRESET_3D_ATTRACTOR)
        .unwrap();
    assert!(!restored.twin.enabled);
    assert_eq!(restored.twin.initial_state, Vector3::new(1.0, 1.0, 1.0));

    // Out-of-bounds epsilon or initial_state is rejected
    let bad_eps =
        r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"epsilon":10.0}"#;
    assert!(restored.try_load_state(bad_eps).is_err());
    let bad_coord = r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"initial_state":[999.0,1.0,1.0]}"#;
    assert!(restored.try_load_state(bad_coord).is_err());
}
