#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::{InteractiveTool, SimulationFramework};
use eframe::egui;
use std::sync::{Arc, RwLock};

#[test]
fn test_lorenz_on_attractor_twin_divergence_crosses_threshold_and_flips_wings() {
    let params = Arc::new(RwLock::new(AttractorSharedParams {
        sigma: 10.0,
        rho: 28.0,
        beta: 8.0 / 3.0,
        dt: 0.01,
        simulation_speed: 1,
        max_points: 4096,
        system_kind: AttractorSystemKind::Lorenz,
        initial_state: ON_ATTRACTOR_LORENZ_STATE,
        shadow_enabled: true,
        perturbation_offset: [1e-9, 0.0, 0.0],
        extra: ExtraChaosParams::default(),
    }));
    let mut runner = AttractorRunner::from_params(params);
    let mut crossed_at: Option<f64> = None;
    let mut lobe_flip_at: Option<f64> = None;

    for _ in 1..=2600 {
        runner.step_once();
        let frame = runner.timeline.current_copied().unwrap();
        if crossed_at.is_none() && frame.separation >= 0.1 {
            crossed_at = Some(frame.time);
        }
        if lobe_flip_at.is_none()
            && frame.primary[0] * frame.shadow[0] < 0.0
            && (frame.primary[0] - frame.shadow[0]).abs() > 5.0
        {
            lobe_flip_at = Some(frame.time);
        }
    }

    let t_cross = crossed_at.expect("separation must cross 0.1");
    assert!(
        (t_cross - 19.96).abs() <= 0.05,
        "Expected crossing near t = 19.96, got {t_cross}"
    );
    let t_flip = lobe_flip_at.expect("twins must flip onto opposite butterfly wings");
    assert!(
        (t_flip - 24.61).abs() <= 0.15,
        "Expected lobe flip near t = 24.61, got {t_flip}"
    );
}

#[test]
fn test_scrub_seek_and_step_back_reproduce_bit_identical_twin_resumption() {
    let params = Arc::new(RwLock::new(AttractorSharedParams {
        initial_state: ON_ATTRACTOR_LORENZ_STATE,
        simulation_speed: 1,
        ..Default::default()
    }));
    let mut runner = AttractorRunner::from_params(params);
    for _ in 0..80 {
        runner.step_once();
    }
    let expected_frame_30 = *runner.timeline.get(30).unwrap();
    let expected_frame_31 = *runner.timeline.get(31).unwrap();
    assert_eq!(runner.timeline.len(), 81);

    // Seek backward to frame 30; timeline retains 81 frames until forward stepping resumes
    runner.process_command(SimCommand::SeekFrame(30));
    assert_eq!(runner.execution_state, SolverExecutionState::Paused);
    assert_eq!(runner.step_count, 30);
    assert_eq!(runner.timeline.current_copied(), Some(expected_frame_30));
    assert_eq!(runner.timeline.len(), 81);

    // Stepping forward truncates frames > 30 and reproduces bit-identical frame 31
    runner.step_once();
    assert_eq!(runner.timeline.len(), 32);
    assert_eq!(runner.timeline.cursor(), 31);
    let resumed_31 = runner.timeline.current_copied().unwrap();
    assert_eq!(resumed_31.primary, expected_frame_31.primary);
    assert_eq!(resumed_31.shadow, expected_frame_31.shadow);
    assert_eq!(
        resumed_31.separation.to_bits(),
        expected_frame_31.separation.to_bits()
    );

    // StepBack(1) rewinds to exact frame 30 again
    runner.process_command(SimCommand::StepBack(1));
    assert_eq!(runner.timeline.current_copied(), Some(expected_frame_30));
}

#[test]
fn test_symbolic_ast_perturbation_and_aria_error_alerts() {
    clear_test_announcements();

    let mut plotter = AttractorPlotter {
        initial_state: [0.5, -1.0, 2.0],
        perturbation_expr: "1e-9 * cos(x0)".to_string(),
        ..Default::default()
    };
    plotter.perturbation_bar.set_expression("1e-9 * cos(x0)");
    plotter.recompute_perturbation_and_sync();
    assert!(plotter.perturbation_eval_error.is_none());
    let expected_dx = 1e-9 * 0.5_f64.cos();
    assert!((plotter.perturbation_offset[0] - expected_dx).abs() < 1e-22);
    assert!(!plotter.perturbation_bar.latex_cache.is_empty());

    // Invalid syntax triggers assertive ARIA alert via MathExpressionInputBar
    clear_test_announcements();
    plotter.perturbation_bar.set_expression("1e-9 * (");
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));
    let syntax_alerts = all_announcements();
    assert!(
        syntax_alerts.iter().any(|(_, prio)| prio == "assertive"),
        "Expected assertive syntax ARIA alert, got {syntax_alerts:?}"
    );

    // Runtime evaluation error (unknown variable) triggers SemanticPalette::error + assertive alert
    clear_test_announcements();
    plotter
        .perturbation_bar
        .set_expression("unknown_var * epsilon");
    plotter.recompute_perturbation_and_sync();
    assert!(plotter.perturbation_eval_error.is_some());
    let _ = ctx.run(egui::RawInput::default(), |ctx| plotter.show(ctx));
    let eval_alerts = all_announcements();
    assert!(
        eval_alerts
            .iter()
            .any(|(msg, prio)| prio == "assertive" && msg.contains("Evaluation Error")),
        "Expected assertive Evaluation Error ARIA alert, got {eval_alerts:?}"
    );
}

#[test]
fn test_zero_allocation_runner_stepping_across_all_three_chaos_models() {
    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());

    for kind in AttractorSystemKind::ALL {
        let params = Arc::new(RwLock::new(AttractorSharedParams {
            system_kind: kind,
            initial_state: kind.default_initial_state(),
            dt: kind.default_dt(),
            simulation_speed: 4,
            max_points: 512,
            shadow_enabled: true,
            perturbation_offset: [1e-9, 0.0, 0.0],
            extra: ExtraChaosParams::default(),
            ..Default::default()
        }));
        let mut runner = AttractorRunner::from_params(params);
        // Warm up once, then verify both step() and step_once() allocate 0 bytes
        runner.step_once();
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..32 {
                runner.step();
                runner.step_once();
            }
        });
        assert!(runner.is_finite_state());
        assert!(runner.timeline.len() > 100);
    }
}

#[test]
fn test_wolf_lyapunov_metrics_correlation_sum_exports_and_url_budget() {
    let mut plotter = AttractorPlotter::default();
    plotter
        .import_preset_json(crate::starter_gallery::PRESET_BUTTERFLY_TWINS)
        .unwrap();
    let saved = plotter.save_state().unwrap();
    assert!(
        saved.len() < 400,
        "Serialized AttractorState JSON must stay < 400 bytes, got {} bytes: {saved}",
        saved.len()
    );
    plotter.ensure_lyapunov_cache();
    assert!(
        (plotter.metrics_cache.wolf_lambda1 - 0.906).abs() < 0.08,
        "Expected Wolf lambda_1 near 0.906, got {}",
        plotter.metrics_cache.wolf_lambda1
    );
    assert!(plotter.metrics_cache.lyapunov_time().unwrap() > 1.0);

    for _ in 0..40 {
        plotter.step_simulation();
    }
    plotter.compute_correlation_dimension_on_demand();
    let (r, c_r) = plotter
        .metrics_cache
        .correlation_result
        .expect("C(r) computed");
    assert!(r > 0.0 && (0.0..=1.0).contains(&c_r));

    let datasets = plotter.export_datasets();
    assert_eq!(datasets.len(), 3);
    assert_eq!(datasets[0].0, "Trajectory");
    assert_eq!(datasets[1].0, "Shadow Trajectory");
    assert_eq!(datasets[2].0, "Log10 Divergence");

    // Check Rossler and DoublePendulum JSON budgets < 400 bytes as well
    for preset in [
        crate::starter_gallery::PRESET_ROSSLER_ATTRACTOR,
        crate::starter_gallery::PRESET_DOUBLE_PENDULUM,
    ] {
        plotter.import_preset_json(preset).unwrap();
        let json = plotter.save_state().unwrap();
        assert!(
            json.len() < 400,
            "Preset state exceeded 400 bytes ({}): {json}",
            json.len()
        );
    }
}

#[test]
fn test_state_guard_rollback_step_back_coexists_with_attractor_twins() {
    let mut fw = SimulationFramework::new("chaos");
    assert!(fw.select_tool_by_name("AttractorPlotter"));
    fw.cache_valid_snapshot();
    assert!(fw.last_valid_snapshot.is_some());

    // Corrupt active tool state and verify state_guard::rollback_step_back recovers cleanly
    let bad_state = r#"{"sigma":10.0,"rho":28.0,"beta":2.6667,"dt":0.01,"simulation_speed":5,"execution_state":{"Diverged":{"invalid_vec":["NaN",1.0,1.0],"diverged_at_step":7}},"show_annotations":true}"#;
    fw.active_tool
        .as_mut()
        .unwrap()
        .try_load_state(bad_state)
        .unwrap();
    assert!(!fw.active_tool.as_ref().unwrap().has_finite_parameters());

    assert!(fw.rollback_step_back());
    assert!(fw.active_tool.as_ref().unwrap().has_finite_parameters());
}
