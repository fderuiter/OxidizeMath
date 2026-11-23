#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_sir_trace_snapshot_pinning_and_exports() {
    let mut tool = SirTool {
        beta: 0.2,
        ..Default::default()
    };
    tool.recalculate();

    // Pin snapshot for beta = 0.2
    let summary = tool.param_summary();
    let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
    let series = vec![
        TraceSeries::new("Susceptible", to_pairs(&tool.s_points)),
        TraceSeries::new("Infected", to_pairs(&tool.i_points)),
        TraceSeries::new("Recovered", to_pairs(&tool.r_points)),
    ];
    assert!(tool.snapshot_manager.add_snapshot("Run 1", summary, series));
    assert_eq!(tool.snapshot_manager.len(), 1);

    // Adjust beta to 0.5
    tool.beta = 0.5;
    tool.recalculate();

    // Snapshot remains preserved
    assert_eq!(
        tool.snapshot_manager.snapshots()[0].param_summary,
        r"\beta=0.20, \gamma=0.10, N=1000"
    );

    // Export includes both active simulation curves and pinned snapshot traces
    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Infected")));
    assert!(exported.iter().any(|(name, _)| name.contains("Run 1")));

    // State serialization roundtrip preserves pinned snapshot
    let state = tool.save_state().expect("serialize state");
    let mut restored = SirTool::default();
    restored.load_state(&state);
    assert_eq!(restored.snapshot_manager.len(), 1);
    assert_eq!(restored.beta, 0.5);
}

#[test]
fn test_sir_borrowed_slice_points() {
    let tool = SirTool::default();
    let s_slice: &[PlotPoint] = &tool.s_points[..];
    let plot_points: egui_plot::PlotPoints<'_> = s_slice.into();
    assert!(matches!(plot_points, egui_plot::PlotPoints::Borrowed(_)));
    assert_eq!(plot_points.points().len(), tool.s_points.len());
}

#[test]
fn test_sir_sensitivity_converges_in_two_frames_and_invalidates_on_duration() {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 640.0),
        )),
        ..Default::default()
    };
    let mut tool = SirTool::default();

    let out1 = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());
    assert!(out1.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());

    let _ = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let out_idle = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(!out_idle.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
    assert!(tool.sensitivity_engine.metrics[2].partial_derivative > 0.0);
    assert!(tool.sensitivity_engine.metrics[3].partial_derivative < 0.0);

    // Changing duration invalidates the 4-parameter sensitivity cache
    tool.duration = 180.0;
    tool.recalculate();
    let out_after_duration = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());
    assert!(out_after_duration.viewport_output[&egui::ViewportId::ROOT]
        .repaint_delay
        .is_zero());
    let _ = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());

    // Pointer-down drag defers SIR perturbations and settles on release
    let settled_baseline = tool.sensitivity_engine.baseline_output;
    tool.beta = 1.2;
    tool.recalculate();
    let mut drag_in = input.clone();
    drag_in.events.push(egui::Event::PointerButton {
        pos: egui::pos2(50.0, 50.0),
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run(drag_in, |ctx| tool.show(ctx));
    assert_eq!(tool.sensitivity_engine.baseline_output, settled_baseline);

    let mut release_in = input.clone();
    release_in.events.push(egui::Event::PointerButton {
        pos: egui::pos2(50.0, 50.0),
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    });
    let _ = ctx.run(release_in, |ctx| tool.show(ctx));
    let _ = ctx.run(input, |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    assert!((tool.sensitivity_engine.baseline_output - settled_baseline).abs() > 1.0);
}

#[test]
fn test_sir_load_state_clamps_duration_and_bounds_steps() {
    let mut sir = SirTool::default();
    sir.load_state(r#"{"n":1000.0,"i0":10.0,"beta":0.5,"gamma":0.1,"duration":500000.0}"#);
    assert_eq!(sir.duration, MAX_SIR_DURATION);
    assert!(sir.export_datasets()[0].1.len() <= 1_000_001);
    assert_eq!(sir.s_points.len(), 3_651);

    let preset_res = sir.import_preset_json(r#"{"metadata":{"schema_version":"1.0","title":"SIR","domain":"epidemiology","timestamp":"2026-11-18T00:00:00Z"},"config":{"n":1000.0,"i0":10.0,"beta":0.5,"gamma":0.1,"duration":0.0}}"#);
    assert!(preset_res.is_ok());
    assert_eq!(sir.duration, MIN_SIR_DURATION);
    assert!(sir.has_finite_parameters());

    let neg_res =
        sir.try_load_state(r#"{"n":1000.0,"i0":10.0,"beta":0.5,"gamma":0.1,"duration":-5.0}"#);
    assert!(neg_res.is_err());

    sir.duration = 1e20;
    sir.recalculate();
    assert_eq!(sir.s_points.len(), MAX_SIR_STEPS + 1);

    sir.duration = f64::NAN;
    sir.recalculate();
    assert!(sir.s_points.is_empty());
}

#[test]
fn test_sir_epidemic_peak_and_logistic_fit_r2() {
    let tool = SirTool::default();
    let (peak_day, peak_infected) =
        metrics::compute_epidemic_peak(&tool.i_points).expect("epidemic peak");
    let max_sample = tool
        .i_points
        .iter()
        .map(|p| p.y)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(peak_day > 10.0 && peak_day < 40.0);
    assert!(peak_infected >= max_sample - 1e-6);

    let fit = math_explorer::fit::fit_model(
        &tool.r_points[..],
        math_explorer::fit::FitModel::Logistic,
        false,
    )
    .expect("SIR Recovered logistic fit");
    assert!(
        fit.r_squared >= 0.99,
        "SIR Recovered Logistic R^2 = {}",
        fit.r_squared
    );

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        verified_engine::engine::VerifiedEngine::run_verified(|| {
            let pk = metrics::compute_epidemic_peak(&tool.i_points);
            assert!(pk.is_some());
        });
    }
}

#[test]
fn test_sir_animated_slider_sweep_recalculates_without_history_flood() {
    use crate::reflective_ui::slider::{save_slider_state, AnimatedSliderState, SweepMode};

    let mut tool = SirTool::default();
    let initial_beta = tool.beta;
    let initial_i_peak = tool.i_points.iter().map(|p| p.y).fold(0.0_f64, f64::max);

    let ctx = egui::Context::default();
    save_slider_state(
        &ctx,
        egui::Id::new("sir_beta"),
        AnimatedSliderState {
            playing: true,
            direction: 1.0,
            sweep_mode: SweepMode::Oscillate,
        },
    );
    let _ = ctx.run(
        egui::RawInput {
            predicted_dt: 0.1,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let updated_i_peak = tool.i_points.iter().map(|p| p.y).fold(0.0_f64, f64::max);
    assert!(tool.beta > initial_beta);
    assert_ne!(updated_i_peak, initial_i_peak);
    assert_eq!(tool.snapshot_manager.history_len(), 0);
}

#[test]
fn test_sir_100k_steps_decimates_display_polylines_and_preserves_peak_and_exports() {
    let mut tool = SirTool {
        duration: 10_000.0,
        ..Default::default()
    };
    tool.recalculate();
    assert_eq!(tool.i_points.len(), 100_001);
    assert!(tool.display_s.len() <= crate::framework::MAX_DISPLAY_POLYLINE_POINTS);
    assert!(tool.display_i.len() <= crate::framework::MAX_DISPLAY_POLYLINE_POINTS);
    assert!(tool.display_r.len() <= crate::framework::MAX_DISPLAY_POLYLINE_POINTS);

    let full_peak = metrics::compute_epidemic_peak(&tool.i_points).expect("full peak");
    let short_tool = SirTool::default();
    let short_peak = metrics::compute_epidemic_peak(&short_tool.i_points).expect("365d peak");
    assert!((full_peak.0 - short_peak.0).abs() < 1e-9);
    assert!((full_peak.1 - short_peak.1).abs() < 1e-9);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    let plot_datasets = egui_plot::get_last_plot_datasets();
    assert!(plot_datasets
        .iter()
        .all(|(_, pts)| pts.len() <= crate::framework::MAX_DISPLAY_POLYLINE_POINTS));

    let exported = tool.export_datasets();
    assert_eq!(exported[0].1.len(), 100_001);
    assert_eq!(exported[1].1.len(), 100_001);
    assert_eq!(exported[2].1.len(), 100_001);
}

#[test]
fn test_sir_forked_100k_steps_decimates_both_branches_and_exports_full_series() {
    let mut tool = SirTool {
        duration: 10_000.0,
        ..Default::default()
    };
    assert!(tool.fork_at_day(10.0));
    tool.beta = 0.2;
    tool.recalculate();

    let cap = crate::framework::MAX_DISPLAY_POLYLINE_POINTS;
    assert!(tool.display_s.len() <= cap && tool.cf.display_s_a.len() <= cap);
    assert!(tool.display_i.len() <= cap && tool.cf.display_i_a.len() <= cap);
    assert!(tool.display_r.len() <= cap && tool.cf.display_r_a.len() <= cap);

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| tool.show(c));
    let plot_datasets = egui_plot::get_last_plot_datasets();
    assert_eq!(plot_datasets.len(), 6);
    assert!(plot_datasets.iter().all(|(_, pts)| pts.len() <= cap));

    let exported = tool.export_datasets();
    assert_eq!(exported.len(), 6);
    assert!(exported.iter().all(|(_, pts)| pts.len() == 100_001));
}

#[test]
fn test_sir_plot_uses_shared_import_csv_toolbar_without_duplicate_header_button() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let mut tool = SirTool::default();
    let out = ctx.run(input, |c| tool.show(c));
    let ak = out
        .platform_output
        .accesskit_update
        .as_ref()
        .expect("AccessKit update");
    let labels: Vec<&str> = ak
        .nodes
        .iter()
        .filter(|(_, n)| n.role() == egui::accesskit::Role::Button)
        .filter_map(|(_, n)| n.label())
        .collect();
    assert!(labels.contains(&"Import CSV"));
    assert!(!labels.contains(&"Import Dataset (CSV)"));
}

#[test]
fn test_sir_timeline_scrub_zero_perturbation_fork_and_counterfactual_intervention() {
    use crate::accessibility::{all_announcements, clear_test_announcements};
    use crate::framework::timeline_branch::exports::assert_framework_exports_all_branch_formats;
    use crate::framework::MAX_TIMELINE_RING_BYTES;
    use crate::url_state::URL_JSON_BOUNDS;

    let mut tool = SirTool {
        beta: 0.30,
        ..SirTool::default()
    };
    tool.recalculate();
    assert!(tool.branch.timeline.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);

    let expected_s_at_180 = tool.s_points[180].y;
    let expected_i_at_180 = tool.i_points[180].y;
    tool.scrub_to_step(180);
    assert_eq!(tool.branch.current_step, 180);
    assert_eq!(tool.s_points.len(), 181);
    assert!((tool.s_points[180].x - 18.0).abs() < 1.0e-9);
    assert!((tool.s_points[180].y - expected_s_at_180).abs() < 1.0e-9);
    assert!((tool.i_points[180].y - expected_i_at_180).abs() < 1.0e-9);

    clear_test_announcements();
    let ctx = egui::Context::default();
    tool.apply_timeline_action(TimelineBranchAction::ToggleForkBranch, Some(&ctx));
    assert!(all_announcements()
        .iter()
        .any(|(msg, pri)| pri == "polite" && msg.contains("Forked")));
    tool.step_lockstep(100);
    let zero_div = tool.branch_divergence_metrics().expect("Fork active");
    assert_eq!(zero_div.steps_since_fork, 100);
    assert_eq!(zero_div.instantaneous_l2, 0.0);
    assert_eq!(zero_div.peak_abs_delta, 0.0);
    assert_eq!(zero_div.cumulative_area_delta, 0.0);

    tool.beta = 0.14;
    tool.recompute_counterfactual_from_fork(false);
    let div = tool.branch_divergence_metrics().expect("Fork active");
    assert!(div.instantaneous_l2 > 0.0 && div.peak_abs_delta > 0.0);
    let (_, intervention_peak) = metrics::compute_epidemic_peak(&tool.i_points).unwrap();
    let (_, baseline_peak) =
        metrics::compute_epidemic_peak(&tool.branch.shadow.as_ref().unwrap().i_points).unwrap();
    assert!(intervention_peak < baseline_peak);

    let saved = tool.save_state().expect("save_state");
    assert!(saved.len() <= URL_JSON_BOUNDS.max_bytes && !saved.contains("Shadow Baseline"));
    assert_framework_exports_all_branch_formats(Box::new(tool), "Infected (Shadow Baseline");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_sir_lockstep_shadow_twin_step_zero_heap_allocations() {
    let mut tool = SirTool::default();
    tool.scrub_to_step(180);
    tool.fork_branch();
    tool.beta = 0.25;
    let dyn_now = tool.current_dynamics();

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((l2, peak), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        tool.branch.step_lockstep_in_place(dyn_now, branch::SIR_DT)
    });
    assert!(l2 > 0.0 && peak > 0.0);
}
