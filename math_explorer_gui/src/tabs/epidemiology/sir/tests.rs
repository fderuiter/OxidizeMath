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
    assert!(
        out1.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );

    let _ = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(tool.sensitivity_engine.is_converged());
    let out_idle = ctx.run(input.clone(), |ctx| tool.show(ctx));
    assert!(
        !out_idle.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );
    assert!(tool.sensitivity_engine.metrics[2].partial_derivative > 0.0);
    assert!(tool.sensitivity_engine.metrics[3].partial_derivative < 0.0);

    // Changing duration invalidates the 4-parameter sensitivity cache
    tool.duration = 180.0;
    tool.recalculate();
    let out_after_duration = ctx.run(input, |ctx| tool.show(ctx));
    assert!(!tool.sensitivity_engine.is_converged());
    assert!(
        out_after_duration.viewport_output[&egui::ViewportId::ROOT]
            .repaint_delay
            .is_zero()
    );
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

    let r_pts: Vec<[f64; 2]> = tool.r_points.iter().map(|p| [p.x, p.y]).collect();
    let fit = egui_plot::fit_model_with_axes(
        &r_pts,
        egui_plot::FitType::Logistic,
        "Recovered",
        false,
        false,
        false,
    )
    .expect("SIR Recovered logistic fit");
    assert!(
        fit.r_squared >= 0.99,
        "SIR Recovered Logistic R^2 = {}",
        fit.r_squared
    );
}

