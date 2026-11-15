#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, TraceSeries};

#[test]
fn test_ode_divergence_detection() {
    let mut tool = OdeSolverTool::default();
    assert!(!tool.diverged());

    // Set exponential growth rate and initial value that causes f64 overflow/divergence
    tool.param_k = 1e300;
    tool.ic_y0 = 1e300;
    tool.recalculate();

    assert!(tool.diverged());
    // Verify time_series and y_series do not contain non-finite values
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.time_series.iter().all(|t| t.is_finite()));

    // Reset parameters recovers from divergence
    tool.reset_parameters();
    assert!(!tool.diverged());
    assert!(!tool.y_series.is_empty());
}

#[test]
fn test_ode_nan_initial_condition() {
    let mut tool = OdeSolverTool {
        ic_y0: f64::NAN,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.is_empty());
}

#[test]
fn test_ode_harmonic_oscillator_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::HarmonicOscillator,
        param_k: 1e300,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.v_series.iter().all(|v| v.is_finite()));
}

#[test]
fn test_ode_logistic_growth_divergence() {
    let mut tool = OdeSolverTool {
        preset: OdePreset::LogisticGrowth,
        param_r: 1e300,
        param_cap_k: 1.0,
        ic_y0: 1e300,
        ..Default::default()
    };
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
}

#[test]
fn test_ode_trace_snapshot_pinning_and_export() {
    let mut tool = OdeSolverTool {
        param_k: 1.0,
        ..Default::default()
    };
    tool.recalculate();

    // Pin baseline trajectory k=1.0
    let summary = tool.param_summary();
    let pts: Vec<[f64; 2]> = tool
        .time_series
        .iter()
        .zip(&tool.y_series)
        .map(|(&t, &y)| [t, y])
        .collect();
    tool.snapshot_manager.add_snapshot(
        "Baseline 1",
        summary,
        vec![TraceSeries::new("y(t)", pts.clone())],
    );

    assert_eq!(tool.snapshot_manager.len(), 1);

    // Change parameter k=2.5 and recalculate active simulation
    tool.param_k = 2.5;
    tool.recalculate();

    // Pinned snapshot retains baseline curve points
    assert_eq!(
        tool.snapshot_manager.snapshots()[0].param_summary,
        r"k=1.00, \Delta t=0.05"
    );
    assert_eq!(tool.snapshot_manager.snapshots()[0].series[0].points, pts);

    // Export datasets includes active + pinned traces
    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Baseline 1")));

    // State serialization roundtrip preserves snapshot
    let state = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&state);
    assert_eq!(restored.snapshot_manager.len(), 1);
}

#[test]
fn test_ode_auto_history_buffering() {
    let mut tool = OdeSolverTool::default();
    assert_eq!(tool.snapshot_manager.history_len(), 0);

    for i in 1..=7 {
        let old_summary = tool.param_summary();
        let pts: Vec<[f64; 2]> = tool.time_series.iter().zip(&tool.y_series).map(|(&t, &y)| [t, y]).collect();
        tool.snapshot_manager.push_history_run(old_summary, vec![TraceSeries::new("y(t)", pts)]);
        tool.param_k = i as f64 * 0.5;
        tool.recalculate();
    }

    assert_eq!(tool.snapshot_manager.history_len(), 5);
    assert!(tool.snapshot_manager.history_runs()[0].param_summary.contains("k=1.00"));
    assert!(tool.snapshot_manager.history_runs()[4].param_summary.contains("k=3.00"));

    let exported = tool.export_datasets();
    assert!(exported.iter().any(|(name, _)| name.contains("Run")));
}

#[test]
fn test_ode_annotations_toggle_and_rendering() {
    let mut tool = OdeSolverTool::default();
    assert!(tool.show_annotations);

    // Toggle off and on
    tool.show_annotations = false;
    assert!(!tool.show_annotations);
    tool.show_annotations = true;

    // Test serialization roundtrip with show_annotations
    tool.show_annotations = false;
    let saved = tool.save_state().expect("save state");
    let mut restored = OdeSolverTool::default();
    restored.load_state(&saved);
    assert!(!restored.show_annotations);

    // Test rendering with each preset
    let ctx = egui::Context::default();

    // 1. Exponential
    tool.show_annotations = true;
    tool.preset = OdePreset::Exponential;
    tool.param_k = -0.5;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // 2. Harmonic Oscillator
    tool.preset = OdePreset::HarmonicOscillator;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // 3. Logistic Growth
    tool.preset = OdePreset::LogisticGrowth;
    tool.param_cap_k = 12.0;
    tool.ic_y0 = 2.0;
    tool.recalculate();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Disable annotations and ensure clean rendering
    tool.show_annotations = false;
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));

    // Export plot screenshot for verification
    let datasets = tool.export_datasets();
    let options = egui_plot::export::ExportOptions::default();
    if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
        let _ = std::fs::write("/tmp/ode_solver_trajectory.png", png_data);
    }
}
