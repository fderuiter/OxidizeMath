#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
        "k=1.00, dt=0.05"
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
