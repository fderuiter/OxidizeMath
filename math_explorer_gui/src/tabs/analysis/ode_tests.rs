#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;

#[test]
fn test_ode_divergence_detection() {
    let mut tool = OdeSolverTool::default();
    assert!(!tool.diverged());

    tool.param_k = 1e300;
    tool.ic_y0 = 1e300;
    tool.recalculate();

    assert!(tool.diverged());
    assert!(tool.y_series.iter().all(|y| y.is_finite()));
    assert!(tool.time_series.iter().all(|t| t.is_finite()));

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
