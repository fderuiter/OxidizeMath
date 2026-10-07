#![cfg_attr(any(), verified(opt_out = "unit_tests"))]

use super::*;

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
fn test_ode_reset_parameters() {
    let mut tool = OdeSolverTool::default();

    // Exponential preset
    tool.param_k = 5.0;
    tool.ic_y0 = 3.0;
    tool.dt = 0.1;
    tool.total_time = 20.0;
    tool.reset_parameters();
    assert_eq!(tool.param_k, 1.0);
    assert_eq!(tool.ic_y0, 1.0);
    assert_eq!(tool.dt, 0.05);
    assert_eq!(tool.total_time, 10.0);
    assert!(!tool.y_series.is_empty());

    // Harmonic Oscillator preset
    tool.preset = OdePreset::HarmonicOscillator;
    tool.param_k = 4.0;
    tool.ic_y0 = 2.0;
    tool.ic_v0 = -1.0;
    tool.reset_parameters();
    assert_eq!(tool.param_k, 1.0);
    assert_eq!(tool.ic_y0, 1.0);
    assert_eq!(tool.ic_v0, 0.0);

    // Logistic Growth preset
    tool.preset = OdePreset::LogisticGrowth;
    tool.param_r = 2.5;
    tool.param_cap_k = 50.0;
    tool.reset_parameters();
    assert_eq!(tool.param_r, 1.0);
    assert_eq!(tool.param_cap_k, 10.0);
}
