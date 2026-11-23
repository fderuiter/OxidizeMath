#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn finite_difference_elasticity_and_batch_bound() {
    let mut engine = SensitivityEngine::new();
    assert!(!engine.is_converged());
    assert!(!engine.has_pending_work());
    let params = [
        ("p1", 1.0_f64),
        ("p2", 2.0_f64),
        ("p3", 3.0_f64),
        ("p4", 4.0_f64),
    ];
    let mut calls = 0_usize;
    engine.compute(&params, |map| {
        calls += 1;
        map["p1"] + 2.0 * map["p2"] + 3.0 * map["p3"].powi(3) + 4.0 * map["p4"]
    });
    assert!(calls <= MAX_SENSITIVITY_BATCH_EVALS);
    assert_eq!(calls, 5);
    assert!(!engine.is_converged());

    calls = 0;
    engine.compute(&params, |map| {
        calls += 1;
        map["p1"] + 2.0 * map["p2"] + 3.0 * map["p3"].powi(3) + 4.0 * map["p4"]
    });
    assert_eq!(calls, 4);
    assert!(engine.is_converged());
    assert!((engine.metrics[0].partial_derivative - 1.0).abs() < 1e-4);
    assert!((engine.metrics[1].partial_derivative - 2.0).abs() < 1e-4);
    assert!((engine.metrics[2].partial_derivative - 81.0).abs() < 1e-2);
    assert!(engine.metrics[2].is_high_sensitivity);
}

#[test]
fn step_tornado_respects_frame_budget_and_computes_swings() {
    let mut engine = SensitivityEngine::new();
    let params = [
        ("dose", 100.0_f64),
        ("ka", 1.0_f64),
        ("ke", 0.2_f64),
        ("vol", 20.0_f64),
        ("f_bio", 0.8_f64),
    ];
    let bounds = [
        TornadoBounds::new(50.0, 200.0),
        TornadoBounds::new(0.5, 2.0),
        TornadoBounds::new(0.05, 0.5),
        TornadoBounds::new(10.0, 40.0),
        TornadoBounds::new(0.4, 1.0),
    ];
    let eval =
        |vals: &[f64]| -> f64 { (vals[0] * vals[4] / vals[3]) * (vals[1] / (vals[1] + vals[2])) };

    let mut step1_calls = 0_usize;
    engine.step_tornado_slice(&params, &bounds, MAX_SENSITIVITY_BATCH_EVALS, |vals| {
        step1_calls += 1;
        eval(vals)
    });
    assert!(step1_calls <= MAX_SENSITIVITY_BATCH_EVALS);
    assert_eq!(step1_calls, 7); // 1 baseline + 3 params * 2 bounds
    assert!(!engine.is_tornado_converged());

    let mut step2_calls = 0_usize;
    engine.step_tornado_slice(&params, &bounds, MAX_SENSITIVITY_BATCH_EVALS, |vals| {
        step2_calls += 1;
        eval(vals)
    });
    assert!(step2_calls <= MAX_SENSITIVITY_BATCH_EVALS);
    assert_eq!(step2_calls, 4); // remaining 2 params * 2 bounds
    assert!(engine.is_tornado_converged());

    let intervals = engine.tornado_intervals();
    assert_eq!(intervals.len(), 5);
    assert!(intervals[0].swing > intervals[1].swing);
    assert!(intervals[0].min_output < engine.baseline_output);
    assert!(intervals[0].max_output > engine.baseline_output);
}

#[test]
fn seeded_splitmix64_monte_carlo_welford_and_quantiles_are_deterministic() {
    let params = [("mu", 10.0_f64), ("slope", 2.0_f64)];
    let unc = [
        ParameterUncertainty::new(1.5, 4.0, 16.0),
        ParameterUncertainty::new(0.25, 1.0, 3.0),
    ];
    let eval = |vals: &[f64]| vals[0] + 3.0 * vals[1];

    let mut engine_a = SensitivityEngine::new();
    let mut engine_b = SensitivityEngine::new();
    while !engine_a.is_monte_carlo_converged() {
        let mut batch_calls = 0_usize;
        engine_a.step_monte_carlo_slice(&params, &unc, 0x1234_5678, 128, 8, |vals| {
            batch_calls += 1;
            eval(vals)
        });
        assert!(batch_calls <= MAX_SENSITIVITY_BATCH_EVALS);
    }
    while !engine_b.is_monte_carlo_converged() {
        engine_b.step_monte_carlo_slice(&params, &unc, 0x1234_5678, 128, 5, eval);
    }

    let sum_a = engine_a.monte_carlo_summary().expect("converged summary");
    let sum_b = engine_b.monte_carlo_summary().expect("converged summary");
    assert_eq!(sum_a, sum_b);
    assert_eq!(sum_a.sample_count, 128);
    assert!(sum_a.min <= sum_a.p05);
    assert!(sum_a.p05 <= sum_a.p50);
    assert!(sum_a.p50 <= sum_a.p95);
    assert!(sum_a.p95 <= sum_a.max);
    assert!((sum_a.mean - 16.0).abs() < 0.6);
    assert!(sum_a.std_dev > 0.8 && sum_a.std_dev < 2.6);
}
