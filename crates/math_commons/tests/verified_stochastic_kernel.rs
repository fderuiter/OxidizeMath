//! Zero-allocation and numerical accuracy verification for `math_commons::stochastic`.
#![cfg(feature = "std")]

use math_commons::stochastic::{
    SolverWorkspace, calibrate_ou, ctmc_steady_state, dtmc_stationary_distribution,
};
use std::hint::black_box;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

fn deterministic_normal_pair(seed: &mut u64) -> (f64, f64) {
    fn next_u01(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let bits = (*state >> 11) | 1;
        (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
    let u1 = next_u01(seed).max(1e-15);
    let u2 = next_u01(seed);
    let r = (-2.0 * u1.ln()).sqrt();
    let angle = core::f64::consts::TAU * u2;
    (r * angle.cos(), r * angle.sin())
}

fn simulate_exact_ou(
    mu: f64,
    theta: f64,
    sigma: f64,
    x0: f64,
    dt: f64,
    steps: usize,
    seed: u64,
) -> Vec<f64> {
    let mut state = seed;
    let phi = (-theta * dt).exp();
    let step_std = (sigma * sigma * (-libm::expm1(-2.0 * theta * dt)) / (2.0 * theta)).sqrt();
    let mut path = Vec::with_capacity(steps + 1);
    let mut x = x0;
    path.push(x);
    for idx in 0..steps {
        let (z0, z1) = deterministic_normal_pair(&mut state);
        let z = if idx % 2 == 0 { z0 } else { z1 };
        x = mu + phi * (x - mu) + step_std * z;
        path.push(x);
    }
    path
}

#[test]
fn periodic_three_state_dtmc_and_multi_sink_rejection_are_zero_alloc() {
    let periodic = [0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let multi_absorbing = [
        1.0, 0.0, 0.0, 0.0, //
        0.5, 0.0, 0.5, 0.0, //
        0.0, 0.5, 0.0, 0.5, //
        0.0, 0.0, 0.0, 1.0, //
    ];
    let transient_to_sink = [
        0.2, 0.4, 0.4, //
        0.0, 0.0, 1.0, //
        0.0, 1.0, 0.0, //
    ];

    let mut ws = SolverWorkspace::new(4, 0);
    let mut pi3 = [0.0_f64; 3];
    let mut pi4 = [0.0_f64; 4];

    VerifiedEngine::run_verified(|| {
        let ok_periodic = ws.solve_dtmc_stationary(black_box(&periodic), 3, &mut pi3);
        assert!(ok_periodic);
        assert!((pi3[0] - 0.5).abs() < 1e-12);
        assert!((pi3[1] - 0.25).abs() < 1e-12);
        assert!((pi3[2] - 0.25).abs() < 1e-12);

        ws.operator_staging_mut(3).copy_from_slice(&periodic);
        let mut staged_pi = [0.0_f64; 3];
        assert!(ws.solve_staged_dtmc_stationary(3, &mut staged_pi));
        assert_eq!(pi3, staged_pi);

        let ok_multi = ws.solve_dtmc_stationary(black_box(&multi_absorbing), 4, &mut pi4);
        assert!(!ok_multi);

        let ok_transient_sink =
            ws.solve_dtmc_stationary(black_box(&transient_to_sink), 3, &mut pi3);
        assert!(ok_transient_sink);
        assert!(pi3[0].abs() < 1e-14);
        assert!((pi3[1] - 0.5).abs() < 1e-12);
        assert!((pi3[2] - 0.5).abs() < 1e-12);
    });

    assert!(dtmc_stationary_distribution(&multi_absorbing, 4).is_none());
}

#[test]
fn slow_ctmc_rates_solve_accurately_with_zero_allocations() {
    let g_slow = [-1e-7, 1e-7, 3e-7, -3e-7];
    let mut ws = SolverWorkspace::new(4, 0);
    let mut pi = [0.0_f64; 2];

    VerifiedEngine::run_verified(|| {
        let solved = ws.solve_ctmc_steady_state(black_box(&g_slow), 2, &mut pi);
        assert!(solved);
        assert!((pi[0] - 0.75).abs() < 1e-12);
        assert!((pi[1] - 0.25).abs() < 1e-12);

        ws.operator_staging_mut(2).copy_from_slice(&g_slow);
        let mut staged_pi = [0.0_f64; 2];
        assert!(ws.solve_staged_ctmc_steady_state(2, &mut staged_pi));
        assert_eq!(pi, staged_pi);
    });

    let r0 = pi[0] * g_slow[0] + pi[1] * g_slow[2];
    let r1 = pi[0] * g_slow[1] + pi[1] * g_slow[3];
    let scale_rel_res = (r0 * r0 + r1 * r1).sqrt() / 3e-7;
    assert!(
        scale_rel_res < 1e-12,
        "expected scale-relative residual < 1e-12, got {scale_rel_res:e}"
    );

    let vec_pi = ctmc_steady_state(&g_slow, 2).expect("unique slow CTMC steady state");
    assert!((vec_pi[0] - 0.75).abs() < 1e-12);
    assert!((vec_pi[1] - 0.25).abs() < 1e-12);
}

#[test]
fn ou_calibration_and_quantile_summary_are_zero_alloc_and_accurate() {
    let slow_obs = simulate_exact_ou(2.0, 0.08, 0.35, 2.0, 0.02, 4_000, 0xCAFE_BABE_1234_5678);
    let fast_obs = simulate_exact_ou(-1.2, 6.0, 0.45, -1.2, 0.05, 4_000, 0xDEAD_BEEF_8765_4321);

    let mut ws = SolverWorkspace::new(0, 4_096);

    VerifiedEngine::run_verified(|| {
        let slow_cal = ws
            .calibrate_ou(black_box(&slow_obs), 0.02)
            .expect("slow OU calibration");
        assert!((slow_cal.sigma - 0.35).abs() / 0.35 < 0.08);

        let fast_cal = ws
            .calibrate_ou(black_box(&fast_obs), 0.05)
            .expect("fast OU calibration");
        assert!((fast_cal.mu - (-1.2)).abs() < 0.08);
        assert!((fast_cal.theta - 6.0).abs() / 6.0 < 0.12);
        assert!((fast_cal.sigma - 0.45).abs() / 0.45 < 0.08);
        assert_eq!(calibrate_ou(&fast_obs, 0.05), Some(fast_cal));

        let summary = ws
            .summarize_samples(black_box(&fast_obs), Some(-1.2))
            .expect("sample summary");
        assert!((summary.mean - (-1.2)).abs() < 0.08);
        assert!(summary.percentile_05 < summary.median);
        assert!(summary.median < summary.percentile_95);
        assert!((summary.prob_above_threshold - 0.5).abs() < 0.08);
    });
}

#[cfg(feature = "serde")]
#[test]
fn stochastic_summaries_serde_roundtrip() {
    use math_commons::stochastic::{OuCalibration, SampleSummary, SccSummary};

    let scc = SccSummary {
        num_components: 3,
        num_sink_components: 1,
        unique_sink_size: 2,
    };
    let scc_json = serde_json::to_string(&scc).expect("serialize SccSummary");
    assert_eq!(
        serde_json::from_str::<SccSummary>(&scc_json).expect("deserialize SccSummary"),
        scc
    );

    let cal = OuCalibration {
        mu: -1.2,
        theta: 6.0,
        sigma: 0.45,
    };
    let cal_json = serde_json::to_string(&cal).expect("serialize OuCalibration");
    assert_eq!(
        serde_json::from_str::<OuCalibration>(&cal_json).expect("deserialize OuCalibration"),
        cal
    );

    let sample = SampleSummary {
        mean: 0.0,
        std_dev: 1.0,
        median: 0.01,
        percentile_05: -1.64,
        percentile_95: 1.65,
        prob_above_threshold: 0.5,
    };
    let sample_json = serde_json::to_string(&sample).expect("serialize SampleSummary");
    assert_eq!(
        serde_json::from_str::<SampleSummary>(&sample_json).expect("deserialize SampleSummary"),
        sample
    );
}
