#![cfg_attr(any(), verified(opt_out = "tests"))]

use approx::assert_relative_eq;
use oxidize_core::rng::OxidizeRng;
use verified_engine::engine::VerifiedEngine;

use super::{
    EW_BETA_1D, InterfaceGrowth1D, InterfaceGrowth2D, InterfaceGrowthKind, InterfaceGrowthParams,
    InterfaceScalingTelemetry, KPZ_BETA_1D,
};

#[test]
fn seeded_1d_edwards_wilkinson_growth_exponent_matches_quarter_power_law() {
    let len = 512usize;
    let params = InterfaceGrowthParams {
        nu: 1.0,
        lambda: 0.0,
        noise_sigma: 1.0,
        dx: 1.0,
        dt: 0.05,
    };
    let sample_steps = 20usize; // dt * 20 = 1.0 time unit per sample
    let total_samples = 40usize; // t in [1.0, 40.0], well before L^2 saturation
    let ensemble_runs = 6usize;

    let mut mean_widths = vec![0.0; total_samples];
    let mut times = vec![0.0; total_samples];

    for run in 0..ensemble_runs {
        let mut rng = OxidizeRng::from_seed(0xE001_0000 + (run as u64) * 7919);
        let mut solver = InterfaceGrowth1D::new(len, params);
        for s in 0..total_samples {
            solver.step_n(&mut rng, sample_steps);
            times[s] = solver.time();
            mean_widths[s] += solver.interface_width() / (ensemble_runs as f64);
        }
    }

    let mut telemetry = InterfaceScalingTelemetry::new(64);
    for (&t, &w) in times.iter().zip(mean_widths.iter()) {
        telemetry.record_ew_1d(t, w, len as f64);
    }

    let (beta_ew, _) = telemetry
        .fit_growth_exponent_beta(4.0, 40.0)
        .expect("EW scaling regression must succeed");
    assert!(
        (beta_ew - EW_BETA_1D).abs() < 0.035,
        "Expected 1D EW beta near 0.25, got {beta_ew:.4}"
    );
}

#[test]
fn seeded_1d_kpz_growth_exponent_matches_one_third_power_law() {
    let len = 512usize;
    let params = InterfaceGrowthParams {
        nu: 0.5,
        lambda: 3.5,
        noise_sigma: 1.0,
        dx: 1.0,
        dt: 0.025,
    };
    let sample_steps = 40usize; // dt * 40 = 1.0 time unit per sample
    let total_samples = 42usize; // t in [1.0, 42.0]
    let ensemble_runs = 6usize;

    let mut mean_widths = vec![0.0; total_samples];
    let mut times = vec![0.0; total_samples];

    for run in 0..ensemble_runs {
        let mut rng = OxidizeRng::from_seed(0x4B5A_2026 + (run as u64) * 104_729);
        let mut solver = InterfaceGrowth1D::new(len, params);
        for s in 0..total_samples {
            solver.step_n(&mut rng, sample_steps);
            times[s] = solver.time();
            mean_widths[s] += solver.interface_width() / (ensemble_runs as f64);
        }
    }

    let mut telemetry = InterfaceScalingTelemetry::new(64);
    for (&t, &w) in times.iter().zip(mean_widths.iter()) {
        telemetry.record_kpz_1d(t, w, len as f64);
    }

    let (beta_kpz, _) = telemetry
        .fit_growth_exponent_beta(6.0, 42.0)
        .expect("KPZ scaling regression must succeed");
    assert!(
        (beta_kpz - KPZ_BETA_1D).abs() < 0.035,
        "Expected 1D KPZ beta near 0.3333, got {beta_kpz:.4}"
    );
}

#[test]
fn family_vicsek_collapse_and_decimation_preserve_scaling_invariants() {
    let (alpha, beta, z) = InterfaceGrowthKind::Kpz.scaling_exponents(1);
    assert_relative_eq!(alpha + z, 2.0, epsilon = 1e-12);
    assert_relative_eq!(beta, alpha / z, epsilon = 1e-12);

    let mut tel = InterfaceScalingTelemetry::new(8);
    let l = 64.0;
    for step in 1..=16 {
        let t = step as f64;
        let w = 0.8 * t.powf(KPZ_BETA_1D);
        tel.record_kpz_1d(t, w, l);
    }
    assert!(tel.len() <= 8);
    let (fitted_beta, _) = tel
        .fit_growth_exponent_beta(1.0, 16.0)
        .expect("decimated series must remain valid");
    assert_relative_eq!(fitted_beta, KPZ_BETA_1D, epsilon = 1e-10);
}

#[test]
fn kpz_1d_and_2d_stepping_perform_zero_heap_allocations() {
    let params = InterfaceGrowthParams::default();
    let mut rng = OxidizeRng::from_seed(9001);
    let mut kpz_1d = InterfaceGrowth1D::new(128, params);
    let mut kpz_2d = InterfaceGrowth2D::new(32, 32, params);
    let mut tel = InterfaceScalingTelemetry::new(32);

    VerifiedEngine::run_verified(|| {
        kpz_1d.deposit_mound(64, 6, 1.5);
        kpz_1d.step_n(&mut rng, 4);
        let w1 = kpz_1d.interface_width();
        tel.record_kpz_1d(kpz_1d.time(), w1, kpz_1d.len() as f64);

        kpz_2d.deposit_mound(16, 16, 4, 1.2);
        kpz_2d.step_n(&mut rng, 2);
        let w2 = kpz_2d.interface_width();
        assert!(w1.is_finite() && w2 > 0.0);
    });
}
