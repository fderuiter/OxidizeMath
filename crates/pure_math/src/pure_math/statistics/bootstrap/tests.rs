use super::*;
use crate::pure_math::statistics::monte_carlo::normal_inv_cdf_fast;
use oxidize_core::rng::OxidizeRng;
use rand::RngCore;
use verified_engine::engine::VerifiedEngine;

#[test]
#[verified_engine::verified]
fn test_bootstrap_matches_analytical_normal_interval_on_gaussian_sample() {
    let n = 120_usize;
    let mut rng = OxidizeRng::new(314159);
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        let u = (((rng.next_u64() >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64);
        data.push(10.0 + 2.0 * normal_inv_cdf_fast(u));
    }

    let mean = sample_mean(&data);
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / ((n - 1) as f64);
    let analytical_se = (var / (n as f64)).sqrt();
    let analytical_lo = mean - 1.959963984540054 * analytical_se;
    let analytical_hi = mean + 1.959963984540054 * analytical_se;

    let mut ws = BootstrapWorkspace::new(n, 2000).expect("valid bootstrap workspace");
    let cmp = ws
        .compare_intervals(&data, 2000, 0.95, &mut rng, sample_mean)
        .expect("bootstrap comparison");

    assert!((cmp.percentile.standard_error - analytical_se).abs() < 0.02);
    assert!((cmp.percentile.ci_lower - analytical_lo).abs() < 0.05);
    assert!((cmp.percentile.ci_upper - analytical_hi).abs() < 0.05);
    assert!((cmp.bca.ci_lower - analytical_lo).abs() < 0.05);
    assert!((cmp.bca.ci_upper - analytical_hi).abs() < 0.05);
}

#[test]
#[verified_engine::verified]
fn test_bca_reduces_skew_bias_and_improves_right_tail_coverage() {
    // Sample variance of Exp(1) has true value 1.0 and strong right-skew + downward bootstrap bias.
    // Across Monte Carlo trials, BCa detects positive z0 and positive acceleration a > 0, shifting
    // upper bounds upward and improving coverage of true variance = 1.0 over naive Percentile.
    let n = 35_usize;
    let trials = 60_usize;
    let mut ws = BootstrapWorkspace::new(n, 800).expect("workspace");
    let mut data = vec![0.0; n];

    let sample_var = |xs: &[f64]| -> f64 {
        let m = sample_mean(xs);
        xs.iter().map(|&v| (v - m) * (v - m)).sum::<f64>() / ((xs.len() - 1) as f64)
    };

    let mut pos_accel_count = 0_usize;
    let mut bca_upper_shift_sum = 0.0;
    let mut pct_covered = 0_usize;
    let mut bca_covered = 0_usize;

    for t in 0..trials {
        let mut rng = OxidizeRng::new(50_000 + t as u64);
        for slot in &mut data {
            let u = (((rng.next_u64() >> 11) as f64) + 0.5) / ((1_u64 << 53) as f64);
            *slot = -u.ln();
        }
        let cmp = ws
            .compare_intervals(&data, 800, 0.90, &mut rng, sample_var)
            .expect("comparison");
        if cmp.bca.acceleration > 0.0 {
            pos_accel_count += 1;
        }
        bca_upper_shift_sum += cmp.bca.ci_upper - cmp.percentile.ci_upper;
        if cmp.percentile.ci_lower <= 1.0 && 1.0 <= cmp.percentile.ci_upper {
            pct_covered += 1;
        }
        if cmp.bca.ci_lower <= 1.0 && 1.0 <= cmp.bca.ci_upper {
            bca_covered += 1;
        }
    }

    assert!(
        pos_accel_count * 2 > trials,
        "BCa should detect positive right-skew acceleration in majority of trials ({pos_accel_count}/{trials})"
    );
    assert!(
        bca_upper_shift_sum > 0.0,
        "BCa should shift right-skewed upper endpoint upward on average ({bca_upper_shift_sum})"
    );
    assert!(
        bca_covered >= pct_covered,
        "BCa coverage ({bca_covered}) should match or exceed Percentile ({pct_covered}) on skewed variance"
    );
}

#[test]
#[verified_engine::verified]
fn test_bootstrap_workspace_zero_allocations_with_median_scratch() {
    let data = [1.0, 2.5, 2.8, 3.1, 4.7, 8.2, 12.4, 19.0, 25.5];
    let mut ws = BootstrapWorkspace::new(data.len(), 500).expect("workspace");
    let mut rng = OxidizeRng::new(99);

    let (cmp, _) = VerifiedEngine::run_verified(|| {
        ws.compare_intervals_with_scratch(&data, 500, 0.95, &mut rng, sample_median_in_place)
            .expect("zero-alloc median bootstrap")
    });

    assert!((cmp.bca.point_estimate - 4.7).abs() < 1e-12);
    assert!(cmp.bca.ci_lower <= cmp.bca.point_estimate);
    assert!(cmp.bca.point_estimate <= cmp.bca.ci_upper);
    assert_eq!(ws.last_replicates().len(), 500);
}

// [cite:clinical_trials_statistics]
