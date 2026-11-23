use super::transforms::{fast_norm_cdf, fast_norm_inv};
use super::{
    CopulaQmcWorkspace, Correlation, CorrelationMatrix, DEFAULT_COPULA_QMC_PAIRS, GaussianCopula,
    Probability, sgp_joint_probability,
};
use std::f64::consts::PI;
use verified_engine::engine::VerifiedEngine;

/// Exact 1D Gauss-Simpson integral for equicorrelated Gaussian copula with `rho >= 0`:
/// `C(u_1, ..., u_d; rho) = \int_{-\infty}^\infty \phi(z) \prod_i \Phi((x_i - \sqrt{\rho} z)/\sqrt{1-\rho}) dz`.
fn exact_equicorrelated_gaussian_copula(u_val: f64, d: usize, rho: f64) -> f64 {
    if rho.abs() <= 1e-12 {
        return u_val.powi(d as i32);
    }
    let x = fast_norm_inv(u_val);
    let sqrt_rho = rho.sqrt();
    let sqrt_one_minus_rho = (1.0 - rho).sqrt();
    let z_max = 9.0;
    let steps = 1200_usize; // even count for composite Simpson's rule
    let h = (2.0 * z_max) / (steps as f64);
    let inv_sqrt_2pi = 1.0 / (2.0 * PI).sqrt();

    let eval_integrand = |z: f64| -> f64 {
        let phi = inv_sqrt_2pi * (-0.5 * z * z).exp();
        let cond = fast_norm_cdf((x - sqrt_rho * z) / sqrt_one_minus_rho);
        phi * cond.powi(d as i32)
    };

    let mut acc = eval_integrand(-z_max) + eval_integrand(z_max);
    for i in 1..steps {
        let z = -z_max + (i as f64) * h;
        let weight = if i % 2 == 1 { 4.0 } else { 2.0 };
        acc += weight * eval_integrand(z);
    }
    acc * (h / 3.0)
}

#[test]
#[verified_engine::verified]
fn test_three_leg_parlay_and_tail_crash_benchmarks() {
    let corr = CorrelationMatrix::equicorrelated(3, Correlation::new(0.60).unwrap()).unwrap();
    let copula = GaussianCopula::new(corr.clone());

    // Benchmark 1: 3-leg SGP with u = 0.55 and rho = 0.60 -> ~0.3272 (naive = 0.166375)
    let u_parlay = [Probability::new(0.55).unwrap(); 3];
    let exact_parlay = exact_equicorrelated_gaussian_copula(0.55, 3, 0.60);
    let qmc_parlay = copula.cdf(&u_parlay).unwrap().value();
    assert!((exact_parlay - 0.3272).abs() < 2e-4);
    assert!(
        (qmc_parlay - exact_parlay).abs() <= 0.005,
        "expected ~{exact_parlay:.6}, got {qmc_parlay:.6}"
    );
    let sgp = sgp_joint_probability(&u_parlay, &corr).unwrap().value();
    assert!((sgp - qmc_parlay).abs() < 1e-12);

    // Benchmark 2: Simultaneous 5% tail crash with u = 0.05 and rho = 0.60 -> ~0.00767 (61.35x naive)
    let u_crash = [Probability::new(0.05).unwrap(); 3];
    let exact_crash = exact_equicorrelated_gaussian_copula(0.05, 3, 0.60);
    let qmc_crash = copula.cdf(&u_crash).unwrap().value();
    let naive_crash = 0.05_f64.powi(3); // 0.000125
    assert!((exact_crash - 0.007_668).abs() < 2e-5);
    assert!(
        (qmc_crash - exact_crash).abs() <= 0.001,
        "expected ~{exact_crash:.6}, got {qmc_crash:.6}"
    );
    let multiplier = qmc_crash / naive_crash;
    assert!(
        (multiplier - 61.35).abs() < 2.0,
        "expected ~61.35x tail crash amplification, got {multiplier:.2}x"
    );
}

#[test]
#[verified_engine::verified]
fn test_multivariate_qmc_accuracy_from_3_to_12_legs() {
    for &legs in &[3_usize, 4, 6, 8, 10, 12] {
        for &(u_val, rho) in &[(0.55, 0.60), (0.35, 0.45), (0.75, 0.30), (0.50, 0.0)] {
            let corr =
                CorrelationMatrix::equicorrelated(legs, Correlation::new(rho).unwrap()).unwrap();
            let copula = GaussianCopula::new(corr);
            let marginals = vec![Probability::new(u_val).unwrap(); legs];
            let estimate = copula.cdf(&marginals).unwrap().value();
            let exact = exact_equicorrelated_gaussian_copula(u_val, legs, rho);
            let abs_err = (estimate - exact).abs();
            assert!(
                abs_err <= 0.005,
                "legs={legs}, u={u_val}, rho={rho}: estimate={estimate:.6}, exact={exact:.6}, err={abs_err:.6}"
            );
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_multivariate_qmc_negative_equicorrelation_and_bounds() {
    // For d = 4, minimum valid equicorrelation is -1/(d-1) = -1/3; test rho = -0.20.
    let corr = CorrelationMatrix::equicorrelated(4, Correlation::new(-0.20).unwrap()).unwrap();
    let copula = GaussianCopula::new(corr);
    let marginals = [Probability::new(0.70).unwrap(); 4];
    let joint = copula.cdf(&marginals).unwrap().value();
    let independent = 0.70_f64.powi(4);
    assert!(joint > 0.0 && joint < independent);

    // Zero marginal forces joint CDF to zero.
    let with_zero = [
        Probability::new(0.70).unwrap(),
        Probability::new(0.0).unwrap(),
        Probability::new(0.80).unwrap(),
        Probability::new(0.90).unwrap(),
    ];
    assert_eq!(copula.cdf(&with_zero).unwrap().value(), 0.0);

    // Unit marginals reduce dimension or yield 1.0 when all equal 1.0.
    let all_ones = [Probability::new(1.0).unwrap(); 4];
    assert_eq!(copula.cdf(&all_ones).unwrap().value(), 1.0);
}

#[test]
#[verified_engine::verified]
fn test_multivariate_copula_zero_allocation_in_verified_engine() {
    let corr_12 = CorrelationMatrix::equicorrelated(12, Correlation::new(0.50).unwrap()).unwrap();
    let copula_12 = GaussianCopula::new(corr_12);
    let marginals_12 = [Probability::new(0.60).unwrap(); 12];

    // Stack-buffered fast paths for <= 12 legs allocate 0 bytes!
    let (res_inline, _) = VerifiedEngine::run_verified(|| copula_12.cdf(&marginals_12));
    let val_inline = res_inline.unwrap().value();
    let exact_12 = exact_equicorrelated_gaussian_copula(0.60, 12, 0.50);
    assert!((val_inline - exact_12).abs() <= 0.005);

    let (res_seed, _) = VerifiedEngine::run_verified(|| {
        copula_12.cdf_with_seed(
            &marginals_12,
            oxidize_core::rng::BASE_SEED,
            DEFAULT_COPULA_QMC_PAIRS,
        )
    });
    let val_seed = res_seed.unwrap().value();
    assert!((val_seed - exact_12).abs() <= 0.005);

    // Preallocated workspace path also allocates 0 bytes and matches stack-buffered Halton QMC!
    let mut workspace = CopulaQmcWorkspace::with_capacity(16);
    let (res_ws, _) = VerifiedEngine::run_verified(|| {
        copula_12.cdf_with_workspace(
            &marginals_12,
            oxidize_core::rng::BASE_SEED,
            DEFAULT_COPULA_QMC_PAIRS,
            &mut workspace,
        )
    });
    assert!((res_ws.unwrap().value() - val_seed).abs() < 1e-12);

    // Bivariate copula also remains zero-allocation.
    let copula_2 = GaussianCopula::bivariate(Correlation::new(0.4).unwrap()).unwrap();
    let u2 = [Probability::new(0.5).unwrap(); 2];
    let (res_2, _) = VerifiedEngine::run_verified(|| copula_2.cdf(&u2));
    assert!(res_2.unwrap().value() > 0.25);
}
