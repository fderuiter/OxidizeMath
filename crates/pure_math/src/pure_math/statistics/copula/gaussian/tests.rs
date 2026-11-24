use super::*;
use crate::pure_math::analysis::integration::{ClenshawCurtis, Integrator};
use crate::pure_math::statistics::monte_carlo::{normal_cdf_fast, normal_inv_cdf_fast};
use approx::assert_relative_eq;
use math_commons::registry::TOLERANCE_HIGH;
use verified_engine::engine::VerifiedEngine;

/// Exact 1D factor integral for equicorrelated Gaussian copula ($\rho \in [0, 1)$):
/// $\Phi_\Sigma(b) = \int_{-\infty}^\infty \phi(z) \prod_{i=1}^d \Phi((b_i - \sqrt{\rho} z)/\sqrt{1 - \rho}) dz$.
fn exact_equicorrelated_copula_cdf(u: &[f64], rho: f64) -> f64 {
    let zs: Vec<f64> = u.iter().map(|&ui| normal_inv_cdf_fast(ui)).collect();
    let s_rho = rho.sqrt();
    let s_rem = (1.0 - rho).sqrt();
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let quad = ClenshawCurtis;
    quad.integrate(
        |z| {
            let phi = inv_sqrt_2pi * (-0.5 * z * z).exp();
            let mut prod = 1.0;
            for &b in &zs {
                prod *= normal_cdf_fast((b - s_rho * z) / s_rem);
            }
            phi * prod
        },
        -9.0,
        9.0,
        1e-11,
    )
    .value
}

fn equicorrelated_matrix(dim: usize, rho: f64) -> CorrelationMatrix {
    let mat = DMatrix::from_fn(dim, dim, |i, j| if i == j { 1.0 } else { rho });
    CorrelationMatrix::new(mat).expect("valid equicorrelated matrix")
}

#[test]
#[verified_engine::verified]
fn test_gaussian_copula_creation() {
    let rho = Correlation::new(0.5).unwrap();
    let copula = GaussianCopula::bivariate(rho).unwrap();
    assert_eq!(copula.correlation().dimension(), 2);
}

#[test]
#[verified_engine::verified]
fn test_gaussian_copula_independence() {
    let rho = Correlation::new(0.0).unwrap();
    let copula = GaussianCopula::bivariate(rho).unwrap();

    let u1 = Probability::new(0.7).unwrap();
    let u2 = Probability::new(0.8).unwrap();
    let joint = copula.cdf(&[u1, u2]).unwrap();

    let expected = u1.value() * u2.value();
    assert_relative_eq!(joint.value(), expected, epsilon = TOLERANCE_HIGH);
}

#[test]
#[verified_engine::verified]
fn test_gaussian_copula_positive_and_negative_correlation() {
    let copula_indep = GaussianCopula::bivariate(Correlation::new(0.0).unwrap()).unwrap();
    let copula_pos = GaussianCopula::bivariate(Correlation::new(0.7).unwrap()).unwrap();
    let copula_neg = GaussianCopula::bivariate(Correlation::new(-0.5).unwrap()).unwrap();

    let u1 = Probability::new(0.8).unwrap();
    let u2 = Probability::new(0.8).unwrap();
    let joint_indep = copula_indep.cdf(&[u1, u2]).unwrap().value();
    assert!(copula_pos.cdf(&[u1, u2]).unwrap().value() > joint_indep);
    assert!(copula_neg.cdf(&[u1, u2]).unwrap().value() < joint_indep);
}

#[test]
#[verified_engine::verified]
fn test_boundary_axioms_and_single_leg_identity() {
    let one_d = GaussianCopula::new(CorrelationMatrix::new(DMatrix::identity(1, 1)).unwrap());
    let u = Probability::new(0.37).unwrap();
    assert_eq!(one_d.cdf(&[u]).unwrap().value(), 0.37);

    let copula = GaussianCopula::bivariate(Correlation::new(-0.42).unwrap()).unwrap();
    let p0 = Probability::new(0.0).unwrap();
    let p1 = Probability::new(1.0).unwrap();
    assert_eq!(copula.cdf(&[p0, p0]).unwrap().value(), 0.0);
    assert_eq!(copula.cdf(&[p0, u]).unwrap().value(), 0.0);
    assert_eq!(copula.cdf(&[u, p0]).unwrap().value(), 0.0);
    assert_eq!(copula.cdf(&[p1, u]).unwrap().value(), 0.37);
    assert_eq!(copula.cdf(&[u, p1]).unwrap().value(), 0.37);
    assert_eq!(copula.cdf(&[p1, p1]).unwrap().value(), 1.0);
}

#[test]
#[verified_engine::verified]
fn test_comonotonic_and_countermonotonic_frechet_bounds() {
    let pos = GaussianCopula::bivariate(Correlation::new(1.0).unwrap()).unwrap();
    let neg = GaussianCopula::bivariate(Correlation::new(-1.0).unwrap()).unwrap();
    let u1 = Probability::new(0.35).unwrap();
    let u2 = Probability::new(0.80).unwrap();

    assert_relative_eq!(pos.cdf(&[u1, u2]).unwrap().value(), 0.35, epsilon = 1e-15);
    assert_relative_eq!(neg.cdf(&[u1, u2]).unwrap().value(), 0.15, epsilon = 1e-15);
    assert_relative_eq!(
        pos.multivariate_cdf(&[u1, u2]).unwrap().value(),
        0.35,
        epsilon = 1e-15
    );
    assert_relative_eq!(
        neg.multivariate_cdf(&[u1, u2]).unwrap().value(),
        0.15,
        epsilon = 1e-15
    );
}

#[test]
#[verified_engine::verified]
fn test_sgp_joint_probability_and_multileg_reduction() {
    let p_a = Probability::new(0.01).unwrap();
    let p_b = Probability::new(0.60).unwrap();
    let corr2 = CorrelationMatrix::bivariate(Correlation::new(-0.15).unwrap()).unwrap();
    let joint2 = sgp_joint_probability(&[p_a, p_b], &corr2).unwrap();
    assert!(joint2.value() < p_a.value() * p_b.value());

    let corr3 = CorrelationMatrix::new(DMatrix::from_row_slice(
        3,
        3,
        &[1.0, -0.15, 0.25, -0.15, 1.0, 0.30, 0.25, 0.30, 1.0],
    ))
    .unwrap();
    let reduced =
        sgp_joint_probability(&[p_a, p_b, Probability::new(1.0).unwrap()], &corr3).unwrap();
    assert_relative_eq!(reduced.value(), joint2.value(), epsilon = 1e-15);
}

#[test]
#[verified_engine::verified]
fn test_multivariate_cholesky_validation() {
    let mat = DMatrix::from_row_slice(3, 3, &[1.0, 0.2, 0.1, 0.2, 1.0, 0.3, 0.1, 0.3, 1.0]);
    let corr = CorrelationMatrix::new(mat).unwrap();
    let copula = GaussianCopula::new(corr);
    let u = [
        Probability::new(0.4).unwrap(),
        Probability::new(0.5).unwrap(),
        Probability::new(0.6).unwrap(),
    ];
    let joint = copula.cdf(&u).unwrap().value();
    assert!(joint > 0.4 * 0.5 * 0.6 && joint <= 0.4);

    let invalid = DMatrix::from_row_slice(3, 3, &[1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0]);
    assert!(matches!(
        CorrelationMatrix::new(invalid),
        Err(CopulaError::NotPositiveDefinite)
    ));
}

#[test]
#[verified_engine::verified]
fn test_multivariate_gaussian_copula_3d_5d_10d_accuracy() {
    // 3D equicorrelated (rho = 0.4)
    let u3_raw = [0.65, 0.75, 0.55];
    let u3: Vec<Probability> = u3_raw
        .iter()
        .map(|&v| Probability::new(v).unwrap())
        .collect();
    let cop3 = GaussianCopula::new(equicorrelated_matrix(3, 0.4));
    let (p3, se3) = cop3.multivariate_cdf_with_error(&u3, 2048, 42).unwrap();
    let exact3 = exact_equicorrelated_copula_cdf(&u3_raw, 0.4);
    assert!(
        (p3.value() - exact3).abs() < 1.5e-3,
        "3D got {} vs {}",
        p3.value(),
        exact3
    );
    assert!((0.0..2e-3).contains(&se3));

    // 5D equicorrelated (rho = 0.35) via both multivariate_cdf and sgp_joint_probability
    let u5_raw = [0.7, 0.6, 0.8, 0.65, 0.75];
    let u5: Vec<Probability> = u5_raw
        .iter()
        .map(|&v| Probability::new(v).unwrap())
        .collect();
    let mat5 = equicorrelated_matrix(5, 0.35);
    let p5_qmc = GaussianCopula::new(mat5.clone())
        .multivariate_cdf(&u5)
        .unwrap();
    let p5 = sgp_joint_probability(&u5, &mat5).unwrap();
    let exact5 = exact_equicorrelated_copula_cdf(&u5_raw, 0.35);
    assert!((p5_qmc.value() - exact5).abs() < 2.5e-3);
    assert!((p5.value() - exact5).abs() < 2.5e-3);

    // 10D equicorrelated (rho = 0.3)
    let u10_raw = [0.75; 10];
    let u10: Vec<Probability> = u10_raw
        .iter()
        .map(|&v| Probability::new(v).unwrap())
        .collect();
    let cop10 = GaussianCopula::new(equicorrelated_matrix(10, 0.3));
    let p10_qmc = cop10.multivariate_cdf(&u10).unwrap();
    let p10 = cop10.cdf(&u10).unwrap();
    let exact10 = exact_equicorrelated_copula_cdf(&u10_raw, 0.3);
    assert!((p10_qmc.value() - exact10).abs() < 5e-3);
    assert!((p10.value() - exact10).abs() < 5e-3);

    // Non-positive-definite 3x3 matrix (all off-diagonals -0.8 -> eigenvalue -0.6 < 0)
    let non_spd = DMatrix::from_fn(3, 3, |i, j| if i == j { 1.0 } else { -0.8 });
    assert_eq!(
        CorrelationMatrix::new(non_spd),
        Err(CopulaError::NotPositiveDefinite)
    );
}

#[test]
#[verified_engine::verified]
fn test_deep_tail_frechet_hoeffding_and_zero_alloc_cdf_with_rng() {
    let copula_indep = GaussianCopula::bivariate(Correlation::new(0.0).unwrap()).unwrap();
    let deep_u = Probability::new(1e-25).unwrap();
    let u2 = Probability::new(0.8).unwrap();
    assert_relative_eq!(
        copula_indep.cdf(&[deep_u, u2]).unwrap().value(),
        0.8e-25,
        max_relative = 1e-6
    );

    for dim in 2..=5 {
        let probs: Vec<Probability> = (0..dim)
            .map(|i| Probability::new(0.55 + 0.07 * (i as f64)).unwrap())
            .collect();
        for &r in &[-0.18, 0.45, 0.8] {
            let corr =
                CorrelationMatrix::equicorrelated(dim, Correlation::new(r).unwrap()).unwrap();
            let c_val = GaussianCopula::new(corr).cdf(&probs).unwrap().value();
            let fh_lower =
                (probs.iter().map(Probability::value).sum::<f64>() - (dim as f64) + 1.0).max(0.0);
            let fh_upper = probs.iter().map(Probability::value).fold(1.0_f64, f64::min);
            assert!(c_val >= fh_lower - 1e-12 && c_val <= fh_upper + 1e-12);
        }
    }

    let corr4 = CorrelationMatrix::equicorrelated(4, Correlation::new(0.4).unwrap()).unwrap();
    let copula4 = GaussianCopula::new(corr4);
    let u4 = [
        Probability::new(0.6).unwrap(),
        Probability::new(0.7).unwrap(),
        Probability::new(0.5).unwrap(),
        Probability::new(0.8).unwrap(),
    ];
    let mut rng = OxidizeRng::from_seed(123);
    let (res, _) = VerifiedEngine::run_verified(|| copula4.cdf_with_rng(&u4, &mut rng, 128));
    assert!(res.unwrap().value() > 0.0);
}

// [cite:clinical_trials_statistics]
