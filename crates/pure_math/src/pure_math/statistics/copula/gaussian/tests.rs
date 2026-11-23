use super::*;
use crate::pure_math::analysis::integration::{ClenshawCurtis, Integrator};
use crate::pure_math::statistics::monte_carlo::{normal_cdf_fast, normal_inv_cdf_fast};
use approx::assert_relative_eq;

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
    assert_relative_eq!(joint.value(), expected, epsilon = 0.05);
}

#[test]
#[verified_engine::verified]
fn test_gaussian_copula_positive_correlation() {
    let rho_indep = Correlation::new(0.0).unwrap();
    let rho_pos = Correlation::new(0.7).unwrap();

    let copula_indep = GaussianCopula::bivariate(rho_indep).unwrap();
    let copula_pos = GaussianCopula::bivariate(rho_pos).unwrap();

    let u1 = Probability::new(0.8).unwrap();
    let u2 = Probability::new(0.8).unwrap();

    let joint_indep = copula_indep.cdf(&[u1, u2]).unwrap();
    let joint_pos = copula_pos.cdf(&[u1, u2]).unwrap();
    assert!(joint_pos.value() > joint_indep.value());
}

#[test]
#[verified_engine::verified]
fn test_gaussian_copula_negative_correlation() {
    let rho_indep = Correlation::new(0.0).unwrap();
    let rho_neg = Correlation::new(-0.5).unwrap();

    let copula_indep = GaussianCopula::bivariate(rho_indep).unwrap();
    let copula_neg = GaussianCopula::bivariate(rho_neg).unwrap();

    let u1 = Probability::new(0.9).unwrap();
    let u2 = Probability::new(0.9).unwrap();

    let joint_indep = copula_indep.cdf(&[u1, u2]).unwrap();
    let joint_neg = copula_neg.cdf(&[u1, u2]).unwrap();
    assert!(joint_neg.value() < joint_indep.value());
}

#[test]
#[verified_engine::verified]
fn test_sgp_joint_probability() {
    let p_a = Probability::new(0.99).unwrap();
    let p_b = Probability::new(0.60).unwrap();

    let rho = Correlation::new(-0.15).unwrap();
    let corr_matrix = CorrelationMatrix::bivariate(rho).unwrap();

    let joint = sgp_joint_probability(&[p_a, p_b], &corr_matrix).unwrap();
    let independent = p_a.value() * p_b.value();
    assert!(joint.value() < independent);
}

#[test]
#[verified_engine::verified]
fn test_copula_bounds() {
    let rho = Correlation::new(0.3).unwrap();
    let copula = GaussianCopula::bivariate(rho).unwrap();

    for &u1_val in &[0.1, 0.5, 0.9] {
        for &u2_val in &[0.1, 0.5, 0.9] {
            let u1 = Probability::new(u1_val).unwrap();
            let u2 = Probability::new(u2_val).unwrap();
            let joint = copula.cdf(&[u1, u2]).unwrap();
            assert!(joint.value() >= 0.0 && joint.value() <= 1.0);
        }
    }
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

    // 5D equicorrelated (rho = 0.35) via sgp_joint_probability
    let u5_raw = [0.7, 0.6, 0.8, 0.65, 0.75];
    let u5: Vec<Probability> = u5_raw
        .iter()
        .map(|&v| Probability::new(v).unwrap())
        .collect();
    let mat5 = equicorrelated_matrix(5, 0.35);
    let p5 = sgp_joint_probability(&u5, &mat5).unwrap();
    let exact5 = exact_equicorrelated_copula_cdf(&u5_raw, 0.35);
    assert!(
        (p5.value() - exact5).abs() < 2.5e-3,
        "5D got {} vs {}",
        p5.value(),
        exact5
    );

    // 10D equicorrelated (rho = 0.3)
    let u10_raw = [0.75; 10];
    let u10: Vec<Probability> = u10_raw
        .iter()
        .map(|&v| Probability::new(v).unwrap())
        .collect();
    let cop10 = GaussianCopula::new(equicorrelated_matrix(10, 0.3));
    let p10 = cop10.cdf(&u10).unwrap();
    let exact10 = exact_equicorrelated_copula_cdf(&u10_raw, 0.3);
    assert!(
        (p10.value() - exact10).abs() < 5e-3,
        "10D got {} vs {}",
        p10.value(),
        exact10
    );

    // Non-positive-definite 3x3 matrix (all off-diagonals -0.8 -> eigenvalue -0.6 < 0)
    let non_spd = equicorrelated_matrix(3, -0.8);
    let bad_cop = GaussianCopula::new(non_spd);
    assert_eq!(
        bad_cop.cdf(&u3).unwrap_err(),
        CopulaError::NotPositiveDefinite
    );
}

// [cite:clinical_trials_statistics]
