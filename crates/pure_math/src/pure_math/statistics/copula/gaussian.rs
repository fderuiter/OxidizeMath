//! Gaussian copula implementation across bivariate and multivariate dimensions.

use super::core::{Correlation, CorrelationMatrix, Probability};
use super::qmc::{bivariate_standard_normal_cdf, genz_qmc_copula_cdf};
use crate::error::CopulaError;
use crate::pure_math::statistics::distributions::special::standard_normal_quantile;
use oxidize_core::rng::OxidizeRng;

const DEFAULT_QMC_SEED: u64 = 0x5EED_C090_1A55_2026;
const DEFAULT_QMC_PAIRS: usize = 1024;

/// Multivariate Gaussian copula $C_R(u_1, \dots, u_d) = \Phi_R(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_d))$.
///
/// Evaluates bivariate copulas via 16-point Gauss-Legendre quadrature over the smooth trigonometric
/// Plackett integral and dimensions $3 \le d \le 16$ via Genz (1992) Cholesky separation-of-variables
/// Quasi-Monte Carlo with zero heap allocations.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     GaussianCopula, Probability, Correlation
/// };
///
/// let rho = Correlation::new(-0.3).unwrap();
/// let copula = GaussianCopula::bivariate(rho).unwrap();
///
/// let u1 = Probability::new(0.99).unwrap();
/// let u2 = Probability::new(0.60).unwrap();
///
/// let joint_prob = copula.cdf(&[u1, u2]).unwrap();
/// assert!(joint_prob.value() < 0.99 * 0.60);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GaussianCopula {
    correlation: CorrelationMatrix,
}

impl GaussianCopula {
    /// Creates a new Gaussian copula from a validated [`CorrelationMatrix`].
    #[verified_engine::verified]
    pub fn new(correlation: CorrelationMatrix) -> Self {
        Self { correlation }
    }

    /// Creates a bivariate Gaussian copula with correlation `rho`.
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let correlation = CorrelationMatrix::bivariate(rho)?;
        Ok(Self::new(correlation))
    }

    /// Computes the joint cumulative probability $C_R(u_1, \dots, u_d)$ deterministically.
    #[verified_engine::verified]
    pub fn cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        let mut rng = OxidizeRng::from_seed(DEFAULT_QMC_SEED);
        self.cdf_with_rng(u, &mut rng, DEFAULT_QMC_PAIRS)
    }

    /// Computes $C_R(u_1, \dots, u_d)$ using explicit `rng` shifts and `n_qmc_pairs` antithetic QMC pairs
    /// when $d \ge 3$ (and exact 16-point Gauss-Legendre quadrature when $d \le 2$).
    #[verified_engine::verified]
    pub fn cdf_with_rng(
        &self,
        u: &[Probability],
        rng: &mut OxidizeRng,
        n_qmc_pairs: usize,
    ) -> Result<Probability, CopulaError> {
        let d = self.correlation.dimension();
        if u.len() != d {
            return Err(CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(d),
                    actual: math_commons::math_kernel::types::Dimension(u.len()),
                },
            ));
        }
        if d > 16 {
            return Err(CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(16),
                    actual: math_commons::math_kernel::types::Dimension(d),
                },
            ));
        }
        if u.iter().any(|p| p.value() <= 0.0) {
            return Probability::new(0.0);
        }
        if d == 1 {
            return Ok(u[0]);
        }
        if d == 2 {
            return self.bivariate_cdf(u[0], u[1]);
        }
        self.multivariate_cdf_qmc(u, rng, n_qmc_pairs)
    }

    #[verified_engine::verified]
    fn bivariate_cdf(&self, u1: Probability, u2: Probability) -> Result<Probability, CopulaError> {
        let p1 = u1.value();
        let p2 = u2.value();
        if p1 <= 0.0 || p2 <= 0.0 {
            return Probability::new(0.0);
        }
        if p1 >= 1.0 {
            return Ok(u2);
        }
        if p2 >= 1.0 {
            return Ok(u1);
        }
        let z1 = standard_normal_quantile(p1);
        let z2 = standard_normal_quantile(p2);
        let rho = self.correlation.matrix()[(0, 1)];
        Probability::new(bivariate_standard_normal_cdf(z1, z2, rho))
    }

    #[verified_engine::verified]
    fn multivariate_cdf_qmc(
        &self,
        u: &[Probability],
        rng: &mut OxidizeRng,
        n_qmc_pairs: usize,
    ) -> Result<Probability, CopulaError> {
        let mut active_indices = [0_usize; 16];
        let mut active_count = 0_usize;
        for (idx, p) in u.iter().enumerate() {
            if p.value() < 1.0 {
                active_indices[active_count] = idx;
                active_count += 1;
            }
        }
        if active_count == 0 {
            return Probability::new(1.0);
        }
        if active_count == 1 {
            return Ok(u[active_indices[0]]);
        }
        if active_count == 2 {
            let i0 = active_indices[0];
            let i1 = active_indices[1];
            let z1 = standard_normal_quantile(u[i0].value());
            let z2 = standard_normal_quantile(u[i1].value());
            let rho = self.correlation.matrix()[(i0, i1)];
            return Probability::new(bivariate_standard_normal_cdf(z1, z2, rho));
        }

        let d = u.len();
        let mut raw_u = [0.0_f64; 16];
        for i in 0..d {
            raw_u[i] = u[i].value();
        }
        let val = genz_qmc_copula_cdf(
            self.correlation.cholesky_factor(),
            &raw_u[..d],
            rng,
            n_qmc_pairs,
        );
        Probability::new(val)
    }

    /// Returns the correlation matrix.
    #[verified_engine::verified]
    pub fn correlation(&self) -> &CorrelationMatrix {
        &self.correlation
    }
}

/// Computes the joint probability for a Same Game Parlay (SGP) across 2 or more correlated legs.
#[verified_engine::verified]
pub fn sgp_joint_probability(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    if marginals.len() != correlation_matrix.dimension() {
        return Err(CopulaError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(
                    correlation_matrix.dimension(),
                ),
                actual: math_commons::math_kernel::types::Dimension(marginals.len()),
            },
        ));
    }
    let copula = GaussianCopula::new(correlation_matrix.clone());
    copula.cdf(marginals)
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DMatrix;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_gaussian_copula_bivariate_properties_and_deep_tails() {
        let rho_zero = Correlation::new(0.0).unwrap();
        let copula_indep = GaussianCopula::bivariate(rho_zero).unwrap();
        let u1 = Probability::new(0.7).unwrap();
        let u2 = Probability::new(0.8).unwrap();
        assert_relative_eq!(
            copula_indep.cdf(&[u1, u2]).unwrap().value(),
            0.56,
            epsilon = 1e-12
        );

        // Deep tail where legacy -10.0 truncated grid erroneously returned 0.0:
        // u = 1e-25 corresponds to z ≈ -10.42 < -10.0!
        let deep_u = Probability::new(1e-25).unwrap();
        let deep_indep = copula_indep.cdf(&[deep_u, u2]).unwrap().value();
        assert_relative_eq!(deep_indep, 0.8e-25, max_relative = 1e-6);

        // Boundary conditions C(u, 0) = 0 and C(u, 1) = u
        let zero = Probability::new(0.0).unwrap();
        let one = Probability::new(1.0).unwrap();
        let copula_pos = GaussianCopula::bivariate(Correlation::new(0.7).unwrap()).unwrap();
        assert_eq!(copula_pos.cdf(&[u1, zero]).unwrap().value(), 0.0);
        assert_relative_eq!(
            copula_pos.cdf(&[u1, one]).unwrap().value(),
            0.7,
            epsilon = 1e-14
        );
        assert!(copula_pos.cdf(&[u1, u2]).unwrap().value() > 0.56);

        let copula_neg = GaussianCopula::bivariate(Correlation::new(-0.5).unwrap()).unwrap();
        assert!(copula_neg.cdf(&[u1, u2]).unwrap().value() < 0.56);
    }

    #[test]
    fn test_multivariate_qmc_dimensions_2_through_5_frechet_hoeffding() {
        for dim in 2..=5 {
            // Independence: C_I(u) = prod u_i
            let id = CorrelationMatrix::new(DMatrix::identity(dim, dim)).unwrap();
            let copula_id = GaussianCopula::new(id);
            let probs: Vec<Probability> = (0..dim)
                .map(|i| Probability::new(0.55 + 0.07 * (i as f64)).unwrap())
                .collect();
            let expected_prod: f64 = probs.iter().map(Probability::value).product();
            let est_id = copula_id.cdf(&probs).unwrap().value();
            assert!(
                (est_id - expected_prod).abs() < 2e-3,
                "dim={dim}: independence QMC {est_id} vs {expected_prod}"
            );

            // Equicorrelated positive & mild negative matrices satisfy Fréchet-Hoeffding bounds
            for &r in &[-0.18, 0.45, 0.8] {
                let corr =
                    CorrelationMatrix::equicorrelated(dim, Correlation::new(r).unwrap()).unwrap();
                let copula = GaussianCopula::new(corr);
                let c_val = copula.cdf(&probs).unwrap().value();
                let fh_lower = (probs.iter().map(Probability::value).sum::<f64>() - (dim as f64)
                    + 1.0)
                    .max(0.0);
                let fh_upper = probs.iter().map(Probability::value).fold(1.0_f64, f64::min);
                assert!(
                    c_val >= fh_lower - 1e-12 && c_val <= fh_upper + 1e-12,
                    "dim={dim}, rho={r}: {c_val} violated [{fh_lower}, {fh_upper}]"
                );
            }
        }
    }

    #[test]
    fn test_copula_cdf_zero_allocation_under_verified_engine() {
        let corr = CorrelationMatrix::equicorrelated(4, Correlation::new(0.4).unwrap()).unwrap();
        let copula = GaussianCopula::new(corr);
        let u = [
            Probability::new(0.6).unwrap(),
            Probability::new(0.7).unwrap(),
            Probability::new(0.5).unwrap(),
            Probability::new(0.8).unwrap(),
        ];
        let mut rng = OxidizeRng::from_seed(123);
        let (res, _) = VerifiedEngine::run_verified(|| copula.cdf_with_rng(&u, &mut rng, 128));
        assert!(res.unwrap().value() > 0.0);
    }
}
