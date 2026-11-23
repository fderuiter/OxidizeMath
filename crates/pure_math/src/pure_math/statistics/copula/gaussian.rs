//! Gaussian copula implementation with stack-buffered, seeded QMC, and caller-scratch evaluation.

use super::core::{Correlation, CorrelationMatrix, Probability};
use super::multivariate::{
    evaluate_copula_cdf_with_rng_and_scratch, evaluate_copula_cdf_with_scratch,
    required_scratch_len,
};
use crate::error::CopulaError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use oxidize_core::rng::OxidizeRng;

/// Maximum number of parlay legs evaluated entirely on the stack by [`GaussianCopula::cdf`]
/// and [`sgp_joint_probability`].
pub const MAX_STACK_LEGS: usize = 16;

/// Stack scratch length corresponding to [`MAX_STACK_LEGS`].
const MAX_STACK_SCRATCH: usize = required_scratch_len(MAX_STACK_LEGS);

/// Evaluates `C(u; corr)` using stack scratch up to [`MAX_STACK_LEGS`] and heap fallback above.
#[verified_engine::verified]
pub(crate) fn evaluate_copula_cdf(
    u: &[Probability],
    corr: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    let d = corr.dimension();
    if u.len() != d {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(d),
            actual: Dimension(u.len()),
        }));
    }
    if d <= MAX_STACK_LEGS {
        let mut stack_scratch = [0.0_f64; MAX_STACK_SCRATCH];
        return evaluate_copula_cdf_with_scratch(
            u,
            corr,
            &mut stack_scratch[..required_scratch_len(d)],
        );
    }
    let mut heap_scratch = vec![0.0_f64; required_scratch_len(d)];
    evaluate_copula_cdf_with_scratch(u, corr, &mut heap_scratch)
}

/// Multivariate Gaussian copula `C(u_1, ..., u_d; Sigma) = Phi_Sigma(Phi^-1(u_1), ..., Phi^-1(u_d))`.
///
/// Uses exact copula boundary reduction, 20-point Genz (2004) Gauss-Legendre quadrature for two
/// active legs, and Genz (1992) separation-of-variables over a deterministic or Cranley-Patterson
/// shifted Richtmyer QMC sequence for three or more active legs.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     Correlation, GaussianCopula, Probability,
/// };
///
/// let rho = Correlation::new(-0.3).unwrap();
/// let copula = GaussianCopula::bivariate(rho).unwrap();
///
/// let u1 = Probability::new(0.01).unwrap();
/// let u2 = Probability::new(0.60).unwrap();
/// let joint_prob = copula.cdf(&[u1, u2]).unwrap();
/// assert!(joint_prob.value() < 0.01 * 0.60);
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

    /// Creates a bivariate Gaussian copula from a single correlation coefficient.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::{Correlation, GaussianCopula};
    ///
    /// let rho = Correlation::new(0.5).unwrap();
    /// let copula = GaussianCopula::bivariate(rho).unwrap();
    /// assert_eq!(copula.correlation().dimension(), 2);
    /// ```
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let correlation = CorrelationMatrix::bivariate(rho)?;
        Ok(Self::new(correlation))
    }

    /// Returns the scratch slice length in `f64` elements required by [`Self::cdf_with_scratch`]
    /// for a copula of `dimension` legs.
    #[verified_engine::verified]
    pub const fn scratch_len(dimension: usize) -> usize {
        required_scratch_len(dimension)
    }

    /// Computes the copula CDF `C(u_1, ..., u_d)` using a stack buffer for up to
    /// [`MAX_STACK_LEGS`] (`16`) legs without heap allocation.
    #[verified_engine::verified]
    pub fn cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        evaluate_copula_cdf(u, &self.correlation)
    }

    /// Computes `C(u_1, ..., u_d)` using Cranley-Patterson `rng` shifts and `n_qmc_pairs`
    /// antithetic QMC pairs for 3+ active legs (zero heap allocation up to [`MAX_STACK_LEGS`] legs).
    #[verified_engine::verified]
    pub fn cdf_with_rng(
        &self,
        u: &[Probability],
        rng: &mut OxidizeRng,
        n_qmc_pairs: usize,
    ) -> Result<Probability, CopulaError> {
        let d = self.correlation.dimension();
        if u.len() != d {
            return Err(CopulaError::Math(MathError::DimensionMismatch {
                expected: Dimension(d),
                actual: Dimension(u.len()),
            }));
        }
        if d <= MAX_STACK_LEGS {
            let mut stack_scratch = [0.0_f64; MAX_STACK_SCRATCH];
            let mut stack_shifts = [0.0_f64; MAX_STACK_LEGS];
            return evaluate_copula_cdf_with_rng_and_scratch(
                u,
                &self.correlation,
                rng,
                n_qmc_pairs,
                &mut stack_scratch[..required_scratch_len(d)],
                &mut stack_shifts[..d],
            );
        }
        let mut heap_scratch = vec![0.0_f64; required_scratch_len(d)];
        let mut heap_shifts = vec![0.0_f64; d];
        evaluate_copula_cdf_with_rng_and_scratch(
            u,
            &self.correlation,
            rng,
            n_qmc_pairs,
            &mut heap_scratch,
            &mut heap_shifts,
        )
    }

    /// Computes the copula CDF `C(u_1, ..., u_d)` for arbitrary leg counts with zero heap
    /// allocations using caller-supplied `scratch` (`scratch.len() >= Self::scratch_len(d)`).
    #[verified_engine::verified]
    pub fn cdf_with_scratch(
        &self,
        u: &[Probability],
        scratch: &mut [f64],
    ) -> Result<Probability, CopulaError> {
        evaluate_copula_cdf_with_scratch(u, &self.correlation, scratch)
    }

    /// Returns the underlying correlation matrix.
    #[verified_engine::verified]
    pub fn correlation(&self) -> &CorrelationMatrix {
        &self.correlation
    }
}

/// Computes the correlated joint win probability for a Same Game Parlay (SGP) without cloning
/// `correlation_matrix` (zero heap allocation up to [`MAX_STACK_LEGS`] legs).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     Correlation, CorrelationMatrix, Probability, sgp_joint_probability,
/// };
///
/// let p_a = Probability::new(0.01).unwrap();
/// let p_b = Probability::new(0.60).unwrap();
/// let corr_matrix = CorrelationMatrix::bivariate(Correlation::new(-0.15).unwrap()).unwrap();
/// let joint_prob = sgp_joint_probability(&[p_a, p_b], &corr_matrix).unwrap();
/// assert!(joint_prob.value() < 0.006);
/// ```
#[verified_engine::verified]
pub fn sgp_joint_probability(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    evaluate_copula_cdf(marginals, correlation_matrix)
}

/// Computes the correlated joint win probability for an arbitrary-leg Same Game Parlay (SGP)
/// with zero heap allocations using caller-supplied `scratch`.
#[verified_engine::verified]
pub fn sgp_joint_probability_with_scratch(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
    scratch: &mut [f64],
) -> Result<Probability, CopulaError> {
    evaluate_copula_cdf_with_scratch(marginals, correlation_matrix, scratch)
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

    use super::*;
    use approx::assert_relative_eq;
    use math_commons::registry::TOLERANCE_HIGH;
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
            epsilon = TOLERANCE_HIGH
        );

        let deep_u = Probability::new(1e-25).unwrap();
        let deep_indep = copula_indep.cdf(&[deep_u, u2]).unwrap().value();
        assert_relative_eq!(deep_indep, 0.8e-25, max_relative = 1e-6);

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
    fn test_boundary_axioms_comonotonic_and_sgp_reduction() {
        let one_d = GaussianCopula::new(CorrelationMatrix::new(DMatrix::identity(1, 1)).unwrap());
        let u = Probability::new(0.37).unwrap();
        assert_eq!(one_d.cdf(&[u]).unwrap().value(), 0.37);

        let pos = GaussianCopula::bivariate(Correlation::new(1.0).unwrap()).unwrap();
        let neg = GaussianCopula::bivariate(Correlation::new(-1.0).unwrap()).unwrap();
        let u1 = Probability::new(0.35).unwrap();
        let u2 = Probability::new(0.80).unwrap();
        assert_relative_eq!(pos.cdf(&[u1, u2]).unwrap().value(), 0.35, epsilon = 1e-15);
        assert_relative_eq!(neg.cdf(&[u1, u2]).unwrap().value(), 0.15, epsilon = 1e-15);

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

    #[test]
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

        let invalid =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0]);
        assert!(matches!(
            CorrelationMatrix::new(invalid),
            Err(CopulaError::NotPositiveDefinite)
        ));
    }
}
