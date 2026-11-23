//! Gaussian copula implementation with stack-buffered and caller-scratch evaluation.

use super::core::{Correlation, CorrelationMatrix, Probability};
use super::multivariate::{evaluate_copula_cdf_with_scratch, required_scratch_len};
use crate::error::CopulaError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;

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
/// active legs, and Genz (1992) separation-of-variables over a deterministic Richtmyer QMC
/// sequence for three or more active legs.
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
    use super::*;
    use approx::assert_relative_eq;
    use math_commons::registry::TOLERANCE_HIGH;
    use nalgebra::DMatrix;

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
}
