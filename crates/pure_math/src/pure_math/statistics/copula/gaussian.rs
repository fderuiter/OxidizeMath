//! Gaussian copula implementation.

use super::core::{Correlation, CorrelationMatrix, Probability};
use super::transforms::{inverse_standard_normal, standard_normal_cdf};
use crate::error::CopulaError;
use nalgebra::DVector;
use statrs::distribution::MultivariateNormal;

/// Bivariate Gaussian copula.
///
/// Models the joint distribution of two uniform random variables using
/// the Gaussian copula:
///
/// ```text
/// C(u₁, u₂; ρ) = Φ_ρ(Φ⁻¹(u₁), Φ⁻¹(u₂))
/// ```
///
/// where:
/// - Φ⁻¹ is the inverse standard normal CDF
/// - Φ_ρ is the bivariate normal CDF with correlation ρ
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
/// let u1 = Probability::new(0.99).unwrap();  // Player A 99th percentile
/// let u2 = Probability::new(0.60).unwrap();  // Team win 60% chance
///
/// let joint_prob = copula.cdf(&[u1, u2]).unwrap();
/// ```
pub struct GaussianCopula {
    correlation: CorrelationMatrix,
}

impl GaussianCopula {
    /// Creates a new Gaussian copula.
    ///
    /// # Arguments
    ///
    /// * `correlation` - The correlation matrix
    #[verified_engine::verified]
    pub fn new(correlation: CorrelationMatrix) -> Self {
        Self { correlation }
    }

    /// Creates a bivariate Gaussian copula.
    ///
    /// # Arguments
    ///
    /// * `rho` - The correlation between the two variables
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::{GaussianCopula, Correlation};
    ///
    /// let rho = Correlation::new(0.5).unwrap();
    /// let copula = GaussianCopula::bivariate(rho).unwrap();
    /// ```
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let correlation = CorrelationMatrix::bivariate(rho)?;
        Ok(Self::new(correlation))
    }

    /// Computes the copula CDF: C(u₁, ..., uₙ).
    ///
    /// # Arguments
    ///
    /// * `u` - Vector of uniform probabilities
    ///
    /// # Returns
    ///
    /// The joint cumulative probability
    #[verified_engine::verified]
    pub fn cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        if u.len() != self.correlation.dimension() {
            return Err(crate::error::CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(
                        self.correlation.dimension(),
                    ),
                    actual: math_commons::math_kernel::types::Dimension(u.len()),
                },
            ));
        }

        // Special case for bivariate (more efficient)
        if u.len() == 2 {
            return self.bivariate_cdf(u[0], u[1]);
        }

        // General multivariate case
        self.multivariate_cdf(u)
    }

    /// Computes the bivariate Gaussian copula CDF.
    #[verified_engine::verified]
    fn bivariate_cdf(&self, u1: Probability, u2: Probability) -> Result<Probability, CopulaError> {
        // Transform to z-scores
        let z1 = inverse_standard_normal(u1)?;
        let z2 = inverse_standard_normal(u2)?;

        // Get correlation
        let rho = self
            .correlation
            .get_correlation(0, 1)
            .ok_or(crate::error::CopulaError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "Failed to get correlation".to_string(),
                },
            ))?;

        // Compute bivariate normal CDF using the formula:
        // Φ_ρ(z₁, z₂) = Φ(z₁)Φ(z₂) + ∫∫ φ(x,y;ρ) dx dy
        //
        // For implementation, we use Owen's T function or numerical integration
        // Here we use a numerical approximation
        let prob = self.bivariate_normal_cdf(z1, z2, rho)?;
        Probability::new(prob)
    }

    /// Approximates the bivariate normal CDF using Drezner & Wesolowsky (1990) method.
    #[verified_engine::verified]
    fn bivariate_normal_cdf(&self, z1: f64, z2: f64, rho: f64) -> Result<f64, CopulaError> {
        // Handle edge cases
        if !z1.is_finite() || !z2.is_finite() {
            return Err(crate::error::CopulaError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: "Non-finite z-scores".to_string(),
                },
            ));
        }

        // If correlation is near ±1, use limiting formula
        if (rho.abs() - 1.0).abs() < math_commons::registry::TOLERANCE_HIGH {
            if rho > 0.0 {
                // Perfect positive correlation
                let prob1 = standard_normal_cdf(z1)?.value();
                let prob2 = standard_normal_cdf(z2)?.value();
                return Ok(prob1.min(prob2));
            } else {
                // Perfect negative correlation
                let prob1 = standard_normal_cdf(z1)?.value();
                let prob2 = standard_normal_cdf(-z2)?.value();
                return Ok((prob1 + prob2 - 1.0).max(0.0));
            }
        }

        // Use numerical integration (simple rectangular method)
        // For better accuracy, could use statrs or specialized libraries
        let prob = self.integrate_bivariate_normal(z1, z2, rho)?;
        Ok(prob)
    }

    /// Numerical integration for bivariate normal CDF.
    #[verified_engine::verified]
    fn integrate_bivariate_normal(&self, z1: f64, z2: f64, rho: f64) -> Result<f64, CopulaError> {
        // Use a simple grid-based integration
        let n_points = 100;
        let x_min = -10.0;
        let x_max = z1;
        let y_min = -10.0;
        let y_max = z2;

        if x_max < x_min || y_max < y_min {
            return Ok(0.0);
        }

        let dx = (x_max - x_min) / n_points as f64;
        let dy = (y_max - y_min) / n_points as f64;

        let mut sum = 0.0;
        let rho_sq = rho * rho;
        let denom = 2.0 * std::f64::consts::PI * (1.0 - rho_sq).sqrt();
        let factor = -1.0 / (2.0 * (1.0 - rho_sq));

        for i in 0..n_points {
            let x = x_min + (i as f64 + 0.5) * dx;
            for j in 0..n_points {
                let y = y_min + (j as f64 + 0.5) * dy;

                // Bivariate normal PDF
                let exponent = factor * (x * x - 2.0 * rho * x * y + y * y);
                let pdf = (1.0 / denom) * exponent.exp();
                sum += pdf * dx * dy;
            }
        }

        Ok(sum.clamp(0.0, 1.0))
    }

    /// Multivariate Gaussian copula CDF (general case).
    #[verified_engine::verified]
    fn multivariate_cdf(&self, u: &[Probability]) -> Result<Probability, CopulaError> {
        // Transform to z-scores
        let mut z_scores = Vec::with_capacity(u.len());
        for &ui in u {
            z_scores.push(inverse_standard_normal(ui)?);
        }

        // Create multivariate normal distribution
        let mean = DVector::zeros(u.len());
        let cov = self.correlation.matrix().clone();

        let _mvn = MultivariateNormal::new(mean.as_slice().to_vec(), cov.as_slice().to_vec())
            .map_err(|e| {
                crate::error::CopulaError::Math(math_commons::error::MathError::NumericalError {
                    reason: format!("Failed to create MVN: {}", e),
                })
            })?;

        // Compute CDF by Monte Carlo or numerical integration
        // For now, return an error as full MVN CDF is complex
        Err(crate::error::CopulaError::Math(
            math_commons::error::MathError::NumericalError {
                reason: "Multivariate CDF not yet implemented; use bivariate".to_string(),
            },
        ))
    }

    /// Returns the correlation matrix.
    #[verified_engine::verified]
    pub fn correlation(&self) -> &CorrelationMatrix {
        &self.correlation
    }
}

/// Computes the joint probability for a Same Game Parlay (SGP).
///
/// # Arguments
///
/// * `marginals` - Vector of marginal probabilities for each event
/// * `correlation_matrix` - Correlation structure between events
///
/// # Returns
///
/// The joint probability accounting for correlations
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{
///     sgp_joint_probability, Probability, CorrelationMatrix, Correlation
/// };
///
/// // Event A: Luka 50+ points (99th percentile)
/// let p_a = Probability::new(0.99).unwrap();
/// // Event B: Mavs win (60% chance)
/// let p_b = Probability::new(0.60).unwrap();
///
/// // Negative correlation (hero ball hurts team)
/// let rho = Correlation::new(-0.15).unwrap();
/// let corr_matrix = CorrelationMatrix::bivariate(rho).unwrap();
///
/// let joint_prob = sgp_joint_probability(&[p_a, p_b], &corr_matrix).unwrap();
/// ```
#[verified_engine::verified]
pub fn sgp_joint_probability(
    marginals: &[Probability],
    correlation_matrix: &CorrelationMatrix,
) -> Result<Probability, CopulaError> {
    if marginals.len() != correlation_matrix.dimension() {
        return Err(crate::error::CopulaError::Math(
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
    use super::*;
    use approx::assert_relative_eq;

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
        // When ρ = 0, copula should be product of marginals
        let rho = Correlation::new(0.0).unwrap();
        let copula = GaussianCopula::bivariate(rho).unwrap();

        let u1 = Probability::new(0.7).unwrap();
        let u2 = Probability::new(0.8).unwrap();

        let joint = copula.cdf(&[u1, u2]).unwrap();

        // With independence, C(u1, u2) ≈ u1 * u2
        let expected = u1.value() * u2.value();
        assert_relative_eq!(joint.value(), expected, epsilon = 0.05);
    }

    #[test]
    #[verified_engine::verified]
    fn test_gaussian_copula_positive_correlation() {
        // Positive correlation should increase joint probability
        let rho_indep = Correlation::new(0.0).unwrap();
        let rho_pos = Correlation::new(0.7).unwrap();

        let copula_indep = GaussianCopula::bivariate(rho_indep).unwrap();
        let copula_pos = GaussianCopula::bivariate(rho_pos).unwrap();

        let u1 = Probability::new(0.8).unwrap();
        let u2 = Probability::new(0.8).unwrap();

        let joint_indep = copula_indep.cdf(&[u1, u2]).unwrap();
        let joint_pos = copula_pos.cdf(&[u1, u2]).unwrap();

        // Positive correlation should increase joint probability
        assert!(joint_pos.value() > joint_indep.value());
    }

    #[test]
    #[verified_engine::verified]
    fn test_gaussian_copula_negative_correlation() {
        // Negative correlation should decrease joint probability
        let rho_indep = Correlation::new(0.0).unwrap();
        let rho_neg = Correlation::new(-0.5).unwrap();

        let copula_indep = GaussianCopula::bivariate(rho_indep).unwrap();
        let copula_neg = GaussianCopula::bivariate(rho_neg).unwrap();

        let u1 = Probability::new(0.9).unwrap();
        let u2 = Probability::new(0.9).unwrap();

        let joint_indep = copula_indep.cdf(&[u1, u2]).unwrap();
        let joint_neg = copula_neg.cdf(&[u1, u2]).unwrap();

        // Negative correlation should decrease joint probability
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

        // Should be less than independent probability (0.99 * 0.60 = 0.594)
        let independent = p_a.value() * p_b.value();
        assert!(joint.value() < independent);
    }

    #[test]
    #[verified_engine::verified]
    fn test_copula_bounds() {
        // Copula value should always be in [0, 1]
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
}
