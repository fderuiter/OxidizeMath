//! Probability integral transform for copula operations.

use super::core::Probability;
use crate::error::CopulaError;
use statrs::distribution::{ContinuousCDF, Normal};

/// Transforms a value from its original distribution to a uniform \[0,1\] distribution.
///
/// This implements the **Probability Integral Transform**:
/// U = F(X)
///
/// where F is the cumulative distribution function (CDF) of X.
pub trait ProbabilityTransform {
    /// Transforms a value to its cumulative probability.
    ///
    /// # Arguments
    ///
    /// * `value` - The value in the original distribution
    ///
    /// # Returns
    ///
    /// The cumulative probability U ∈ [0, 1]
    #[verified_engine::verified]
    fn to_uniform(&self, value: f64) -> Result<Probability, CopulaError>;

    /// Inverse transform from uniform to original distribution.
    ///
    /// # Arguments
    ///
    /// * `u` - The uniform probability
    ///
    /// # Returns
    ///
    /// The value in the original distribution
    #[verified_engine::verified]
    fn inverse_transform(&self, u: Probability) -> f64;
}

/// Standard normal (Gaussian) transformation.
///
/// Uses Φ(x) where Φ is the CDF of N(0,1).
pub struct NormalTransform {
    normal: Normal,
}

impl NormalTransform {
    /// Creates a new normal transform with mean and standard deviation.
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean of the normal distribution
    /// * `std_dev` - The standard deviation (must be positive)
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::NormalTransform;
    ///
    /// let transform = NormalTransform::new(0.0, 1.0).unwrap();
    /// ```
    #[verified_engine::verified]
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, CopulaError> {
        let normal = Normal::new(mean, std_dev).map_err(|e| {
            crate::error::CopulaError::Math(math_commons::error::MathError::NumericalError {
                reason: format!("Failed to create normal distribution: {}", e),
            })
        })?;
        Ok(Self { normal })
    }

    /// Creates a standard normal transform N(0,1).
    #[verified_engine::verified]
    pub fn standard() -> Result<Self, CopulaError> {
        Self::new(0.0, 1.0)
    }
}

impl ProbabilityTransform for NormalTransform {
    #[verified_engine::verified]
    fn to_uniform(&self, value: f64) -> Result<Probability, CopulaError> {
        let u = self.normal.cdf(value);
        Probability::new(u)
    }

    #[verified_engine::verified]
    fn inverse_transform(&self, u: Probability) -> f64 {
        // Inverse CDF (quantile function)
        self.normal.inverse_cdf(u.value())
    }
}

/// Inverse standard normal CDF (Φ⁻¹).
///
/// Maps a uniform probability u ∈ \[0,1\] to a z-score in N(0,1).
///
/// # Arguments
///
/// * `u` - The uniform probability
///
/// # Returns
///
/// The z-score such that Φ(z) = u
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{inverse_standard_normal, Probability};
///
/// let u = Probability::new(0.975).unwrap();
/// let z = inverse_standard_normal(u).unwrap();
/// // z ≈ 1.96 (97.5th percentile)
/// ```
#[verified_engine::verified]
pub fn inverse_standard_normal(u: Probability) -> Result<f64, CopulaError> {
    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        crate::error::CopulaError::Math(math_commons::error::MathError::NumericalError {
            reason: format!("Failed to create standard normal: {}", e),
        })
    })?;
    Ok(normal.inverse_cdf(u.value()))
}

/// Standard normal CDF (Φ).
///
/// Maps a z-score to a cumulative probability.
///
/// # Arguments
///
/// * `z` - The z-score
///
/// # Returns
///
/// The cumulative probability Φ(z)
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::standard_normal_cdf;
///
/// let z = 1.96;
/// let p = standard_normal_cdf(z).unwrap();
/// // p ≈ 0.975
/// ```
#[verified_engine::verified]
pub fn standard_normal_cdf(z: f64) -> Result<Probability, CopulaError> {
    let normal = Normal::new(0.0, 1.0).map_err(|e| {
        crate::error::CopulaError::Math(math_commons::error::MathError::NumericalError {
            reason: format!("Failed to create standard normal: {}", e),
        })
    })?;
    Probability::new(normal.cdf(z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_normal_transform_standard() {
        let transform = NormalTransform::standard().unwrap();

        // Test median (z=0 -> u=0.5)
        let u = transform.to_uniform(0.0).unwrap();
        assert_relative_eq!(
            u.value(),
            0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Test inverse
        let u = Probability::new(0.5).unwrap();
        let z = transform.inverse_transform(u);
        assert_relative_eq!(z, 0.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_normal_transform_with_parameters() {
        let transform = NormalTransform::new(10.0, 2.0).unwrap();

        // At mean, CDF should be 0.5
        let u = transform.to_uniform(10.0).unwrap();
        assert_relative_eq!(
            u.value(),
            0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_inverse_standard_normal() {
        // Test known values
        let u_median = Probability::new(0.5).unwrap();
        let z_median = inverse_standard_normal(u_median).unwrap();
        assert_relative_eq!(
            z_median,
            0.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // 97.5th percentile ≈ 1.96
        let u_975 = Probability::new(0.975).unwrap();
        let z_975 = inverse_standard_normal(u_975).unwrap();
        assert_relative_eq!(z_975, 1.96, epsilon = 0.01);
    }

    #[test]
    #[verified_engine::verified]
    fn test_standard_normal_cdf() {
        // Test z=0 -> p=0.5
        let p = standard_normal_cdf(0.0).unwrap();
        assert_relative_eq!(
            p.value(),
            0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Test z=1.96 -> p≈0.975
        let p = standard_normal_cdf(1.96).unwrap();
        assert_relative_eq!(p.value(), 0.975, epsilon = 0.001);
    }

    #[test]
    #[verified_engine::verified]
    fn test_round_trip() {
        let transform = NormalTransform::standard().unwrap();

        for &value in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let u = transform.to_uniform(value).unwrap();
            let recovered = transform.inverse_transform(u);
            assert_relative_eq!(
                recovered,
                value,
                epsilon = math_commons::registry::TOLERANCE_STANDARD
            );
        }
    }
}
