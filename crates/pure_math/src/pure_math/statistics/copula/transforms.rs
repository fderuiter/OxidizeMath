//! Probability integral transform for copula operations.

use super::core::Probability;
use crate::error::CopulaError;
use statrs::distribution::{ContinuousCDF, Normal};
use statrs::function::erf::{erfc, erfc_inv};
use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2, TAU};

/// Fast zero-allocation standard normal CDF `Phi(z)` accurate to full IEEE-754 precision.
#[inline]
pub(crate) fn fast_norm_cdf(z: f64) -> f64 {
    if z.is_nan() {
        return f64::NAN;
    }
    if z == f64::NEG_INFINITY {
        return 0.0;
    }
    if z == f64::INFINITY {
        return 1.0;
    }
    (0.5 * erfc(-z * FRAC_1_SQRT_2)).clamp(0.0, 1.0)
}

/// Fast zero-allocation inverse standard normal CDF `Phi^-1(p)` refined to 1-ULP `f64` accuracy
/// via a single cubic Halley step on `erfc`.
#[inline]
pub(crate) fn fast_norm_inv(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let (z0, delta) = if p <= 0.5 {
        let z = -SQRT_2 * erfc_inv(2.0 * p);
        (z, 0.5 * erfc(-z * FRAC_1_SQRT_2) - p)
    } else {
        let q = 1.0 - p;
        let z = SQRT_2 * erfc_inv(2.0 * q);
        (z, q - 0.5 * erfc(z * FRAC_1_SQRT_2))
    };
    let pdf = (-0.5 * z0 * z0).exp() / TAU.sqrt();
    if pdf > f64::MIN_POSITIVE {
        let ratio = delta / pdf;
        z0 - ratio / (1.0 + 0.5 * z0 * ratio)
    } else {
        z0
    }
}

/// Transforms a value from its original distribution to a uniform `[0, 1]` distribution.
///
/// Implements the **Probability Integral Transform** `U = F(X)`.
pub trait ProbabilityTransform {
    /// Transforms a value to its cumulative probability `U in [0, 1]`.
    #[verified_engine::verified]
    fn to_uniform(&self, value: f64) -> Result<Probability, CopulaError>;

    /// Inverse transform from uniform `U in [0, 1]` to the original distribution.
    #[verified_engine::verified]
    fn inverse_transform(&self, u: Probability) -> f64;
}

/// Normal (Gaussian) probability integral transformation.
pub struct NormalTransform {
    normal: Normal,
}

impl NormalTransform {
    /// Creates a new normal transform with `mean` and positive `std_dev`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::NormalTransform;
    ///
    /// let transform = NormalTransform::new(0.0, 1.0).unwrap();
    /// # let _ = transform;
    /// ```
    #[verified_engine::verified]
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, CopulaError> {
        if !mean.is_finite() {
            return Err(CopulaError::InvalidCorrelation { value: mean });
        }
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return Err(CopulaError::InvalidCorrelation { value: std_dev });
        }
        let normal = Normal::new(mean, std_dev)
            .map_err(|_| CopulaError::InvalidCorrelation { value: std_dev })?;
        Ok(Self { normal })
    }

    /// Creates a standard normal transform `N(0, 1)`.
    #[verified_engine::verified]
    pub fn standard() -> Result<Self, CopulaError> {
        Self::new(0.0, 1.0)
    }
}

impl ProbabilityTransform for NormalTransform {
    #[verified_engine::verified]
    fn to_uniform(&self, value: f64) -> Result<Probability, CopulaError> {
        Probability::new(self.normal.cdf(value))
    }

    #[verified_engine::verified]
    fn inverse_transform(&self, u: Probability) -> f64 {
        self.normal.inverse_cdf(u.value())
    }
}

/// Inverse standard normal CDF (`Phi^-1`).
///
/// Maps a uniform probability `u in [0, 1]` to a standard normal quantile `z`.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::{Probability, inverse_standard_normal};
///
/// let u = Probability::new(0.975).unwrap();
/// let z = inverse_standard_normal(u).unwrap();
/// assert!((z - 1.95996).abs() < 1e-4);
/// ```
#[verified_engine::verified]
pub fn inverse_standard_normal(u: Probability) -> Result<f64, CopulaError> {
    Ok(fast_norm_inv(u.value()))
}

/// Standard normal CDF (`Phi`).
///
/// Maps a standard normal quantile `z` to a cumulative probability in `[0, 1]`.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::copula::standard_normal_cdf;
///
/// let p = standard_normal_cdf(1.959964).unwrap();
/// assert!((p.value() - 0.975).abs() < 1e-6);
/// ```
#[verified_engine::verified]
pub fn standard_normal_cdf(z: f64) -> Result<Probability, CopulaError> {
    if z.is_nan() {
        return Err(CopulaError::InvalidProbability { value: z });
    }
    Probability::new(fast_norm_cdf(z))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_normal_transform_standard_and_parametric() {
        let std_tf = NormalTransform::standard().unwrap();
        let u = std_tf.to_uniform(0.0).unwrap();
        assert_relative_eq!(
            u.value(),
            0.5,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
        assert_relative_eq!(
            std_tf.inverse_transform(Probability::new(0.5).unwrap()),
            0.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );

        let param_tf = NormalTransform::new(10.0, 2.0).unwrap();
        assert_relative_eq!(
            param_tf.to_uniform(10.0).unwrap().value(),
            0.5,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
        assert!(NormalTransform::new(0.0, 0.0).is_err());
        assert!(NormalTransform::new(0.0, -1.0).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_standard_normal_round_trip_and_boundaries() {
        for &value in &[-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0] {
            let u = standard_normal_cdf(value).unwrap();
            let recovered = inverse_standard_normal(u).unwrap();
            assert_relative_eq!(
                recovered,
                value,
                epsilon = math_commons::registry::TOLERANCE_HIGH
            );
        }
        assert_eq!(
            inverse_standard_normal(Probability::new(0.0).unwrap()).unwrap(),
            f64::NEG_INFINITY
        );
        assert_eq!(
            inverse_standard_normal(Probability::new(1.0).unwrap()).unwrap(),
            f64::INFINITY
        );
    }
}
