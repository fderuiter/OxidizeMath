//! Core types for Ornstein-Uhlenbeck process.

use crate::error::OuError;
use math_commons::primitives::{NonNegativeFloat, PositiveFloat};

/// The long-term mean μ (mu).
///
/// The equilibrium level to which the process reverts over time.
/// In sports analytics, this represents the player's "true skill" or baseline performance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LongTermMean(f64);

impl LongTermMean {
    /// Creates a new long-term mean.
    ///
    /// # Arguments
    ///
    /// * `value` - The mean value (any finite value)
    ///
    /// # Returns
    ///
    /// * `Result<LongTermMean, OuError>` - The validated mean or an error
    ///
    /// # Errors
    ///
    /// Returns `OuError::InvalidSimulationParams` if the value is non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::LongTermMean;
    ///
    /// let mu = LongTermMean::new(0.45).unwrap();
    /// assert_eq!(mu.value(), 0.45);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, OuError> {
        if !value.is_finite() {
            return Err(OuError::InvalidSimulationParams {
                reason: "Long-term mean must be finite".to_string(),
            });
        }
        Ok(Self(value))
    }

    /// Returns the raw mean value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for LongTermMean {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for LongTermMean {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = f64::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// Parameters for an Ornstein-Uhlenbeck process.
///
/// This struct encapsulates all the parameters needed to define an OU process:
/// - μ (mu): Long-term mean
/// - θ (theta): Mean reversion rate
/// - σ (sigma): Volatility
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct OuParams {
    /// The long-term mean μ.
    pub mu: LongTermMean,
    /// The mean reversion rate θ.
    pub theta: PositiveFloat,
    /// The volatility σ.
    pub sigma: NonNegativeFloat,
}

impl OuParams {
    /// Creates new OU parameters.
    ///
    /// # Arguments
    ///
    /// * `mu` - The long-term mean
    /// * `theta` - The mean reversion rate
    /// * `sigma` - The volatility
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::ou_process::{
    ///     OuParams, LongTermMean
    /// };
    /// use math_explorer::primitives::{PositiveFloat, NonNegativeFloat};
    ///
    /// let params = OuParams::new(
    ///     LongTermMean::new(0.5).unwrap(),
    ///     PositiveFloat::new(1.0).unwrap(),
    ///     NonNegativeFloat::new(0.3).unwrap()
    /// );
    /// ```
    #[verified_engine::verified]
    pub fn new(mu: LongTermMean, theta: PositiveFloat, sigma: NonNegativeFloat) -> Self {
        Self { mu, theta, sigma }
    }

    /// Creates new OU parameters from raw f64 values with validation.
    ///
    /// # Arguments
    ///
    /// * `mu` - The long-term mean (any finite value)
    /// * `theta` - The mean reversion rate (must be positive)
    /// * `sigma` - The volatility (must be non-negative)
    ///
    /// # Returns
    ///
    /// * `Result<OuParams, OuError>` - The validated parameters or an error
    ///
    /// # Errors
    ///
    /// Returns an `OuError` if any parameter is invalid (e.g., negative volatility).
    #[verified_engine::verified]
    pub fn from_values(mu: f64, theta: f64, sigma: f64) -> Result<Self, OuError> {
        Ok(Self {
            mu: LongTermMean::new(mu)?,
            theta: PositiveFloat::new(theta)
                .map_err(|_| OuError::InvalidMeanReversionRate { value: theta })?,
            sigma: NonNegativeFloat::new(sigma)
                .map_err(|_| OuError::InvalidVolatility { value: sigma })?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_long_term_mean_valid() {
        let mu = LongTermMean::new(0.5).unwrap();
        assert_eq!(mu.value(), 0.5);

        let mu_negative = LongTermMean::new(-1.5).unwrap();
        assert_eq!(mu_negative.value(), -1.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_long_term_mean_invalid() {
        assert!(LongTermMean::new(f64::NAN).is_err());
        assert!(LongTermMean::new(f64::INFINITY).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_ou_params() {
        let params = OuParams::from_values(0.5, 1.0, 0.3).unwrap();
        assert_eq!(params.mu.value(), 0.5);
        assert_eq!(params.theta.value(), 1.0);
        assert_eq!(params.sigma.value(), 0.3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_ou_params_invalid() {
        // Invalid theta (negative)
        assert!(OuParams::from_values(0.5, -1.0, 0.3).is_err());

        // Invalid sigma (negative)
        assert!(OuParams::from_values(0.5, 1.0, -0.3).is_err());

        // Invalid mu (NaN)
        assert!(OuParams::from_values(f64::NAN, 1.0, 0.3).is_err());
    }
}
