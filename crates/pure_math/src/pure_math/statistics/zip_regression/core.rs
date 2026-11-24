//! Core types for Zero-Inflated Poisson regression.

use crate::error::ZipError;

/// The rate parameter λ (lambda) for the Poisson process.
///
/// Represents the expected count *given* the subject is in the active state.
/// Must be strictly positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonRate(f64);

impl PoissonRate {
    /// Creates a new PoissonRate.
    ///
    /// # Arguments
    ///
    /// * `value` - The rate value (must be positive)
    ///
    /// # Returns
    ///
    /// * `Result<PoissonRate, ZipError>` - The validated rate or an error
    ///
    /// # Errors
    ///
    /// Returns `ZipError::InvalidRate` if the value is zero, negative, or non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::PoissonRate;
    ///
    /// let lambda = PoissonRate::new(2.5).unwrap();
    /// assert_eq!(lambda.value(), 2.5);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, ZipError> {
        if value <= 0.0 || !value.is_finite() {
            return Err(ZipError::InvalidRate { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw rate value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for PoissonRate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for PoissonRate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = f64::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// The zero-inflation parameter ρ (rho).
///
/// Represents the probability of a structural zero (i.e., the subject is in the "Always Zero" state).
/// Must be in the range [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZeroInflation(f64);

impl ZeroInflation {
    /// Creates a new ZeroInflation parameter.
    ///
    /// # Arguments
    ///
    /// * `value` - The probability value (must be in [0, 1])
    ///
    /// # Returns
    ///
    /// * `Result<ZeroInflation, ZipError>` - The validated parameter or an error
    ///
    /// # Errors
    ///
    /// Returns `ZipError::InvalidProbability` if the value is outside [0, 1] or non-finite.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::ZeroInflation;
    ///
    /// let rho = ZeroInflation::new(0.3).unwrap();
    /// assert_eq!(rho.value(), 0.3);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, ZipError> {
        if !(0.0..=1.0).contains(&value) || !value.is_finite() {
            return Err(ZipError::InvalidProbability {
                value,
                parameter: "rho (zero-inflation)".to_string(),
            });
        }
        Ok(Self(value))
    }

    /// Returns the raw probability value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for ZeroInflation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_f64(self.0)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ZeroInflation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = f64::deserialize(deserializer)?;
        Self::new(raw).map_err(serde::de::Error::custom)
    }
}

/// A count observation (non-negative integer represented as f64).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Count(u32);

impl Count {
    /// Creates a new Count from a u32.
    #[verified_engine::verified]
    pub fn new(value: u32) -> Self {
        Self(value)
    }

    /// Creates a Count from an f64, checking that it's a non-negative integer.
    ///
    /// # Arguments
    ///
    /// * `value` - The count value (must be a non-negative integer)
    ///
    /// # Returns
    ///
    /// * `Result<Count, ZipError>` - The validated count or an error
    ///
    /// # Errors
    ///
    /// Returns `ZipError::InvalidCount` if the value is negative, non-finite, or has a fractional part.
    #[verified_engine::verified]
    pub fn from_f64(value: f64) -> Result<Self, ZipError> {
        if value < 0.0 || !value.is_finite() || value.fract() != 0.0 {
            return Err(ZipError::InvalidCount { value });
        }
        Ok(Self(value as u32))
    }

    /// Returns the count as a u32.
    #[verified_engine::verified]
    pub fn value(&self) -> u32 {
        self.0
    }

    /// Returns the count as an f64.
    #[verified_engine::verified]
    pub fn as_f64(&self) -> f64 {
        self.0 as f64
    }
}

/// Parameters for a Zero-Inflated Poisson distribution.
///
/// This struct encapsulates both the zero-inflation probability (ρ)
/// and the Poisson rate (λ).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ZipParams {
    /// The zero-inflation probability ρ.
    pub rho: ZeroInflation,
    /// The Poisson rate λ.
    pub lambda: PoissonRate,
}

impl ZipParams {
    /// Creates new ZIP parameters.
    ///
    /// # Arguments
    ///
    /// * `rho` - The zero-inflation probability
    /// * `lambda` - The Poisson rate
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::{ZipParams, ZeroInflation, PoissonRate};
    ///
    /// let params = ZipParams::new(
    ///     ZeroInflation::new(0.2).unwrap(),
    ///     PoissonRate::new(3.0).unwrap()
    /// );
    /// ```
    #[verified_engine::verified]
    pub fn new(rho: ZeroInflation, lambda: PoissonRate) -> Self {
        Self { rho, lambda }
    }

    /// Creates new ZIP parameters from raw f64 values with validation.
    ///
    /// # Arguments
    ///
    /// * `rho` - The zero-inflation probability (must be in [0, 1])
    /// * `lambda` - The Poisson rate (must be positive)
    ///
    /// # Returns
    ///
    /// * `Result<ZipParams, ZipError>` - The validated parameters or an error
    ///
    /// # Errors
    ///
    /// Returns `ZipError` if either parameter is invalid according to its constraints.
    #[verified_engine::verified]
    pub fn from_values(rho: f64, lambda: f64) -> Result<Self, ZipError> {
        Ok(Self {
            rho: ZeroInflation::new(rho)?,
            lambda: PoissonRate::new(lambda)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_poisson_rate_valid() {
        let rate = PoissonRate::new(2.5).unwrap();
        assert_eq!(rate.value(), 2.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_poisson_rate_invalid() {
        assert!(PoissonRate::new(-1.0).is_err());
        assert!(PoissonRate::new(0.0).is_err());
        assert!(PoissonRate::new(f64::NAN).is_err());
        assert!(PoissonRate::new(f64::INFINITY).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_zero_inflation_valid() {
        let rho = ZeroInflation::new(0.3).unwrap();
        assert_eq!(rho.value(), 0.3);

        let rho_zero = ZeroInflation::new(0.0).unwrap();
        assert_eq!(rho_zero.value(), 0.0);

        let rho_one = ZeroInflation::new(1.0).unwrap();
        assert_eq!(rho_one.value(), 1.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_zero_inflation_invalid() {
        assert!(ZeroInflation::new(-0.1).is_err());
        assert!(ZeroInflation::new(1.1).is_err());
        assert!(ZeroInflation::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_count_valid() {
        let count = Count::new(5);
        assert_eq!(count.value(), 5);
        assert_eq!(count.as_f64(), 5.0);

        let count_from_f64 = Count::from_f64(3.0).unwrap();
        assert_eq!(count_from_f64.value(), 3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_count_invalid() {
        assert!(Count::from_f64(-1.0).is_err());
        assert!(Count::from_f64(2.5).is_err());
        assert!(Count::from_f64(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_params() {
        let params = ZipParams::from_values(0.2, 3.0).unwrap();
        assert_eq!(params.rho.value(), 0.2);
        assert_eq!(params.lambda.value(), 3.0);
    }
}
