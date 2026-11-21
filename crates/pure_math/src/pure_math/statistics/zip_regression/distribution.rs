//! Zero-Inflated Poisson distribution implementation.
//!
//! This module implements the probability mass function (PMF) and statistical
//! properties of the Zero-Inflated Poisson (ZIP) distribution.

use super::core::{Count, ZipParams};
use crate::error::ZipError;
use statrs::function::gamma::ln_gamma;

/// Zero-Inflated Poisson distribution.
///
/// A ZIP distribution is a mixture model that combines:
/// 1. A point mass at zero (structural zeros) with probability ρ
/// 2. A Poisson distribution with rate λ and probability (1-ρ)
///
/// # Mathematical Definition
///
/// The probability mass function is:
///
/// ```text
/// P(Y = 0) = ρ + (1-ρ)e^(-λ)
/// P(Y = k) = (1-ρ)(λ^k e^(-λ))/k!  for k > 0
/// ```
///
/// # Statistical Properties
///
/// - **Mean**: E\[Y\] = (1-ρ)λ
/// - **Variance**: Var\[Y\] = (1-ρ)λ(1 + ρλ)
/// - The variance is strictly greater than the mean when ρ > 0, demonstrating overdispersion
///
/// # References
///
/// Lambert, D. (1992). "Zero-Inflated Poisson Regression, with an Application to Defects in Manufacturing."
/// *Technometrics*, 34(1), 1-14.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipDistribution {
    params: ZipParams,
}

impl ZipDistribution {
    /// Creates a new ZIP distribution.
    ///
    /// # Arguments
    ///
    /// * `params` - The distribution parameters (ρ and λ)
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::{ZipDistribution, ZipParams};
    ///
    /// let params = ZipParams::from_values(0.3, 2.0).unwrap();
    /// let dist = ZipDistribution::new(params);
    /// ```
    #[verified_engine::verified]
    pub fn new(params: ZipParams) -> Self {
        Self { params }
    }

    /// Creates a new ZIP distribution from raw parameter values.
    ///
    /// # Arguments
    ///
    /// * `rho` - The zero-inflation probability (must be in [0, 1])
    /// * `lambda` - The Poisson rate (must be positive)
    ///
    /// # Returns
    ///
    /// * `Result<ZipDistribution, ZipError>` - The distribution or an error
    ///
    /// # Errors
    ///
    /// Returns a `ZipError` if the underlying parameters are invalid (e.g., negative lambda).
    #[verified_engine::verified]
    pub fn from_values(rho: f64, lambda: f64) -> Result<Self, ZipError> {
        Ok(Self {
            params: ZipParams::from_values(rho, lambda)?,
        })
    }

    /// Returns the distribution parameters.
    #[verified_engine::verified]
    pub fn params(&self) -> ZipParams {
        self.params
    }

    /// Computes the probability mass function P(Y = k).
    ///
    /// # Arguments
    ///
    /// * `count` - The count value k
    ///
    /// # Returns
    ///
    /// The probability of observing exactly k events
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::{ZipDistribution, Count};
    ///
    /// let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
    /// let prob_zero = dist.pmf(Count::new(0));
    /// let prob_one = dist.pmf(Count::new(1));
    /// ```
    #[verified_engine::verified]
    pub fn pmf(&self, count: Count) -> f64 {
        let k = count.value();
        let rho = self.params.rho.value();
        let lambda = self.params.lambda.value();

        if k == 0 {
            // P(Y = 0) = ρ + (1-ρ)e^(-λ)
            rho + (1.0 - rho) * (-lambda).exp()
        } else {
            // P(Y = k) = (1-ρ)(λ^k e^(-λ))/k!
            (1.0 - rho) * self.poisson_pmf(lambda, k)
        }
    }

    /// Computes the cumulative distribution function P(Y ≤ k).
    ///
    /// # Arguments
    ///
    /// * `count` - The count value k
    ///
    /// # Returns
    ///
    /// The probability of observing k or fewer events
    #[verified_engine::verified]
    pub fn cdf(&self, count: Count) -> f64 {
        let k = count.value();
        let mut sum = 0.0;
        for i in 0..=k {
            sum += self.pmf(Count::new(i));
        }
        sum
    }

    /// Computes the mean (expected value) of the distribution.
    ///
    /// The mean is given by: E\[Y\] = (1-ρ)λ
    ///
    /// # Returns
    ///
    /// The expected value
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::ZipDistribution;
    ///
    /// let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
    /// let mean = dist.mean();
    /// assert!((mean - 2.4).abs() < math_explorer::registry::TOLERANCE_STANDARD);  // (1-0.2)*3.0 = 2.4
    /// ```
    #[verified_engine::verified]
    pub fn mean(&self) -> f64 {
        let rho = self.params.rho.value();
        let lambda = self.params.lambda.value();
        (1.0 - rho) * lambda
    }

    /// Computes the variance of the distribution.
    ///
    /// The variance is given by: Var\[Y\] = (1-ρ)λ(1 + ρλ)
    ///
    /// Note that Var\[Y\] > E\[Y\] when ρ > 0, demonstrating overdispersion.
    ///
    /// # Returns
    ///
    /// The variance
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::ZipDistribution;
    ///
    /// let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
    /// let variance = dist.variance();
    /// let mean = dist.mean();
    /// // For ZIP with ρ > 0, variance > mean (overdispersion)
    /// assert!(variance > mean);
    /// ```
    #[verified_engine::verified]
    pub fn variance(&self) -> f64 {
        let rho = self.params.rho.value();
        let lambda = self.params.lambda.value();
        (1.0 - rho) * lambda * (1.0 + rho * lambda)
    }

    /// Computes the standard deviation of the distribution.
    #[verified_engine::verified]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Helper function to compute the Poisson PMF: P(X = k) = (λ^k e^(-λ))/k!
    #[verified_engine::verified]
    fn poisson_pmf(&self, lambda: f64, k: u32) -> f64 {
        let k_f64 = k as f64;
        let log_prob = k_f64 * lambda.ln() - lambda - self.log_factorial(k);
        log_prob.exp()
    }

    /// Helper function to compute log(k!) using the gamma function.
    #[verified_engine::verified]
    fn log_factorial(&self, k: u32) -> f64 {
        if k == 0 {
            0.0
        } else {
            // log(k!) = log(Γ(k+1))
            Self::log_gamma((k + 1) as f64)
        }
    }

    /// Helper function to compute log(Γ(x)) using statrs.
    #[verified_engine::verified]
    fn log_gamma(x: f64) -> f64 {
        ln_gamma(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_zip_pmf_zero_count() {
        // With rho=0.2, lambda=3.0
        // P(Y=0) = 0.2 + 0.8 * e^(-3.0)
        let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
        let prob = dist.pmf(Count::new(0));

        let expected = 0.2 + 0.8 * (-3.0_f64).exp();
        assert!((prob - expected).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_pmf_positive_count() {
        // With rho=0.2, lambda=3.0
        // P(Y=1) = 0.8 * (3^1 * e^(-3)) / 1!
        let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
        let prob = dist.pmf(Count::new(1));

        let lambda = 3.0_f64;
        let poisson_prob = lambda * (-lambda).exp();
        let expected = 0.8 * poisson_prob;
        assert!((prob - expected).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_mean() {
        let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
        let mean = dist.mean();
        // E[Y] = (1-0.2)*3.0 = 2.4
        assert!((mean - 2.4).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_variance() {
        let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
        let variance = dist.variance();
        // Var[Y] = (1-0.2)*3.0*(1 + 0.2*3.0) = 0.8*3.0*1.6 = 3.84
        let expected = 0.8 * 3.0 * 1.6;
        assert!((variance - expected).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_overdispersion() {
        // ZIP distributions with rho > 0 should exhibit overdispersion
        let dist = ZipDistribution::from_values(0.3, 2.5).unwrap();
        let mean = dist.mean();
        let variance = dist.variance();

        // Variance should be strictly greater than mean
        assert!(variance > mean);
    }

    #[test]
    #[verified_engine::verified]
    fn test_reduce_to_poisson() {
        // When rho = 0, ZIP reduces to standard Poisson
        let dist = ZipDistribution::from_values(0.0, 2.0).unwrap();
        let mean = dist.mean();
        let variance = dist.variance();

        // For standard Poisson, mean = variance = lambda
        assert!((mean - 2.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
        assert!((variance - 2.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
    }

    #[test]
    #[verified_engine::verified]
    fn test_pmf_sums_to_one() {
        let dist = ZipDistribution::from_values(0.2, 2.0).unwrap();
        let mut sum = 0.0;

        // Sum over first 20 values (should be very close to 1.0)
        for k in 0..20 {
            sum += dist.pmf(Count::new(k));
        }

        assert!((sum - 1.0).abs() < 1e-3);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cdf_monotonic() {
        let dist = ZipDistribution::from_values(0.15, 2.5).unwrap();

        let mut prev_cdf = 0.0;
        for k in 0..10 {
            let cdf = dist.cdf(Count::new(k));
            assert!(cdf >= prev_cdf);
            assert!(cdf <= 1.0);
            prev_cdf = cdf;
        }
    }
}
