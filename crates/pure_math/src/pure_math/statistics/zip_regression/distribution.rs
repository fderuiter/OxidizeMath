//! Zero-Inflated Poisson distribution implementation.

use super::core::{Count, ZipParams};
use crate::error::{DistributionError, ZipError};
use crate::pure_math::statistics::distributions::DiscreteDistribution;
use crate::pure_math::statistics::distributions::special::{
    sample_poisson, sample_uniform_open, validate_probability,
};
use oxidize_core::rng::OxidizeRng;
use statrs::function::gamma::{gamma_ur, ln_gamma};

/// Zero-Inflated Poisson distribution $\mathrm{ZIP}(\rho, \lambda)$.
///
/// Combines a structural point mass at zero with probability $\rho \in [0, 1]$ and an active
/// $\mathrm{Poisson}(\lambda)$ process with weight $1 - \rho$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZipDistribution {
    params: ZipParams,
}

impl ZipDistribution {
    /// Creates a new ZIP distribution from validated [`ZipParams`].
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::zip_regression::{ZipDistribution, ZipParams};
    ///
    /// let params = ZipParams::from_values(0.3, 2.0).unwrap();
    /// let dist = ZipDistribution::new(params);
    /// assert!(dist.mean() > 0.0);
    /// ```
    #[verified_engine::verified]
    pub fn new(params: ZipParams) -> Self {
        Self { params }
    }

    /// Creates a new ZIP distribution from raw `rho` $\in [0, 1]$ and `lambda` $> 0$.
    #[verified_engine::verified]
    pub fn from_values(rho: f64, lambda: f64) -> Result<Self, ZipError> {
        Ok(Self {
            params: ZipParams::from_values(rho, lambda)?,
        })
    }

    /// Returns the underlying [`ZipParams`].
    #[verified_engine::verified]
    pub fn params(&self) -> ZipParams {
        self.params
    }

    /// Computes the probability mass function $P(Y = k)$.
    #[verified_engine::verified]
    pub fn pmf(&self, count: Count) -> f64 {
        self.pmf_u64(count.value() as u64)
    }

    /// Computes the natural log probability mass $\ln P(Y = k)$.
    #[verified_engine::verified]
    pub fn ln_pmf(&self, count: Count) -> f64 {
        self.ln_pmf_u64(count.value() as u64)
    }

    /// Computes the cumulative distribution function $P(Y \le k)$ in $O(1)$ via regularized gamma.
    #[verified_engine::verified]
    pub fn cdf(&self, count: Count) -> f64 {
        self.cdf_u64(count.value() as u64)
    }

    /// Computes the smallest [`Count`] $k$ satisfying $F(k) \ge p$ for $p \in [0, 1]$.
    #[verified_engine::verified]
    pub fn quantile(&self, p: f64) -> Result<Count, ZipError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ZipError::InvalidProbability {
                value: p,
                parameter: "p".to_string(),
            });
        }
        let k = self
            .quantile_u64(p)
            .map_err(|_| ZipError::InvalidProbability {
                value: p,
                parameter: "p".to_string(),
            })?;
        Ok(Count::new(k.min(u32::MAX as u64) as u32))
    }

    /// Draws a deterministic [`Count`] from $\mathrm{ZIP}(\rho, \lambda)$ using `rng`.
    #[verified_engine::verified]
    pub fn sample(&self, rng: &mut OxidizeRng) -> Count {
        let k = <Self as DiscreteDistribution>::sample(self, rng);
        Count::new(k.min(u32::MAX as u64) as u32)
    }

    /// Computes the mean $E(Y) = (1 - \rho)\lambda$.
    #[verified_engine::verified]
    pub fn mean(&self) -> f64 {
        (1.0 - self.params.rho.value()) * self.params.lambda.value()
    }

    /// Computes the variance $\mathrm{Var}(Y) = (1 - \rho)\lambda(1 + \rho\lambda)$.
    #[verified_engine::verified]
    pub fn variance(&self) -> f64 {
        let rho = self.params.rho.value();
        let lambda = self.params.lambda.value();
        (1.0 - rho) * lambda * (1.0 + rho * lambda)
    }

    /// Computes the standard deviation $\sqrt{\mathrm{Var}(Y)}$.
    #[verified_engine::verified]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    #[verified_engine::verified]
    fn pmf_u64(&self, k: u64) -> f64 {
        let rho = self.params.rho.value();
        let lambda = self.params.lambda.value();
        if k == 0 {
            rho + (1.0 - rho) * (-lambda).exp()
        } else if rho >= 1.0 {
            0.0
        } else {
            self.ln_pmf_u64(k).exp()
        }
    }

    #[verified_engine::verified]
    fn ln_pmf_u64(&self, k: u64) -> f64 {
        let rho = self.params.rho.value();
        let lambda = self.params.lambda.value();
        if k == 0 {
            let p0 = rho + (1.0 - rho) * (-lambda).exp();
            if p0 <= 0.0 {
                f64::NEG_INFINITY
            } else {
                p0.ln()
            }
        } else if rho >= 1.0 {
            f64::NEG_INFINITY
        } else {
            let kf = k as f64;
            (-rho).ln_1p() + kf * lambda.ln() - lambda - ln_gamma(kf + 1.0)
        }
    }

    #[verified_engine::verified]
    fn cdf_u64(&self, k: u64) -> f64 {
        let rho = self.params.rho.value();
        if rho >= 1.0 {
            return 1.0;
        }
        let lambda = self.params.lambda.value();
        (rho + (1.0 - rho) * gamma_ur((k as f64) + 1.0, lambda)).clamp(0.0, 1.0)
    }

    #[verified_engine::verified]
    fn quantile_u64(&self, p: f64) -> Result<u64, DistributionError> {
        validate_probability(p)?;
        let rho = self.params.rho.value();
        if rho >= 1.0 {
            return Ok(0);
        }
        if p >= 1.0 {
            return Err(DistributionError::InvalidProbability { value: p });
        }
        if p <= self.pmf_u64(0) {
            return Ok(0);
        }
        let mut lo = 0_u64;
        let mut hi = (self.params.lambda.value().ceil() as u64).max(4);
        while self.cdf_u64(hi) < p && hi < (1_u64 << 40) {
            lo = hi + 1;
            hi = hi.saturating_mul(2);
        }
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.cdf_u64(mid) >= p {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        Ok(lo)
    }
}

impl DiscreteDistribution for ZipDistribution {
    #[verified_engine::verified]
    fn pmf(&self, k: u64) -> f64 {
        self.pmf_u64(k)
    }

    #[verified_engine::verified]
    fn ln_pmf(&self, k: u64) -> f64 {
        self.ln_pmf_u64(k)
    }

    #[verified_engine::verified]
    fn cdf(&self, k: u64) -> f64 {
        self.cdf_u64(k)
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<u64, DistributionError> {
        self.quantile_u64(p)
    }

    #[verified_engine::verified]
    fn mean(&self) -> f64 {
        ZipDistribution::mean(self)
    }

    #[verified_engine::verified]
    fn variance(&self) -> f64 {
        ZipDistribution::variance(self)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> u64 {
        if sample_uniform_open(rng) < self.params.rho.value() {
            0
        } else {
            sample_poisson(self.params.lambda.value(), rng)
        }
    }
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    fn test_zip_pmf_moments_and_overdispersion() {
        let dist = ZipDistribution::from_values(0.2, 3.0).unwrap();
        let expected_zero = 0.2 + 0.8 * (-3.0_f64).exp();
        assert!((dist.pmf(Count::new(0)) - expected_zero).abs() < 1e-12);
        assert!((dist.ln_pmf(Count::new(1)).exp() - dist.pmf(Count::new(1))).abs() < 1e-12);
        assert!((dist.mean() - 2.4).abs() < 1e-12);
        assert!((dist.variance() - 3.84).abs() < 1e-12);
        assert!(dist.variance() > dist.mean());

        let pois_equiv = ZipDistribution::from_values(0.0, 2.0).unwrap();
        assert!((pois_equiv.mean() - 2.0).abs() < 1e-12);
        assert!((pois_equiv.variance() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_zip_quantiles_sampling_and_zero_allocation() {
        let dist = ZipDistribution::from_values(0.35, 4.0).unwrap();
        for &q in &[0.1, 0.4, 0.75, 0.95] {
            let k = dist.quantile(q).unwrap();
            assert!(dist.cdf(k) + 1e-12 >= q);
            if k.value() > 0 {
                assert!(dist.cdf(Count::new(k.value() - 1)) <= q + 1e-12);
            }
        }
        let mut rng = OxidizeRng::from_seed(314);
        let (acc, _) = VerifiedEngine::run_verified(|| {
            let q = DiscreteDistribution::quantile(&dist, 0.8).unwrap();
            let s = DiscreteDistribution::sample(&dist, &mut rng);
            q + s
        });
        assert!(acc < 10_000);
    }
}
