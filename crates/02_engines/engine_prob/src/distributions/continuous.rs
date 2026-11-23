//! Univariate continuous distributions: [`Normal`], [`Uniform`], and [`Exponential`].

use super::{ContinuousDist, next_f64_open01, next_f64_unit, sample_standard_normal};
use crate::error::ProbError;
use crate::special::{
    standard_normal_cdf, standard_normal_ln_pdf, standard_normal_pdf, standard_normal_quantile,
};
use rand_core::RngCore;

/// Univariate Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ parameterized by mean $\mu$ and standard deviation $\sigma > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Normal {
    /// Constructs $\mathcal{N}(\mu, \sigma^2)$ with $\sigma > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, ProbError> {
        if !mean.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "mean",
                value: mean,
            });
        }
        if !std_dev.is_finite() || std_dev <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "std_dev",
                value: std_dev,
            });
        }
        debug_assert!(std_dev > 0.0);
        Ok(Self { mean, std_dev })
    }

    /// Returns the standard normal distribution $\mathcal{N}(0, 1)$.
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub const fn standard() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    /// Distribution mean $\mu$.
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub const fn location(&self) -> f64 {
        self.mean
    }

    /// Distribution standard deviation $\sigma$.
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub const fn std_dev(&self) -> f64 {
        self.std_dev
    }
}

impl ContinuousDist for Normal {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        standard_normal_pdf((x - self.mean) / self.std_dev) / self.std_dev
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        standard_normal_ln_pdf((x - self.mean) / self.std_dev) - libm::log(self.std_dev)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        standard_normal_cdf((x - self.mean) / self.std_dev)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        let z = standard_normal_quantile(p)?;
        Ok(self.mean + self.std_dev * z)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn mean(&self) -> f64 {
        self.mean
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        let val = self.mean + self.std_dev * sample_standard_normal(rng);
        debug_assert!(val.is_finite());
        val
    }
}

/// Continuous uniform distribution $\mathcal{U}(a, b)$ over $[a, b]$ with $a < b$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Uniform {
    low: f64,
    high: f64,
}

impl Uniform {
    /// Constructs $\mathcal{U}(\text{low}, \text{high})$ requiring finite `low < high`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(low: f64, high: f64) -> Result<Self, ProbError> {
        if !low.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "low",
                value: low,
            });
        }
        if !high.is_finite() || high <= low {
            return Err(ProbError::InvalidParameter {
                name: "high",
                value: high,
            });
        }
        debug_assert!(high > low);
        Ok(Self { low, high })
    }

    /// Lower support bound $a$.
    #[must_use]
    #[inline]
    pub const fn low(&self) -> f64 {
        self.low
    }

    /// Upper support bound $b$.
    #[must_use]
    #[inline]
    pub const fn high(&self) -> f64 {
        self.high
    }
}

impl ContinuousDist for Uniform {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        if x < self.low || x > self.high {
            0.0
        } else {
            1.0 / (self.high - self.low)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.low || x > self.high {
            f64::NEG_INFINITY
        } else {
            -libm::log(self.high - self.low)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.low {
            0.0
        } else if x >= self.high {
            1.0
        } else {
            (x - self.low) / (self.high - self.low)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        Ok(self.low + p * (self.high - self.low))
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn mean(&self) -> f64 {
        0.5 * (self.low + self.high)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn variance(&self) -> f64 {
        let span = self.high - self.low;
        span * span / 12.0
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        self.low + next_f64_unit(rng) * (self.high - self.low)
    }
}

/// Exponential distribution $\operatorname{Exp}(\lambda)$ with rate $\lambda > 0$ over $[0, \infty)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exponential {
    rate: f64,
}

impl Exponential {
    /// Constructs $\operatorname{Exp}(\lambda)$ with $\lambda > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(rate: f64) -> Result<Self, ProbError> {
        if !rate.is_finite() || rate <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "rate",
                value: rate,
            });
        }
        debug_assert!(rate > 0.0);
        Ok(Self { rate })
    }

    /// Rate parameter $\lambda$.
    #[must_use]
    #[inline]
    pub const fn rate(&self) -> f64 {
        self.rate
    }
}

impl ContinuousDist for Exponential {
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * libm::exp(-self.rate * x)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            libm::log(self.rate) - self.rate * x
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            -libm::expm1(-self.rate * x)
        }
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn quantile(&self, p: f64) -> Result<f64, ProbError> {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(ProbError::InvalidProbability { value: p });
        }
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(-libm::log1p(-p) / self.rate)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn mean(&self) -> f64 {
        1.0 / self.rate
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn variance(&self) -> f64 {
        1.0 / (self.rate * self.rate)
    }

    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> f64 {
        -libm::log(next_f64_open01(rng)) / self.rate
    }
}
