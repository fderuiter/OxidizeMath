use super::special::{
    sample_mean_and_var, sample_standard_normal, sample_uniform_open, standard_normal_cdf,
    standard_normal_ln_pdf, standard_normal_pdf, standard_normal_quantile, validate_positive,
    validate_probability,
};
use super::traits::ContinuousDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;

/// Univariate Gaussian distribution $\mathcal{N}(\mu, \sigma^2)$ parameterized by mean $\mu$ and
/// standard deviation $\sigma > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalDist {
    mean: f64,
    std_dev: f64,
}

impl NormalDist {
    /// Constructs $\mathcal{N}(\mu, \sigma^2)$ with finite `mean` and strictly positive `std_dev`.
    #[verified_engine::verified]
    pub fn new(mean: f64, std_dev: f64) -> Result<Self, DistributionError> {
        if !mean.is_finite() {
            return Err(DistributionError::InvalidParameter {
                name: "mean",
                value: mean,
            });
        }
        validate_positive("std_dev", std_dev)?;
        Ok(Self { mean, std_dev })
    }

    /// Standard normal distribution $\mathcal{N}(0, 1)$.
    #[verified_engine::verified]
    pub fn standard() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }

    /// Returns the location parameter $\mu$.
    #[verified_engine::verified]
    pub fn mu(&self) -> f64 {
        self.mean
    }

    /// Returns the scale parameter $\sigma$.
    #[verified_engine::verified]
    pub fn sigma(&self) -> f64 {
        self.std_dev
    }

    /// Maximum likelihood estimator $\hat{\mu} = \bar{x}$, $\hat{\sigma} = \sqrt{\frac{1}{n}\sum (x_i - \bar{x})^2}$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        let (mean, pop_var, _) = sample_mean_and_var(data)?;
        if pop_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "zero sample variance",
            });
        }
        Self::new(mean, pop_var.sqrt())
    }

    /// Method-of-moments estimator using unbiased sample variance $s^2 = \frac{1}{n-1}\sum (x_i - \bar{x})^2$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        let (mean, _, sample_var) = sample_mean_and_var(data)?;
        if sample_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "zero sample variance",
            });
        }
        Self::new(mean, sample_var.sqrt())
    }
}

impl ContinuousDistribution for NormalDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        standard_normal_pdf((x - self.mean) / self.std_dev) / self.std_dev
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        standard_normal_ln_pdf((x - self.mean) / self.std_dev) - self.std_dev.ln()
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        standard_normal_cdf((x - self.mean) / self.std_dev)
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        if p == 0.0 {
            return Ok(f64::NEG_INFINITY);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(self.mean + self.std_dev * standard_normal_quantile(p))
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(self.mean)
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        Some(self.std_dev * self.std_dev)
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (f64::NEG_INFINITY, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        self.mean + self.std_dev * sample_standard_normal(rng)
    }
}

/// Continuous uniform distribution $\mathcal{U}(a, b)$ on $[a, b]$ with $a < b$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformDist {
    lower: f64,
    upper: f64,
}

impl UniformDist {
    /// Constructs $\mathcal{U}(a, b)$ verifying finite bounds and `lower < upper`.
    #[verified_engine::verified]
    pub fn new(lower: f64, upper: f64) -> Result<Self, DistributionError> {
        if !lower.is_finite() || !upper.is_finite() || lower >= upper {
            return Err(DistributionError::InvalidBounds { lower, upper });
        }
        Ok(Self { lower, upper })
    }

    /// Returns the lower endpoint $a$.
    #[verified_engine::verified]
    pub fn lower(&self) -> f64 {
        self.lower
    }

    /// Returns the upper endpoint $b$.
    #[verified_engine::verified]
    pub fn upper(&self) -> f64 {
        self.upper
    }

    /// Order-statistic maximum likelihood estimator $\hat{a} = \min_i x_i$, $\hat{b} = \max_i x_i$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        if data.len() < 2 {
            return Err(DistributionError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &x in data {
            if !x.is_finite() {
                return Err(DistributionError::FittingFailed {
                    reason: "sample contains non-finite values",
                });
            }
            lo = lo.min(x);
            hi = hi.max(x);
        }
        if lo >= hi {
            return Err(DistributionError::FittingFailed {
                reason: "degenerate constant sample",
            });
        }
        Self::new(lo, hi)
    }

    /// Method-of-moments estimator matching $\bar{x} = \frac{a+b}{2}$ and $s^2 = \frac{(b-a)^2}{12}$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        let (mean, _, sample_var) = sample_mean_and_var(data)?;
        if sample_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "zero sample variance",
            });
        }
        let half_span = (3.0 * sample_var).sqrt();
        Self::new(mean - half_span, mean + half_span)
    }
}

impl ContinuousDistribution for UniformDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x >= self.lower && x <= self.upper {
            1.0 / (self.upper - self.lower)
        } else {
            0.0
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x >= self.lower && x <= self.upper {
            -(self.upper - self.lower).ln()
        } else {
            f64::NEG_INFINITY
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= self.lower {
            0.0
        } else if x >= self.upper {
            1.0
        } else {
            (x - self.lower) / (self.upper - self.lower)
        }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        Ok(self.lower + p * (self.upper - self.lower))
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(0.5 * (self.lower + self.upper))
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        let span = self.upper - self.lower;
        Some((span * span) / 12.0)
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (self.lower, self.upper)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        self.lower + sample_uniform_open(rng) * (self.upper - self.lower)
    }
}

/// Exponential distribution $\mathrm{Exp}(\lambda)$ on $[0, \infty)$ with rate $\lambda > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExponentialDist {
    rate: f64,
}

impl ExponentialDist {
    /// Constructs $\mathrm{Exp}(\lambda)$ with rate $\lambda > 0$.
    #[verified_engine::verified]
    pub fn new(rate: f64) -> Result<Self, DistributionError> {
        validate_positive("rate", rate)?;
        Ok(Self { rate })
    }

    /// Returns the rate parameter $\lambda$.
    #[verified_engine::verified]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Maximum likelihood estimator $\hat{\lambda} = 1 / \bar{x}$ on non-negative samples.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        if data.is_empty() {
            return Err(DistributionError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        let mut sum = 0.0;
        for &x in data {
            if !x.is_finite() || x < 0.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "exponential observations must be finite and non-negative",
                });
            }
            sum += x;
        }
        let mean = sum / (data.len() as f64);
        if mean <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "sample mean must be strictly positive",
            });
        }
        Self::new(1.0 / mean)
    }

    /// First-moment Method-of-Moments estimator $\hat{\lambda} = 1 / \bar{x}$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        Self::fit_mle(data)
    }
}

impl ContinuousDistribution for ExponentialDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else {
            self.rate * (-self.rate * x).exp()
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            f64::NEG_INFINITY
        } else {
            self.rate.ln() - self.rate * x
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            -(-self.rate * x).exp_m1()
        }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        if p == 0.0 {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Ok(f64::INFINITY);
        }
        Ok(-(-p).ln_1p() / self.rate)
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(1.0 / self.rate)
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        Some(1.0 / (self.rate * self.rate))
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        -sample_uniform_open(rng).ln() / self.rate
    }
}

// [cite:clinical_trials_statistics]
