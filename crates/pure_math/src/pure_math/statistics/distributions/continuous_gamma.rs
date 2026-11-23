use super::special::{
    gamma_unit_quantile, sample_gamma_unit_scale, sample_mean_and_var, validate_positive,
    validate_probability,
};
use super::traits::ContinuousDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use statrs::function::gamma;

/// Gamma distribution $\mathrm{Gamma}(\alpha, \beta)$ parameterized by shape $\alpha > 0$ and rate $\beta > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaDist {
    shape: f64,
    rate: f64,
}

impl GammaDist {
    /// Constructs $\mathrm{Gamma}(\alpha, \beta)$ with `shape > 0` and `rate > 0`.
    #[verified_engine::verified]
    pub fn new(shape: f64, rate: f64) -> Result<Self, DistributionError> {
        validate_positive("shape", shape)?;
        validate_positive("rate", rate)?;
        Ok(Self { shape, rate })
    }

    /// Constructs a Gamma distribution from `shape` $\alpha > 0$ and `scale` $\theta = 1 / \beta > 0$.
    #[verified_engine::verified]
    pub fn from_shape_scale(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        validate_positive("shape", shape)?;
        validate_positive("scale", scale)?;
        Ok(Self {
            shape,
            rate: 1.0 / scale,
        })
    }

    /// Returns shape $\alpha$.
    #[verified_engine::verified]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns rate $\beta$.
    #[verified_engine::verified]
    pub fn rate(&self) -> f64 {
        self.rate
    }

    /// Returns scale $\theta = 1 / \beta$.
    #[verified_engine::verified]
    pub fn scale(&self) -> f64 {
        1.0 / self.rate
    }

    /// Method-of-moments estimator $\hat{\alpha} = \bar{x}^2 / s^2$, $\hat{\beta} = \bar{x} / s^2$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        for &x in data {
            if !x.is_finite() || x <= 0.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "Gamma observations must be strictly positive",
                });
            }
        }
        let (mean, _, sample_var) = sample_mean_and_var(data)?;
        if mean <= 0.0 || sample_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "degenerate sample moments",
            });
        }
        Self::new((mean * mean) / sample_var, mean / sample_var)
    }

    /// Maximum likelihood estimator solving $\ln \alpha - \psi(\alpha) = \ln \bar{x} - \overline{\ln x}$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        let mom = Self::fit_mom(data)?;
        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let mean_ln = data.iter().map(|&x| x.ln()).sum::<f64>() / n;
        let s = mean.ln() - mean_ln;
        if s <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "degenerate constant sample",
            });
        }
        let mut lo = (mom.shape * 0.05).max(1e-4);
        let mut hi = (mom.shape * 20.0).max(10.0);
        while (lo.ln() - gamma::digamma(lo)) < s && lo > 1e-8 {
            lo *= 0.5;
        }
        while (hi.ln() - gamma::digamma(hi)) > s && hi < 1e8 {
            hi *= 2.0;
        }
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if mid.ln() - gamma::digamma(mid) > s {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let alpha = 0.5 * (lo + hi);
        Self::new(alpha, alpha / mean)
    }
}

impl ContinuousDistribution for GammaDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            if x == 0.0 && (self.shape - 1.0).abs() < 1e-15 {
                self.rate
            } else {
                0.0
            }
        } else {
            self.ln_pdf(x).exp()
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            if x == 0.0 && (self.shape - 1.0).abs() < 1e-15 {
                self.rate.ln()
            } else {
                f64::NEG_INFINITY
            }
        } else {
            self.shape * self.rate.ln() - gamma::ln_gamma(self.shape) + (self.shape - 1.0) * x.ln()
                - self.rate * x
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            gamma::gamma_lr(self.shape, self.rate * x)
        }
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<f64, DistributionError> {
        validate_probability(p)?;
        Ok(gamma_unit_quantile(self.shape, p) / self.rate)
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(self.shape / self.rate)
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        Some(self.shape / (self.rate * self.rate))
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        sample_gamma_unit_scale(self.shape, rng) / self.rate
    }
}

// [cite:clinical_trials_statistics]
