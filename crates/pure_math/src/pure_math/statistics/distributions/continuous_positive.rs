use super::special::{
    sample_mean_and_var, sample_standard_normal, sample_uniform_open, standard_normal_cdf,
    standard_normal_ln_pdf, standard_normal_pdf, standard_normal_quantile, validate_positive,
    validate_probability,
};
use super::traits::ContinuousDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use statrs::function::gamma;

/// Log-normal distribution $\ln X \sim \mathcal{N}(\mu, \sigma^2)$ supported on $(0, \infty)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalDist {
    mu: f64,
    sigma: f64,
}

impl LogNormalDist {
    /// Constructs a log-normal model with finite log-location `mu` and positive log-scale `sigma`.
    #[verified_engine::verified]
    pub fn new(mu: f64, sigma: f64) -> Result<Self, DistributionError> {
        if !mu.is_finite() {
            return Err(DistributionError::InvalidParameter {
                name: "mu",
                value: mu,
            });
        }
        validate_positive("sigma", sigma)?;
        Ok(Self { mu, sigma })
    }

    /// Returns log-location $\mu$.
    #[verified_engine::verified]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns log-scale $\sigma$.
    #[verified_engine::verified]
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    /// Exact maximum likelihood estimator from $\ln x_i$ on strictly positive observations.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        if data.len() < 2 {
            return Err(DistributionError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }
        let n = data.len() as f64;
        let mut sum_ln = 0.0;
        for &x in data {
            if !x.is_finite() || x <= 0.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "log-normal observations must be strictly positive",
                });
            }
            sum_ln += x.ln();
        }
        let mu = sum_ln / n;
        let mut sq = 0.0;
        for &x in data {
            let d = x.ln() - mu;
            sq += d * d;
        }
        let var = sq / n;
        if var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "zero log-variance",
            });
        }
        Self::new(mu, var.sqrt())
    }

    /// Method-of-moments estimator matching $E(X) = \bar{x}$ and $\mathrm{Var}(X) = s^2$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        for &x in data {
            if !x.is_finite() || x <= 0.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "log-normal observations must be strictly positive",
                });
            }
        }
        let (mean, _, sample_var) = sample_mean_and_var(data)?;
        if mean <= 0.0 || sample_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "degenerate sample moments",
            });
        }
        let sigma2 = (sample_var / (mean * mean)).ln_1p();
        let mu = mean.ln() - 0.5 * sigma2;
        Self::new(mu, sigma2.sqrt())
    }
}

impl ContinuousDistribution for LogNormalDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            standard_normal_pdf((x.ln() - self.mu) / self.sigma) / (x * self.sigma)
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            f64::NEG_INFINITY
        } else {
            standard_normal_ln_pdf((x.ln() - self.mu) / self.sigma) - x.ln() - self.sigma.ln()
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            standard_normal_cdf((x.ln() - self.mu) / self.sigma)
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
        Ok((self.mu + self.sigma * standard_normal_quantile(p)).exp())
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some((self.mu + 0.5 * self.sigma * self.sigma).exp())
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        let s2 = self.sigma * self.sigma;
        Some(s2.exp_m1() * (2.0 * self.mu + s2).exp())
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        (self.mu + self.sigma * sample_standard_normal(rng)).exp()
    }
}

/// Two-parameter Weibull distribution on $[0, \infty)$ with shape $k > 0$ and scale $\lambda > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeibullDist {
    shape: f64,
    scale: f64,
}

impl WeibullDist {
    /// Constructs $\mathrm{Weibull}(k, \lambda)$ with `shape > 0` and `scale > 0`.
    #[verified_engine::verified]
    pub fn new(shape: f64, scale: f64) -> Result<Self, DistributionError> {
        validate_positive("shape", shape)?;
        validate_positive("scale", scale)?;
        Ok(Self { shape, scale })
    }

    /// Returns shape parameter $k$.
    #[verified_engine::verified]
    pub fn shape(&self) -> f64 {
        self.shape
    }

    /// Returns scale parameter $\lambda$.
    #[verified_engine::verified]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Method-of-moments estimator inverting squared coefficient of variation via bisection.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[f64]) -> Result<Self, DistributionError> {
        for &x in data {
            if !x.is_finite() || x <= 0.0 {
                return Err(DistributionError::FittingFailed {
                    reason: "Weibull observations must be strictly positive",
                });
            }
        }
        let (mean, _, sample_var) = sample_mean_and_var(data)?;
        if mean <= 0.0 || sample_var <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "degenerate sample moments",
            });
        }
        let target_cv2 = sample_var / (mean * mean);
        let mut lo = 0.05_f64;
        let mut hi = 200.0_f64;
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            let g1 = gamma::gamma(1.0 + 1.0 / mid);
            let g2 = gamma::gamma(1.0 + 2.0 / mid);
            let cv2 = g2 / (g1 * g1) - 1.0;
            if cv2 > target_cv2 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let shape = 0.5 * (lo + hi);
        let scale = mean / gamma::gamma(1.0 + 1.0 / shape);
        Self::new(shape, scale)
    }

    /// Profile maximum likelihood estimator solving $\frac{1}{k} + \overline{\ln x} - \frac{\sum x_i^k \ln x_i}{\sum x_i^k} = 0$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[f64]) -> Result<Self, DistributionError> {
        let mom = Self::fit_mom(data)?;
        let mean_ln = data.iter().map(|&x| x.ln()).sum::<f64>() / (data.len() as f64);
        let mut lo = (mom.shape * 0.1).max(0.05);
        let mut hi = (mom.shape * 10.0).min(500.0);
        while weibull_profile_score(data, mean_ln, lo) < 0.0 && lo > 1e-3 {
            lo *= 0.5;
        }
        while weibull_profile_score(data, mean_ln, hi) > 0.0 && hi < 1e4 {
            hi *= 2.0;
        }
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if weibull_profile_score(data, mean_ln, mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let k = 0.5 * (lo + hi);
        let mean_pow = data.iter().map(|&x| x.powf(k)).sum::<f64>() / (data.len() as f64);
        Self::new(k, mean_pow.powf(1.0 / k))
    }
}

fn weibull_profile_score(data: &[f64], mean_ln: f64, k: f64) -> f64 {
    let mut sum_xk = 0.0;
    let mut sum_xk_ln = 0.0;
    for &x in data {
        let xk = x.powf(k);
        sum_xk += xk;
        sum_xk_ln += xk * x.ln();
    }
    1.0 / k + mean_ln - sum_xk_ln / sum_xk
}

impl ContinuousDistribution for WeibullDist {
    #[verified_engine::verified]
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 {
            0.0
        } else if x == 0.0 {
            if (self.shape - 1.0).abs() < 1e-15 {
                1.0 / self.scale
            } else if self.shape > 1.0 {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            let u = x / self.scale;
            (self.shape / self.scale) * u.powf(self.shape - 1.0) * (-u.powf(self.shape)).exp()
        }
    }

    #[verified_engine::verified]
    fn ln_pdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            if x == 0.0 && (self.shape - 1.0).abs() < 1e-15 {
                -self.scale.ln()
            } else {
                f64::NEG_INFINITY
            }
        } else {
            let u = x / self.scale;
            self.shape.ln() - self.scale.ln() + (self.shape - 1.0) * u.ln() - u.powf(self.shape)
        }
    }

    #[verified_engine::verified]
    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 {
            0.0
        } else {
            -(-(x / self.scale).powf(self.shape)).exp_m1()
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
        Ok(self.scale * (-(-p).ln_1p()).powf(1.0 / self.shape))
    }

    #[verified_engine::verified]
    fn mean(&self) -> Option<f64> {
        Some(self.scale * gamma::gamma(1.0 + 1.0 / self.shape))
    }

    #[verified_engine::verified]
    fn variance(&self) -> Option<f64> {
        let g1 = gamma::gamma(1.0 + 1.0 / self.shape);
        let g2 = gamma::gamma(1.0 + 2.0 / self.shape);
        Some(self.scale * self.scale * (g2 - g1 * g1))
    }

    #[verified_engine::verified]
    fn support(&self) -> (f64, f64) {
        (0.0, f64::INFINITY)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> f64 {
        self.scale * (-sample_uniform_open(rng).ln()).powf(1.0 / self.shape)
    }
}

// [cite:clinical_trials_statistics]
