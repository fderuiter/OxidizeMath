use super::special::{
    sample_gamma_unit_scale, sample_poisson, sample_uniform_open, validate_positive,
    validate_probability,
};
use super::traits::DiscreteDistribution;
use crate::error::DistributionError;
use oxidize_core::rng::OxidizeRng;
use statrs::function::{beta, gamma};

/// Poisson distribution $\mathrm{Poisson}(\lambda)$ on $\{0, 1, 2, \dots\}$ with rate $\lambda > 0$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoissonDist {
    lambda: f64,
}

impl PoissonDist {
    /// Constructs $\mathrm{Poisson}(\lambda)$ with `lambda > 0`.
    #[verified_engine::verified]
    pub fn new(lambda: f64) -> Result<Self, DistributionError> {
        validate_positive("lambda", lambda)?;
        Ok(Self { lambda })
    }

    /// Returns mean/rate parameter $\lambda$.
    #[verified_engine::verified]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Maximum likelihood estimator $\hat{\lambda} = \bar{k}$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[u64]) -> Result<Self, DistributionError> {
        if data.is_empty() {
            return Err(DistributionError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        let mean = data.iter().map(|&k| k as f64).sum::<f64>() / (data.len() as f64);
        if mean <= 0.0 {
            return Err(DistributionError::FittingFailed {
                reason: "Poisson sample mean must be strictly positive",
            });
        }
        Self::new(mean)
    }

    /// Method-of-moments estimator $\hat{\lambda} = \bar{k}$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[u64]) -> Result<Self, DistributionError> {
        Self::fit_mle(data)
    }
}

impl DiscreteDistribution for PoissonDist {
    #[verified_engine::verified]
    fn pmf(&self, k: u64) -> f64 {
        self.ln_pmf(k).exp()
    }

    #[verified_engine::verified]
    fn ln_pmf(&self, k: u64) -> f64 {
        let kf = k as f64;
        kf * self.lambda.ln() - self.lambda - gamma::ln_gamma(kf + 1.0)
    }

    #[verified_engine::verified]
    fn cdf(&self, k: u64) -> f64 {
        gamma::gamma_ur((k as f64) + 1.0, self.lambda)
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<u64, DistributionError> {
        validate_probability(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        if p >= 1.0 {
            return Err(DistributionError::InvalidProbability { value: p });
        }
        invert_unbounded_discrete_cdf(p, self.lambda.max(1.0), |k| self.cdf(k))
    }

    #[verified_engine::verified]
    fn mean(&self) -> f64 {
        self.lambda
    }

    #[verified_engine::verified]
    fn variance(&self) -> f64 {
        self.lambda
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> u64 {
        sample_poisson(self.lambda, rng)
    }
}

/// Geometric distribution on $\{0, 1, 2, \dots\}$ counting failures prior to the first success
/// with success probability $p \in (0, 1]$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricDist {
    p: f64,
}

impl GeometricDist {
    /// Constructs $\mathrm{Geometric}(p)$ verifying $p \in (0, 1]$.
    #[verified_engine::verified]
    pub fn new(p: f64) -> Result<Self, DistributionError> {
        if !p.is_finite() || p <= 0.0 || p > 1.0 {
            return Err(DistributionError::InvalidParameter {
                name: "p",
                value: p,
            });
        }
        Ok(Self { p })
    }

    /// Returns success probability $p$.
    #[verified_engine::verified]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Maximum likelihood estimator $\hat{p} = \frac{1}{1 + \bar{k}}$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[u64]) -> Result<Self, DistributionError> {
        if data.is_empty() {
            return Err(DistributionError::InsufficientData {
                required: 1,
                actual: 0,
            });
        }
        let mean = data.iter().map(|&k| k as f64).sum::<f64>() / (data.len() as f64);
        Self::new(1.0 / (1.0 + mean))
    }

    /// Method-of-moments estimator $\hat{p} = \frac{1}{1 + \bar{k}}$.
    #[verified_engine::verified]
    pub fn fit_mom(data: &[u64]) -> Result<Self, DistributionError> {
        Self::fit_mle(data)
    }
}

impl DiscreteDistribution for GeometricDist {
    #[verified_engine::verified]
    fn pmf(&self, k: u64) -> f64 {
        if (self.p - 1.0).abs() < 1e-15 {
            return if k == 0 { 1.0 } else { 0.0 };
        }
        self.ln_pmf(k).exp()
    }

    #[verified_engine::verified]
    fn ln_pmf(&self, k: u64) -> f64 {
        if (self.p - 1.0).abs() < 1e-15 {
            return if k == 0 { 0.0 } else { f64::NEG_INFINITY };
        }
        self.p.ln() + (k as f64) * (-self.p).ln_1p()
    }

    #[verified_engine::verified]
    fn cdf(&self, k: u64) -> f64 {
        if (self.p - 1.0).abs() < 1e-15 {
            return 1.0;
        }
        -(((k as f64) + 1.0) * (-self.p).ln_1p()).exp_m1()
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<u64, DistributionError> {
        validate_probability(p)?;
        if p >= 1.0 {
            if (self.p - 1.0).abs() < 1e-15 {
                return Ok(0);
            }
            return Err(DistributionError::InvalidProbability { value: p });
        }
        if p <= self.p || (self.p - 1.0).abs() < 1e-15 {
            return Ok(0);
        }
        let raw = ((-p).ln_1p() / (-self.p).ln_1p()).ceil() - 1.0;
        let mut k = raw.max(0.0) as u64;
        while k > 0 && self.cdf(k - 1) >= p {
            k -= 1;
        }
        while self.cdf(k) < p {
            k = k.saturating_add(1);
        }
        Ok(k)
    }

    #[verified_engine::verified]
    fn mean(&self) -> f64 {
        (1.0 - self.p) / self.p
    }

    #[verified_engine::verified]
    fn variance(&self) -> f64 {
        (1.0 - self.p) / (self.p * self.p)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> u64 {
        if (self.p - 1.0).abs() < 1e-15 {
            return 0;
        }
        let u = sample_uniform_open(rng);
        (u.ln() / (-self.p).ln_1p()).floor().max(0.0) as u64
    }
}

/// Negative binomial distribution $\mathrm{NegBin}(r, p)$ on $\{0, 1, 2, \dots\}$ counting failures
/// before $r > 0$ successes with success probability $p \in (0, 1)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NegativeBinomialDist {
    r: f64,
    p: f64,
}

impl NegativeBinomialDist {
    /// Constructs $\mathrm{NegBin}(r, p)$ verifying `r > 0` and $p \in (0, 1)$.
    #[verified_engine::verified]
    pub fn new(r: f64, p: f64) -> Result<Self, DistributionError> {
        validate_positive("r", r)?;
        if !p.is_finite() || p <= 0.0 || p >= 1.0 {
            return Err(DistributionError::InvalidParameter {
                name: "p",
                value: p,
            });
        }
        Ok(Self { r, p })
    }

    /// Returns dispersion / target-success parameter $r$.
    #[verified_engine::verified]
    pub fn r(&self) -> f64 {
        self.r
    }

    /// Returns success probability $p$.
    #[verified_engine::verified]
    pub fn p(&self) -> f64 {
        self.p
    }

    /// Method-of-moments estimator on overdispersed count data ($s^2 > \bar{k} > 0$).
    #[verified_engine::verified]
    pub fn fit_mom(data: &[u64]) -> Result<Self, DistributionError> {
        if data.len() < 2 {
            return Err(DistributionError::InsufficientData {
                required: 2,
                actual: data.len(),
            });
        }
        let n = data.len() as f64;
        let mean = data.iter().map(|&k| k as f64).sum::<f64>() / n;
        let var = data
            .iter()
            .map(|&k| {
                let d = (k as f64) - mean;
                d * d
            })
            .sum::<f64>()
            / (n - 1.0);
        if mean <= 0.0 || var <= mean {
            return Err(DistributionError::FittingFailed {
                reason: "negative binomial MoM requires overdispersion (variance > mean > 0)",
            });
        }
        let p = (mean / var).clamp(1e-8, 1.0 - 1e-8);
        let r = (mean * mean) / (var - mean);
        Self::new(r, p)
    }

    /// Profile maximum likelihood estimator solving $\sum_i (\psi(k_i + r) - \psi(r)) + n \ln\frac{r}{r + \bar{k}} = 0$.
    #[verified_engine::verified]
    pub fn fit_mle(data: &[u64]) -> Result<Self, DistributionError> {
        let mom = Self::fit_mom(data)?;
        let mean = data.iter().map(|&k| k as f64).sum::<f64>() / (data.len() as f64);
        let mut lo = (mom.r * 0.05).max(1e-4);
        let mut hi = (mom.r * 20.0).max(50.0);
        while negbin_profile_score(data, mean, lo) < 0.0 && lo > 1e-6 {
            lo *= 0.5;
        }
        while negbin_profile_score(data, mean, hi) > 0.0 && hi < 1e6 {
            hi *= 2.0;
        }
        for _ in 0..64 {
            let mid = 0.5 * (lo + hi);
            if negbin_profile_score(data, mean, mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let r = 0.5 * (lo + hi);
        let p = (r / (r + mean)).clamp(1e-8, 1.0 - 1e-8);
        Self::new(r, p)
    }
}

fn negbin_profile_score(data: &[u64], mean: f64, r: f64) -> f64 {
    let n = data.len() as f64;
    let psi_r = gamma::digamma(r);
    let digamma_sum: f64 = data
        .iter()
        .map(|&k| gamma::digamma((k as f64) + r) - psi_r)
        .sum();
    digamma_sum + n * (r / (r + mean)).ln()
}

fn invert_unbounded_discrete_cdf<FCdf>(
    p: f64,
    scale_hint: f64,
    cdf_fn: FCdf,
) -> Result<u64, DistributionError>
where
    FCdf: Fn(u64) -> f64,
{
    let mut lo = 0_u64;
    let mut hi = (scale_hint.ceil() as u64).max(4);
    while cdf_fn(hi) < p {
        lo = hi + 1;
        hi = hi.saturating_mul(2);
        if hi >= (1_u64 << 50) {
            break;
        }
    }
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf_fn(mid) >= p {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    Ok(lo)
}

impl DiscreteDistribution for NegativeBinomialDist {
    #[verified_engine::verified]
    fn pmf(&self, k: u64) -> f64 {
        self.ln_pmf(k).exp()
    }

    #[verified_engine::verified]
    fn ln_pmf(&self, k: u64) -> f64 {
        let kf = k as f64;
        gamma::ln_gamma(kf + self.r) - gamma::ln_gamma(kf + 1.0) - gamma::ln_gamma(self.r)
            + self.r * self.p.ln()
            + kf * (-self.p).ln_1p()
    }

    #[verified_engine::verified]
    fn cdf(&self, k: u64) -> f64 {
        beta::beta_reg(self.r, (k as f64) + 1.0, self.p)
    }

    #[verified_engine::verified]
    fn quantile(&self, p: f64) -> Result<u64, DistributionError> {
        validate_probability(p)?;
        if p == 0.0 {
            return Ok(0);
        }
        if p >= 1.0 {
            return Err(DistributionError::InvalidProbability { value: p });
        }
        invert_unbounded_discrete_cdf(p, self.mean().max(2.0), |k| self.cdf(k))
    }

    #[verified_engine::verified]
    fn mean(&self) -> f64 {
        self.r * (1.0 - self.p) / self.p
    }

    #[verified_engine::verified]
    fn variance(&self) -> f64 {
        self.r * (1.0 - self.p) / (self.p * self.p)
    }

    #[verified_engine::verified]
    fn sample(&self, rng: &mut OxidizeRng) -> u64 {
        let rate = self.p / (1.0 - self.p);
        let lambda = sample_gamma_unit_scale(self.r, rng) / rate;
        sample_poisson(lambda, rng)
    }
}

// [cite:clinical_trials_statistics]
