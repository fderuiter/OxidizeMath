//! Gaussian conjugate families: [`NormalNormal`] (known variance) and [`NormalInverseGamma`] (unknown mean & variance).

use crate::distributions::{ContinuousDist, Gamma, Normal, StudentsT};
use crate::error::ProbError;
use crate::special::ln_gamma;
use core::f64::consts::TAU;

fn validate_credible_level(level: f64) -> Result<(f64, f64), ProbError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(ProbError::InvalidProbability { value: level });
    }
    let tail = 0.5 * (1.0 - level);
    Ok((tail, 1.0 - tail))
}

fn welford_sufficient_stats(xs: &[f64]) -> Result<(usize, f64, f64), ProbError> {
    let mut n = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for &x in xs {
        if !x.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "observation",
                value: x,
            });
        }
        n += 1;
        let delta = x - mean;
        mean += delta / (n as f64);
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    Ok((n, mean, m2))
}

/// Conjugate Normal-Normal model with known observation standard deviation $\sigma > 0$
/// and Gaussian prior $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalNormal {
    mean: f64,
    std_dev: f64,
    obs_std_dev: f64,
}

impl NormalNormal {
    /// Initializes $\mu \sim \mathcal{N}(\text{prior\_mean}, \text{prior\_std}^2)$ for likelihood std dev `obs_std_dev > 0`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(prior_mean: f64, prior_std: f64, obs_std_dev: f64) -> Result<Self, ProbError> {
        let _ = Normal::new(prior_mean, prior_std)?;
        if !obs_std_dev.is_finite() || obs_std_dev <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "obs_std_dev",
                value: obs_std_dev,
            });
        }
        debug_assert!(prior_std > 0.0 && obs_std_dev > 0.0);
        Ok(Self {
            mean: prior_mean,
            std_dev: prior_std,
            obs_std_dev,
        })
    }

    /// Posterior mean $\mu_n$.
    #[must_use]
    #[inline]
    pub const fn posterior_mean(&self) -> f64 {
        self.mean
    }

    /// Posterior standard deviation $\tau_n$.
    #[must_use]
    #[inline]
    pub const fn posterior_std(&self) -> f64 {
        self.std_dev
    }

    /// Posterior variance $\tau_n^2$.
    #[must_use]
    #[inline]
    pub fn posterior_variance(&self) -> f64 {
        self.std_dev * self.std_dev
    }

    /// Known observation standard deviation $\sigma$.
    #[must_use]
    #[inline]
    pub const fn obs_std_dev(&self) -> f64 {
        self.obs_std_dev
    }

    /// Updates $(\mu, \tau)$ from sufficient statistics `(n, sample_mean)` with zero heap allocations.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update_summary(&self, n: usize, sample_mean: f64) -> Result<Self, ProbError> {
        if n == 0 {
            return Ok(*self);
        }
        if !sample_mean.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "sample_mean",
                value: sample_mean,
            });
        }
        let prior_prec = 1.0 / (self.std_dev * self.std_dev);
        let data_prec = (n as f64) / (self.obs_std_dev * self.obs_std_dev);
        let post_prec = prior_prec + data_prec;
        let post_mean = (prior_prec * self.mean + data_prec * sample_mean) / post_prec;
        let post_std = libm::sqrt(1.0 / post_prec);
        debug_assert!(post_std <= self.std_dev + 1.0e-15);
        Ok(Self {
            mean: post_mean,
            std_dev: post_std,
            obs_std_dev: self.obs_std_dev,
        })
    }

    /// Updates $(\mu, \tau)$ directly from a slice of observations `xs` with zero heap allocations.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update_slice(&self, xs: &[f64]) -> Result<Self, ProbError> {
        let (n, sample_mean, _) = welford_sufficient_stats(xs)?;
        self.update_summary(n, sample_mean)
    }

    /// Exact posterior [`Normal`] distribution for $\mu$.
    #[inline]
    pub fn posterior(&self) -> Result<Normal, ProbError> {
        Normal::new(self.mean, self.std_dev)
    }

    /// Posterior predictive [`Normal`] distribution $\mathcal{N}(\mu_n, \tau_n^2 + \sigma^2)$ for $X_{\text{new}}$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn posterior_predictive(&self) -> Result<Normal, ProbError> {
        let pred_std =
            libm::sqrt(self.std_dev * self.std_dev + self.obs_std_dev * self.obs_std_dev);
        Normal::new(self.mean, pred_std)
    }

    /// Exact equal-tailed credible interval for $\mu$ at `level` $\in (0, 1)$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        let (lo_p, hi_p) = validate_credible_level(level)?;
        let post = self.posterior()?;
        Ok((post.quantile(lo_p)?, post.quantile(hi_p)?))
    }

    /// Exact log marginal likelihood $\ln p(\mathbf{x})$ across `xs`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn log_marginal_likelihood(&self, xs: &[f64]) -> Result<f64, ProbError> {
        let (n, x_bar, s_xx) = welford_sufficient_stats(xs)?;
        if n == 0 {
            return Ok(0.0);
        }
        let nf = n as f64;
        let sigma2 = self.obs_std_dev * self.obs_std_dev;
        let tau2 = self.std_dev * self.std_dev;
        let marginal_mean_var = sigma2 + nf * tau2;
        let diff = x_bar - self.mean;
        let lml = -0.5 * nf * libm::log(TAU)
            - 0.5 * (nf - 1.0) * libm::log(sigma2)
            - 0.5 * libm::log(marginal_mean_var)
            - 0.5 * s_xx / sigma2
            - 0.5 * nf * diff * diff / marginal_mean_var;
        debug_assert!(lml.is_finite());
        Ok(lml)
    }

    /// Exact Bayes factor $BF_{10}$ relative to `baseline` across `xs`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn bayes_factor(&self, baseline: &Self, xs: &[f64]) -> Result<f64, ProbError> {
        let l1 = self.log_marginal_likelihood(xs)?;
        let l0 = baseline.log_marginal_likelihood(xs)?;
        Ok(libm::exp(l1 - l0))
    }
}

/// Conjugate Normal-Inverse-Gamma model for unknown $(\mu, \sigma^2)$:
/// $\sigma^2 \sim \operatorname{InvGamma}(\alpha, \beta)$, $\mu \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2 / \kappa)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalInverseGamma {
    mu: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
}

impl NormalInverseGamma {
    /// Initializes $\operatorname{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$ with $\kappa_0 > 0, \alpha_0 > 0, \beta_0 > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(mu: f64, kappa: f64, alpha: f64, beta: f64) -> Result<Self, ProbError> {
        if !mu.is_finite() {
            return Err(ProbError::InvalidParameter {
                name: "mu",
                value: mu,
            });
        }
        if !kappa.is_finite() || kappa <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "kappa",
                value: kappa,
            });
        }
        let _ = Gamma::new(alpha, beta)?;
        debug_assert!(kappa > 0.0 && alpha > 0.0 && beta > 0.0);
        Ok(Self {
            mu,
            kappa,
            alpha,
            beta,
        })
    }

    /// Location hyperparameter $\mu_n$.
    #[must_use]
    #[inline]
    pub const fn mu(&self) -> f64 {
        self.mu
    }

    /// Precision pseudo-count $\kappa_n$.
    #[must_use]
    #[inline]
    pub const fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Inverse-Gamma shape $\alpha_n$.
    #[must_use]
    #[inline]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Inverse-Gamma scale $\beta_n$.
    #[must_use]
    #[inline]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Posterior expectation $\mathbb{E}\[\mu\] = \mu_n$.
    #[must_use]
    #[inline]
    pub const fn posterior_mean(&self) -> f64 {
        self.mu
    }

    /// Posterior expectation $\mathbb{E}\[\sigma^2\] = \frac{\beta_n}{\alpha_n - 1}$ (finite for $\alpha_n > 1$).
    #[must_use]
    #[inline]
    pub fn expected_obs_variance(&self) -> f64 {
        if self.alpha > 1.0 {
            self.beta / (self.alpha - 1.0)
        } else {
            f64::INFINITY
        }
    }

    /// Updates $(\mu, \kappa, \alpha, \beta)$ via single-pass Welford sufficient statistics (zero heap allocations).
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update_slice(&self, xs: &[f64]) -> Result<Self, ProbError> {
        let (n, x_bar, s_xx) = welford_sufficient_stats(xs)?;
        if n == 0 {
            return Ok(*self);
        }
        let nf = n as f64;
        let kappa_n = self.kappa + nf;
        let mu_n = (self.kappa * self.mu + nf * x_bar) / kappa_n;
        let alpha_n = self.alpha + 0.5 * nf;
        let diff = x_bar - self.mu;
        let beta_n = self.beta + 0.5 * s_xx + 0.5 * (self.kappa * nf * diff * diff) / kappa_n;
        debug_assert!(kappa_n > self.kappa && alpha_n > self.alpha);
        Ok(Self {
            mu: mu_n,
            kappa: kappa_n,
            alpha: alpha_n,
            beta: beta_n,
        })
    }

    /// Exact equal-tailed marginal Student-$t$ credible interval for $\mu$ at `level` $\in (0, 1)$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn mean_credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        let (lo_p, hi_p) = validate_credible_level(level)?;
        let t_dist = StudentsT::new(2.0 * self.alpha)?;
        let scale = libm::sqrt(self.beta / (self.alpha * self.kappa));
        Ok((
            self.mu + scale * t_dist.quantile(lo_p)?,
            self.mu + scale * t_dist.quantile(hi_p)?,
        ))
    }

    /// Exact equal-tailed marginal Inverse-Gamma credible interval for $\sigma^2$ at `level` $\in (0, 1)$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn variance_credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        let (lo_p, hi_p) = validate_credible_level(level)?;
        let prec_gamma = Gamma::new(self.alpha, self.beta)?;
        let prec_hi = prec_gamma.quantile(hi_p)?;
        let prec_lo = prec_gamma.quantile(lo_p)?;
        Ok((1.0 / prec_hi, 1.0 / prec_lo))
    }

    /// Posterior predictive Student-$t$ log-density $\ln p(x_{\text{new}})$ at `x`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn predictive_ln_pdf(&self, x: f64) -> Result<f64, ProbError> {
        let t_dist = StudentsT::new(2.0 * self.alpha)?;
        let scale = libm::sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa));
        Ok(t_dist.ln_pdf((x - self.mu) / scale) - libm::log(scale))
    }

    /// Exact closed-form log marginal likelihood $\ln p(\mathbf{x})$ across `xs`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn log_marginal_likelihood(&self, xs: &[f64]) -> Result<f64, ProbError> {
        if xs.is_empty() {
            return Ok(0.0);
        }
        let post = self.update_slice(xs)?;
        let nf = xs.len() as f64;
        let lml = ln_gamma(post.alpha) - ln_gamma(self.alpha) + self.alpha * libm::log(self.beta)
            - post.alpha * libm::log(post.beta)
            + 0.5 * (libm::log(self.kappa) - libm::log(post.kappa))
            - 0.5 * nf * libm::log(TAU);
        debug_assert!(lml.is_finite());
        Ok(lml)
    }

    /// Exact Bayes factor $BF_{10}$ relative to `baseline` across `xs`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn bayes_factor(&self, baseline: &Self, xs: &[f64]) -> Result<f64, ProbError> {
        let l1 = self.log_marginal_likelihood(xs)?;
        let l0 = baseline.log_marginal_likelihood(xs)?;
        Ok(libm::exp(l1 - l0))
    }
}
