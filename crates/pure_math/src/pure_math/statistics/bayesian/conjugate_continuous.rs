use super::intervals_and_bf::{
    BayesFactor, CredibleInterval, equal_tailed_credible_interval, validate_hyperparam,
};
use crate::error::BayesianError;
use crate::pure_math::statistics::distributions::{NormalDist, StudentTDist};
use statrs::function::gamma;
use std::f64::consts::{PI, TAU};

/// Conjugate Normal-Normal model with known observation variance $\sigma^2 > 0$ and Gaussian prior
/// $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalNormal {
    mu: f64,
    tau_sq: f64,
    sigma_sq: f64,
}

impl NormalNormal {
    /// Constructs a Normal-Normal model with prior mean `mu`, prior variance `tau_sq > 0`, and
    /// known likelihood variance `sigma_sq > 0`.
    #[verified_engine::verified]
    pub fn new(mu: f64, tau_sq: f64, sigma_sq: f64) -> Result<Self, BayesianError> {
        if !mu.is_finite() {
            return Err(BayesianError::InvalidHyperparameter {
                name: "mu",
                value: mu,
            });
        }
        validate_hyperparam("tau_sq", tau_sq)?;
        validate_hyperparam("sigma_sq", sigma_sq)?;
        Ok(Self {
            mu,
            tau_sq,
            sigma_sq,
        })
    }

    /// Returns current mean $\mu_n$.
    #[verified_engine::verified]
    pub fn posterior_mean(&self) -> f64 {
        self.mu
    }

    /// Returns current variance $\tau_n^2$.
    #[verified_engine::verified]
    pub fn posterior_variance(&self) -> f64 {
        self.tau_sq
    }

    /// Returns the known observation variance $\sigma^2$.
    #[verified_engine::verified]
    pub fn known_sigma_sq(&self) -> f64 {
        self.sigma_sq
    }

    /// Conjugate precision-weighted update from sufficient statistics `(sample_mean, n)`.
    #[verified_engine::verified]
    pub fn update_summary(&self, sample_mean: f64, n: usize) -> Result<Self, BayesianError> {
        if n == 0 {
            return Ok(*self);
        }
        if !sample_mean.is_finite() {
            return Err(BayesianError::InvalidObservation {
                reason: "sample_mean must be finite",
            });
        }
        let prior_prec = 1.0 / self.tau_sq;
        let data_prec = (n as f64) / self.sigma_sq;
        let post_prec = prior_prec + data_prec;
        let post_mu = (prior_prec * self.mu + data_prec * sample_mean) / post_prec;
        Self::new(post_mu, 1.0 / post_prec, self.sigma_sq)
    }

    /// Conjugate update from a slice of real observations.
    #[verified_engine::verified]
    pub fn update_slice(&self, data: &[f64]) -> Result<Self, BayesianError> {
        if data.is_empty() {
            return Ok(*self);
        }
        let mut sum = 0.0;
        for &x in data {
            if !x.is_finite() {
                return Err(BayesianError::InvalidObservation {
                    reason: "Normal observations must be finite",
                });
            }
            sum += x;
        }
        self.update_summary(sum / (data.len() as f64), data.len())
    }

    /// Returns the exact [`NormalDist`] posterior distribution $\mathcal{N}(\mu_n, \tau_n^2)$.
    #[verified_engine::verified]
    pub fn posterior_dist(&self) -> Result<NormalDist, BayesianError> {
        NormalDist::new(self.mu, self.tau_sq.sqrt()).map_err(|_| {
            BayesianError::NumericalInstability {
                reason: "invalid posterior Normal parameters",
            }
        })
    }

    /// Returns the posterior predictive distribution $\mathcal{N}(\mu_n, \tau_n^2 + \sigma^2)$.
    #[verified_engine::verified]
    pub fn predictive_dist(&self) -> Result<NormalDist, BayesianError> {
        NormalDist::new(self.mu, (self.tau_sq + self.sigma_sq).sqrt()).map_err(|_| {
            BayesianError::NumericalInstability {
                reason: "invalid predictive Normal parameters",
            }
        })
    }

    /// Equal-tailed (and HPD, by symmetry) credible interval at mass `level`.
    #[verified_engine::verified]
    pub fn credible_interval(&self, level: f64) -> Result<CredibleInterval, BayesianError> {
        equal_tailed_credible_interval(&self.posterior_dist()?, level)
    }

    /// Exact Bayes factor $BF_{10}$ comparing $M_1: \mu \sim \mathcal{N}(\mu_0, \tau_0^2)$ vs $M_0: \mu = \mu_{\text{null}}$.
    #[verified_engine::verified]
    pub fn bayes_factor_vs_point(
        &self,
        null_mu: f64,
        data: &[f64],
    ) -> Result<BayesFactor, BayesianError> {
        if !null_mu.is_finite() {
            return Err(BayesianError::InvalidHyperparameter {
                name: "null_mu",
                value: null_mu,
            });
        }
        if data.is_empty() {
            return BayesFactor::from_ln_marginals(0.0, 0.0);
        }
        let mut sum = 0.0;
        for &x in data {
            if !x.is_finite() {
                return Err(BayesianError::InvalidObservation {
                    reason: "Normal observations must be finite",
                });
            }
            sum += x;
        }
        let n = data.len() as f64;
        let x_bar = sum / n;
        let var_bar = self.sigma_sq / n;
        let var_m1 = self.tau_sq + var_bar;
        let ln_m1 = -0.5 * ((TAU * var_m1).ln() + (x_bar - self.mu).powi(2) / var_m1);
        let ln_m0 = -0.5 * ((TAU * var_bar).ln() + (x_bar - null_mu).powi(2) / var_bar);
        BayesFactor::from_ln_marginals(ln_m1, ln_m0)
    }
}

/// Conjugate Normal-Inverse-Gamma model $(\mu, \sigma^2) \sim \mathrm{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$
/// for Gaussian data with both unknown mean and unknown variance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalInverseGamma {
    mu: f64,
    kappa: f64,
    alpha: f64,
    beta: f64,
}

impl NormalInverseGamma {
    /// Constructs $\mathrm{NIG}(\mu_0, \kappa_0, \alpha_0, \beta_0)$ with finite `mu` and positive
    /// `kappa`, `alpha`, `beta`.
    #[verified_engine::verified]
    pub fn new(mu: f64, kappa: f64, alpha: f64, beta: f64) -> Result<Self, BayesianError> {
        if !mu.is_finite() {
            return Err(BayesianError::InvalidHyperparameter {
                name: "mu",
                value: mu,
            });
        }
        validate_hyperparam("kappa", kappa)?;
        validate_hyperparam("alpha", alpha)?;
        validate_hyperparam("beta", beta)?;
        Ok(Self {
            mu,
            kappa,
            alpha,
            beta,
        })
    }

    /// Returns location hyperparameter $\mu_n$.
    #[verified_engine::verified]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Returns precision-scale hyperparameter $\kappa_n$.
    #[verified_engine::verified]
    pub fn kappa(&self) -> f64 {
        self.kappa
    }

    /// Returns Inverse-Gamma shape $\alpha_n$.
    #[verified_engine::verified]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns Inverse-Gamma scale $\beta_n$.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Conjugate update from a slice of Gaussian observations.
    #[verified_engine::verified]
    pub fn update_slice(&self, data: &[f64]) -> Result<Self, BayesianError> {
        if data.is_empty() {
            return Ok(*self);
        }
        let n = data.len() as f64;
        let mut sum = 0.0;
        for &x in data {
            if !x.is_finite() {
                return Err(BayesianError::InvalidObservation {
                    reason: "observations must be finite",
                });
            }
            sum += x;
        }
        let x_bar = sum / n;
        let mut sse = 0.0;
        for &x in data {
            let d = x - x_bar;
            sse += d * d;
        }
        let kappa_n = self.kappa + n;
        let mu_n = (self.kappa * self.mu + n * x_bar) / kappa_n;
        let alpha_n = self.alpha + 0.5 * n;
        let shrink = (self.kappa * n * (x_bar - self.mu).powi(2)) / (2.0 * kappa_n);
        let beta_n = self.beta + 0.5 * sse + shrink;
        Self::new(mu_n, kappa_n, alpha_n, beta_n)
    }

    /// Exact marginal Student-$t$ posterior for $\mu$: $t_{2\alpha_n}\!\left(\mu_n, \sqrt{\frac{\beta_n}{\alpha_n \kappa_n}}\right)$.
    #[verified_engine::verified]
    pub fn marginal_mean_dist(&self) -> Result<StudentTDist, BayesianError> {
        let scale = (self.beta / (self.alpha * self.kappa)).sqrt();
        StudentTDist::new(2.0 * self.alpha, self.mu, scale).map_err(|_| {
            BayesianError::NumericalInstability {
                reason: "invalid marginal Student-t parameters",
            }
        })
    }

    /// Exact Student-$t$ posterior predictive distribution for a new observation $\tilde{x}$.
    #[verified_engine::verified]
    pub fn predictive_dist(&self) -> Result<StudentTDist, BayesianError> {
        let scale = (self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa)).sqrt();
        StudentTDist::new(2.0 * self.alpha, self.mu, scale).map_err(|_| {
            BayesianError::NumericalInstability {
                reason: "invalid predictive Student-t parameters",
            }
        })
    }

    /// Posterior expectation $E[\sigma^2 \mid D] = \frac{\beta_n}{\alpha_n - 1}$ when $\alpha_n > 1$.
    #[verified_engine::verified]
    pub fn posterior_sigma2_mean(&self) -> Option<f64> {
        if self.alpha > 1.0 {
            Some(self.beta / (self.alpha - 1.0))
        } else {
            None
        }
    }

    /// Equal-tailed credible interval for $\mu$ at mass `level`.
    #[verified_engine::verified]
    pub fn credible_interval_mean(&self, level: f64) -> Result<CredibleInterval, BayesianError> {
        equal_tailed_credible_interval(&self.marginal_mean_dist()?, level)
    }

    /// Exact log marginal likelihood $\ln P(x_{1:n} \mid \mu_0, \kappa_0, \alpha_0, \beta_0)$.
    #[verified_engine::verified]
    pub fn ln_marginal_likelihood(&self, data: &[f64]) -> Result<f64, BayesianError> {
        let post = self.update_slice(data)?;
        let n = data.len() as f64;
        let ln_m = gamma::ln_gamma(post.alpha) - gamma::ln_gamma(self.alpha)
            + self.alpha * self.beta.ln()
            - post.alpha * post.beta.ln()
            + 0.5 * (self.kappa / post.kappa).ln()
            - 0.5 * n * (2.0 * PI).ln();
        Ok(ln_m)
    }
}

// [cite:clinical_trials_statistics]
