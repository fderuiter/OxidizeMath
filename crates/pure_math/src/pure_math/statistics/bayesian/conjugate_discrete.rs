use super::intervals_and_bf::{
    BayesFactor, CredibleInterval, equal_tailed_credible_interval, hpd_credible_interval,
    validate_hyperparam,
};
use crate::error::BayesianError;
use crate::pure_math::statistics::distributions::{BetaDist, GammaDist, NegativeBinomialDist};
use statrs::function::{beta, gamma};

/// Conjugate Beta-Binomial model $\theta \sim \mathrm{Beta}(\alpha, \beta)$, $k \mid \theta \sim \mathrm{Binomial}(n, \theta)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaBinomial {
    alpha: f64,
    beta: f64,
}

impl BetaBinomial {
    /// Constructs a $\mathrm{Beta}(\alpha, \beta)$ prior/posterior model with `alpha > 0` and `beta > 0`.
    #[verified_engine::verified]
    pub fn new(alpha: f64, beta_param: f64) -> Result<Self, BayesianError> {
        validate_hyperparam("alpha", alpha)?;
        validate_hyperparam("beta", beta_param)?;
        Ok(Self {
            alpha,
            beta: beta_param,
        })
    }

    /// Non-informative Laplace uniform prior $\mathrm{Beta}(1, 1)$.
    #[verified_engine::verified]
    pub fn uniform() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Jeffreys invariant prior $\mathrm{Beta}(0.5, 0.5)$.
    #[verified_engine::verified]
    pub fn jeffreys() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Returns $\alpha$.
    #[verified_engine::verified]
    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Returns $\beta$.
    #[verified_engine::verified]
    pub fn beta(&self) -> f64 {
        self.beta
    }

    /// Conjugate update after observing `successes` out of `trials` Bernoulli trials.
    #[verified_engine::verified]
    pub fn update(&self, successes: u64, trials: u64) -> Result<Self, BayesianError> {
        if successes > trials {
            return Err(BayesianError::InvalidObservation {
                reason: "successes cannot exceed trials",
            });
        }
        let failures = trials - successes;
        Self::new(
            self.alpha + (successes as f64),
            self.beta + (failures as f64),
        )
    }

    /// Returns the exact [`BetaDist`] posterior distribution.
    #[verified_engine::verified]
    pub fn posterior_dist(&self) -> Result<BetaDist, BayesianError> {
        BetaDist::new(self.alpha, self.beta).map_err(|_| BayesianError::NumericalInstability {
            reason: "invalid posterior Beta parameters",
        })
    }

    /// Posterior mean $\frac{\alpha}{\alpha + \beta}$.
    #[verified_engine::verified]
    pub fn posterior_mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance $\frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$.
    #[verified_engine::verified]
    pub fn posterior_variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// Posterior mode $\frac{\alpha - 1}{\alpha + \beta - 2}$ when $\alpha > 1$ and $\beta > 1$.
    #[verified_engine::verified]
    pub fn posterior_mode(&self) -> Option<f64> {
        if self.alpha > 1.0 && self.beta > 1.0 {
            Some((self.alpha - 1.0) / (self.alpha + self.beta - 2.0))
        } else {
            None
        }
    }

    /// Equal-tailed credible interval at mass `level`.
    #[verified_engine::verified]
    pub fn credible_interval(&self, level: f64) -> Result<CredibleInterval, BayesianError> {
        equal_tailed_credible_interval(&self.posterior_dist()?, level)
    }

    /// Highest Posterior Density (HPD) credible interval at mass `level`.
    #[verified_engine::verified]
    pub fn hpd_interval(&self, level: f64) -> Result<CredibleInterval, BayesianError> {
        hpd_credible_interval(&self.posterior_dist()?, level)
    }

    /// Posterior predictive probability of `future_successes` in `future_trials` new trials.
    #[verified_engine::verified]
    pub fn predictive_pmf(
        &self,
        future_successes: u64,
        future_trials: u64,
    ) -> Result<f64, BayesianError> {
        let ln_p = self.ln_marginal_likelihood(future_successes, future_trials)?;
        Ok(ln_p.exp())
    }

    /// Log marginal likelihood $\ln P(k \mid n, \alpha, \beta)$ under the Beta-Binomial compound model.
    #[verified_engine::verified]
    pub fn ln_marginal_likelihood(
        &self,
        successes: u64,
        trials: u64,
    ) -> Result<f64, BayesianError> {
        if successes > trials {
            return Err(BayesianError::InvalidObservation {
                reason: "successes cannot exceed trials",
            });
        }
        let nf = trials as f64;
        let kf = successes as f64;
        let ln_comb =
            gamma::ln_gamma(nf + 1.0) - gamma::ln_gamma(kf + 1.0) - gamma::ln_gamma(nf - kf + 1.0);
        let ln_b_num = beta::ln_beta(self.alpha + kf, self.beta + (nf - kf));
        let ln_b_den = beta::ln_beta(self.alpha, self.beta);
        Ok(ln_comb + ln_b_num - ln_b_den)
    }

    /// Bayes factor $BF_{10}$ comparing $M_1: \theta \sim \mathrm{Beta}(\alpha, \beta)$ against point null $M_0: \theta = p_0$.
    #[verified_engine::verified]
    pub fn bayes_factor_vs_point(
        &self,
        null_p: f64,
        successes: u64,
        trials: u64,
    ) -> Result<BayesFactor, BayesianError> {
        if !null_p.is_finite() || null_p <= 0.0 || null_p >= 1.0 {
            return Err(BayesianError::InvalidHyperparameter {
                name: "null_p",
                value: null_p,
            });
        }
        let ln_m1 = self.ln_marginal_likelihood(successes, trials)?;
        let nf = trials as f64;
        let kf = successes as f64;
        let ln_comb =
            gamma::ln_gamma(nf + 1.0) - gamma::ln_gamma(kf + 1.0) - gamma::ln_gamma(nf - kf + 1.0);
        let ln_m0 = ln_comb + kf * null_p.ln() + (nf - kf) * (-null_p).ln_1p();
        BayesFactor::from_ln_marginals(ln_m1, ln_m0)
    }
}

/// Conjugate Gamma-Poisson model $\lambda \sim \mathrm{Gamma}(\alpha, \beta)$, $y_i \mid \lambda \sim \mathrm{Poisson}(\lambda)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaPoisson {
    shape: f64,
    rate: f64,
}

impl GammaPoisson {
    /// Constructs a $\mathrm{Gamma}(\text{shape}, \text{rate})$ model with `shape > 0` and `rate > 0`.
    #[verified_engine::verified]
    pub fn new(shape: f64, rate: f64) -> Result<Self, BayesianError> {
        validate_hyperparam("shape", shape)?;
        validate_hyperparam("rate", rate)?;
        Ok(Self { shape, rate })
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

    /// Conjugate update given aggregate `total_count` over positive `exposure` time/units.
    #[verified_engine::verified]
    pub fn update(&self, total_count: u64, exposure: f64) -> Result<Self, BayesianError> {
        if !exposure.is_finite() || exposure < 0.0 {
            return Err(BayesianError::InvalidObservation {
                reason: "exposure must be finite and non-negative",
            });
        }
        Self::new(self.shape + (total_count as f64), self.rate + exposure)
    }

    /// Conjugate update from a slice of unit-exposure Poisson counts.
    #[verified_engine::verified]
    pub fn update_counts(&self, counts: &[u64]) -> Result<Self, BayesianError> {
        let total: u64 = counts.iter().copied().sum();
        self.update(total, counts.len() as f64)
    }

    /// Returns the exact [`GammaDist`] posterior distribution.
    #[verified_engine::verified]
    pub fn posterior_dist(&self) -> Result<GammaDist, BayesianError> {
        GammaDist::new(self.shape, self.rate).map_err(|_| BayesianError::NumericalInstability {
            reason: "invalid posterior Gamma parameters",
        })
    }

    /// Posterior mean $\alpha / \beta$.
    #[verified_engine::verified]
    pub fn posterior_mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Posterior variance $\alpha / \beta^2$.
    #[verified_engine::verified]
    pub fn posterior_variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Equal-tailed credible interval at mass `level`.
    #[verified_engine::verified]
    pub fn credible_interval(&self, level: f64) -> Result<CredibleInterval, BayesianError> {
        equal_tailed_credible_interval(&self.posterior_dist()?, level)
    }

    /// Highest Posterior Density (HPD) credible interval at mass `level`.
    #[verified_engine::verified]
    pub fn hpd_interval(&self, level: f64) -> Result<CredibleInterval, BayesianError> {
        hpd_credible_interval(&self.posterior_dist()?, level)
    }

    /// Exact Negative-Binomial posterior predictive distribution for a unit-exposure count.
    #[verified_engine::verified]
    pub fn predictive_dist(&self) -> Result<NegativeBinomialDist, BayesianError> {
        let p = self.rate / (self.rate + 1.0);
        NegativeBinomialDist::new(self.shape, p).map_err(|_| BayesianError::NumericalInstability {
            reason: "invalid predictive NegativeBinomial parameters",
        })
    }

    /// Log marginal likelihood $\ln P(y_{1:n} \mid \alpha, \beta)$ of unit-exposure counts.
    #[verified_engine::verified]
    pub fn ln_marginal_likelihood(&self, counts: &[u64]) -> Result<f64, BayesianError> {
        let n = counts.len() as f64;
        let sum_y: f64 = counts.iter().map(|&k| k as f64).sum();
        let sum_ln_fact: f64 = counts
            .iter()
            .map(|&k| gamma::ln_gamma((k as f64) + 1.0))
            .sum();
        let alpha_n = self.shape + sum_y;
        let beta_n = self.rate + n;
        Ok(
            gamma::ln_gamma(alpha_n) - gamma::ln_gamma(self.shape) + self.shape * self.rate.ln()
                - alpha_n * beta_n.ln()
                - sum_ln_fact,
        )
    }

    /// Bayes factor $BF_{10}$ comparing $M_1: \lambda \sim \mathrm{Gamma}(\alpha, \beta)$ vs $M_0: \lambda = \lambda_0$.
    #[verified_engine::verified]
    pub fn bayes_factor_vs_rate(
        &self,
        null_lambda: f64,
        counts: &[u64],
    ) -> Result<BayesFactor, BayesianError> {
        validate_hyperparam("null_lambda", null_lambda)?;
        let ln_m1 = self.ln_marginal_likelihood(counts)?;
        let n = counts.len() as f64;
        let sum_y: f64 = counts.iter().map(|&k| k as f64).sum();
        let sum_ln_fact: f64 = counts
            .iter()
            .map(|&k| gamma::ln_gamma((k as f64) + 1.0))
            .sum();
        let ln_m0 = sum_y * null_lambda.ln() - n * null_lambda - sum_ln_fact;
        BayesFactor::from_ln_marginals(ln_m1, ln_m0)
    }
}

// [cite:clinical_trials_statistics]
