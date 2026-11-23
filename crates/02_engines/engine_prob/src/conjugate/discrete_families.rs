//! Discrete-likelihood conjugate families: [`BetaBinomial`], [`DirichletMultinomial`], and [`GammaPoisson`].

use crate::distributions::{Beta, ContinuousDist, Gamma};
use crate::error::ProbError;
use crate::special::{ln_beta, ln_factorial, ln_gamma};

fn validate_credible_level(level: f64) -> Result<(f64, f64), ProbError> {
    if !level.is_finite() || level <= 0.0 || level >= 1.0 {
        return Err(ProbError::InvalidProbability { value: level });
    }
    let tail = 0.5 * (1.0 - level);
    Ok((tail, 1.0 - tail))
}

/// Conjugate Beta-Binomial model $\theta \sim \operatorname{Beta}(\alpha, \beta)$, $Y \mid \theta \sim \operatorname{Binomial}(n, \theta)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BetaBinomial {
    alpha: f64,
    beta: f64,
}

impl BetaBinomial {
    /// Initializes a Beta prior $\operatorname{Beta}(\alpha, \beta)$ with $\alpha > 0, \beta > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(alpha: f64, beta: f64) -> Result<Self, ProbError> {
        let _ = Beta::new(alpha, beta)?;
        debug_assert!(alpha > 0.0 && beta > 0.0);
        Ok(Self { alpha, beta })
    }

    /// Uniform Laplace prior $\operatorname{Beta}(1, 1)$.
    #[must_use]
    #[inline]
    pub const fn uniform_prior() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    /// Jeffreys invariant prior $\operatorname{Beta}(1/2, 1/2)$.
    #[must_use]
    #[inline]
    pub const fn jeffreys_prior() -> Self {
        Self {
            alpha: 0.5,
            beta: 0.5,
        }
    }

    /// Current hyperparameter $\alpha$.
    #[must_use]
    #[inline]
    pub const fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Current hyperparameter $\beta$.
    #[must_use]
    #[inline]
    pub const fn beta(&self) -> f64 {
        self.beta
    }

    /// Returns the conjugate posterior after observing `successes` and `failures` (zero heap allocations).
    #[must_use]
    #[inline]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update(&self, successes: u64, failures: u64) -> Self {
        let next = Self {
            alpha: self.alpha + (successes as f64),
            beta: self.beta + (failures as f64),
        };
        debug_assert!(next.alpha >= self.alpha && next.beta >= self.beta);
        next
    }

    /// Returns the exact posterior [`Beta`] distribution.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn posterior(&self) -> Result<Beta, ProbError> {
        Beta::new(self.alpha, self.beta)
    }

    /// Posterior mean $\frac{\alpha}{\alpha + \beta}$.
    #[must_use]
    #[inline]
    pub fn posterior_mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    /// Posterior variance $\frac{\alpha \beta}{(\alpha + \beta)^2(\alpha + \beta + 1)}$.
    #[must_use]
    #[inline]
    pub fn posterior_variance(&self) -> f64 {
        let s = self.alpha + self.beta;
        (self.alpha * self.beta) / (s * s * (s + 1.0))
    }

    /// Exact equal-tailed posterior credible interval at `level` $\in (0, 1)$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        let (lo_p, hi_p) = validate_credible_level(level)?;
        let post = self.posterior()?;
        Ok((post.quantile(lo_p)?, post.quantile(hi_p)?))
    }

    /// Evaluates exact log marginal likelihood $\ln P(Y = k \mid n, \alpha, \beta)$ for `k <= n`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn log_marginal_likelihood(&self, successes: u64, trials: u64) -> Result<f64, ProbError> {
        if successes > trials {
            return Err(ProbError::InvalidParameter {
                name: "successes",
                value: successes as f64,
            });
        }
        let failures = trials - successes;
        let log_choose = ln_factorial(trials) - ln_factorial(successes) - ln_factorial(failures);
        let lml = log_choose
            + ln_beta(
                self.alpha + (successes as f64),
                self.beta + (failures as f64),
            )
            - ln_beta(self.alpha, self.beta);
        debug_assert!(lml <= 1.0e-12);
        Ok(lml)
    }

    /// Posterior predictive probability mass $P(Y_{\text{new}} = k \mid n_{\text{new}})$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn predictive_pmf(&self, successes: u64, trials: u64) -> Result<f64, ProbError> {
        Ok(libm::exp(self.log_marginal_likelihood(successes, trials)?))
    }

    /// Exact Bayes factor $BF_{10} = \frac{P(Y = k \mid n, \text{self})}{P(Y = k \mid n, \text{baseline})}$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn bayes_factor(
        &self,
        baseline: &Self,
        successes: u64,
        trials: u64,
    ) -> Result<f64, ProbError> {
        let diff = self.log_marginal_likelihood(successes, trials)?
            - baseline.log_marginal_likelihood(successes, trials)?;
        Ok(libm::exp(diff))
    }
}

/// Updates Dirichlet concentration parameters `alphas` with `counts` into `out` (zero heap allocations).
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn dirichlet_update_into(
    alphas: &[f64],
    counts: &[u64],
    out: &mut [f64],
) -> Result<(), ProbError> {
    if alphas.is_empty() {
        return Err(ProbError::EmptyInput {
            context: "dirichlet_update_into",
        });
    }
    if counts.len() != alphas.len() {
        return Err(ProbError::DimensionMismatch {
            expected: alphas.len(),
            actual: counts.len(),
        });
    }
    if out.len() < alphas.len() {
        return Err(ProbError::BufferTooSmall {
            required: alphas.len(),
            actual: out.len(),
        });
    }
    for i in 0..alphas.len() {
        let a = alphas[i];
        if !a.is_finite() || a <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "alpha",
                value: a,
            });
        }
        out[i] = a + (counts[i] as f64);
    }
    Ok(())
}

/// Evaluates the exact Polya-urn / Dirichlet-Multinomial log marginal likelihood $\ln P(\mathbf{c} \mid \boldsymbol{\alpha})$.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn dirichlet_log_marginal_likelihood(alphas: &[f64], counts: &[u64]) -> Result<f64, ProbError> {
    if alphas.is_empty() {
        return Err(ProbError::EmptyInput {
            context: "dirichlet_log_marginal_likelihood",
        });
    }
    if counts.len() != alphas.len() {
        return Err(ProbError::DimensionMismatch {
            expected: alphas.len(),
            actual: counts.len(),
        });
    }
    let mut alpha_sum = 0.0;
    let mut total_n = 0_u64;
    let mut term_sum = 0.0;
    for (&a, &c) in alphas.iter().zip(counts.iter()) {
        if !a.is_finite() || a <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "alpha",
                value: a,
            });
        }
        alpha_sum += a;
        total_n = total_n.saturating_add(c);
        term_sum += ln_gamma(a + (c as f64)) - ln_gamma(a) - ln_factorial(c);
    }
    let lml = ln_factorial(total_n) + ln_gamma(alpha_sum) - ln_gamma(alpha_sum + (total_n as f64))
        + term_sum;
    debug_assert!(lml.is_finite());
    Ok(lml)
}

/// Computes the exact Bayes factor $BF_{10}$ between two Dirichlet-Multinomial priors on `counts`.
#[cfg_attr(feature = "verified", verified_engine::verified)]
pub fn dirichlet_bayes_factor(
    alphas_candidate: &[f64],
    alphas_baseline: &[f64],
    counts: &[u64],
) -> Result<f64, ProbError> {
    let l1 = dirichlet_log_marginal_likelihood(alphas_candidate, counts)?;
    let l0 = dirichlet_log_marginal_likelihood(alphas_baseline, counts)?;
    Ok(libm::exp(l1 - l0))
}

/// Stack-allocated $K$-category Dirichlet-Multinomial conjugate model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletMultinomial<const K: usize> {
    alphas: [f64; K],
}

impl<const K: usize> DirichletMultinomial<K> {
    /// Initializes $\operatorname{Dirichlet}(\alpha_1, \dots, \alpha_K)$ requiring $K \ge 2$ and $\alpha_i > 0$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(alphas: [f64; K]) -> Result<Self, ProbError> {
        if K < 2 {
            return Err(ProbError::InvalidParameter {
                name: "K",
                value: K as f64,
            });
        }
        for a in alphas {
            if !a.is_finite() || a <= 0.0 {
                return Err(ProbError::InvalidParameter {
                    name: "alpha",
                    value: a,
                });
            }
        }
        Ok(Self { alphas })
    }

    /// Returns the current concentration vector.
    #[must_use]
    #[inline]
    pub const fn alphas(&self) -> &[f64; K] {
        &self.alphas
    }

    /// Updates concentration parameters with `counts` without heap allocations.
    #[must_use]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update(&self, counts: &[u64; K]) -> Self {
        let mut next = self.alphas;
        for (slot, &c) in next.iter_mut().zip(counts.iter()) {
            *slot += c as f64;
        }
        Self { alphas: next }
    }

    /// Posterior mean vector $\mathbb{E}\[\theta_i\] = \alpha_i / \alpha_0$.
    #[must_use]
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn posterior_means(&self) -> [f64; K] {
        let sum: f64 = self.alphas.iter().sum();
        let mut out = [0.0; K];
        for (slot, &a) in out.iter_mut().zip(self.alphas.iter()) {
            *slot = a / sum;
        }
        out
    }

    /// Exact equal-tailed marginal credible interval for category `index` via $\operatorname{Beta}(\alpha_i, \alpha_0 - \alpha_i)$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn credible_interval(&self, index: usize, level: f64) -> Result<(f64, f64), ProbError> {
        if index >= K {
            return Err(ProbError::InvalidParameter {
                name: "index",
                value: index as f64,
            });
        }
        let (lo_p, hi_p) = validate_credible_level(level)?;
        let sum: f64 = self.alphas.iter().sum();
        let ai = self.alphas[index];
        let marginal = Beta::new(ai, sum - ai)?;
        Ok((marginal.quantile(lo_p)?, marginal.quantile(hi_p)?))
    }

    /// Evaluates $\ln P(\mathbf{c} \mid \boldsymbol{\alpha})$ for `counts`.
    #[inline]
    pub fn log_marginal_likelihood(&self, counts: &[u64; K]) -> Result<f64, ProbError> {
        dirichlet_log_marginal_likelihood(&self.alphas, counts)
    }

    /// Evaluates Bayes factor $BF_{10}$ relative to `baseline`.
    #[inline]
    pub fn bayes_factor(&self, baseline: &Self, counts: &[u64; K]) -> Result<f64, ProbError> {
        dirichlet_bayes_factor(&self.alphas, &baseline.alphas, counts)
    }
}

/// Conjugate Gamma-Poisson model $\lambda \sim \operatorname{Gamma}(\alpha, \beta)$ (shape-rate), $Y \mid \lambda \sim \operatorname{Poisson}(e \lambda)$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaPoisson {
    shape: f64,
    rate: f64,
}

impl GammaPoisson {
    /// Initializes a $\operatorname{Gamma}(\text{shape}, \text{rate})$ prior with `shape > 0, rate > 0`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn new(shape: f64, rate: f64) -> Result<Self, ProbError> {
        let _ = Gamma::new(shape, rate)?;
        debug_assert!(shape > 0.0 && rate > 0.0);
        Ok(Self { shape, rate })
    }

    /// Shape parameter $\alpha$.
    #[must_use]
    #[inline]
    pub const fn shape(&self) -> f64 {
        self.shape
    }

    /// Rate (effective prior exposure) parameter $\beta$.
    #[must_use]
    #[inline]
    pub const fn rate(&self) -> f64 {
        self.rate
    }

    /// Updates the prior with `total_count` observed over `exposure > 0`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update(&self, total_count: u64, exposure: f64) -> Result<Self, ProbError> {
        if !exposure.is_finite() || exposure < 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "exposure",
                value: exposure,
            });
        }
        Ok(Self {
            shape: self.shape + (total_count as f64),
            rate: self.rate + exposure,
        })
    }

    /// Updates the prior with a slice of unit-exposure count observations.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn update_counts(&self, counts: &[u64]) -> Self {
        let sum: u64 = counts.iter().copied().sum();
        Self {
            shape: self.shape + (sum as f64),
            rate: self.rate + (counts.len() as f64),
        }
    }

    /// Exact posterior [`Gamma`] distribution.
    #[inline]
    pub fn posterior(&self) -> Result<Gamma, ProbError> {
        Gamma::new(self.shape, self.rate)
    }

    /// Posterior mean $\alpha / \beta$.
    #[must_use]
    #[inline]
    pub fn posterior_mean(&self) -> f64 {
        self.shape / self.rate
    }

    /// Posterior variance $\alpha / \beta^2$.
    #[must_use]
    #[inline]
    pub fn posterior_variance(&self) -> f64 {
        self.shape / (self.rate * self.rate)
    }

    /// Exact equal-tailed posterior credible interval at `level` $\in (0, 1)$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn credible_interval(&self, level: f64) -> Result<(f64, f64), ProbError> {
        let (lo_p, hi_p) = validate_credible_level(level)?;
        let post = self.posterior()?;
        Ok((post.quantile(lo_p)?, post.quantile(hi_p)?))
    }

    /// Negative-Binomial posterior predictive log-PMF $\ln P(Y_{\text{new}} = k \mid \text{exposure})$.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn predictive_ln_pmf(&self, count: u64, exposure: f64) -> Result<f64, ProbError> {
        if !exposure.is_finite() || exposure <= 0.0 {
            return Err(ProbError::InvalidParameter {
                name: "exposure",
                value: exposure,
            });
        }
        let kf = count as f64;
        let lp = ln_gamma(self.shape + kf) - ln_gamma(self.shape) - ln_factorial(count)
            + self.shape * libm::log(self.rate)
            + kf * libm::log(exposure)
            - (self.shape + kf) * libm::log(self.rate + exposure);
        debug_assert!(lp <= 1.0e-12);
        Ok(lp)
    }

    /// Evaluates exact log marginal likelihood $\ln P(\mathbf{y} \mid \alpha, \beta)$ across unit-exposure `counts`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn log_marginal_likelihood(&self, counts: &[u64]) -> Result<f64, ProbError> {
        if counts.is_empty() {
            return Ok(0.0);
        }
        let mut total = 0_u64;
        let mut sum_ln_fact = 0.0;
        for &c in counts {
            total = total.saturating_add(c);
            sum_ln_fact += ln_factorial(c);
        }
        let nf = counts.len() as f64;
        let sf = total as f64;
        let lml = self.shape * libm::log(self.rate) - (self.shape + sf) * libm::log(self.rate + nf)
            + ln_gamma(self.shape + sf)
            - ln_gamma(self.shape)
            - sum_ln_fact;
        debug_assert!(lml.is_finite());
        Ok(lml)
    }

    /// Exact Bayes factor $BF_{10}$ relative to `baseline` across unit-exposure `counts`.
    #[cfg_attr(feature = "verified", verified_engine::verified)]
    pub fn bayes_factor(&self, baseline: &Self, counts: &[u64]) -> Result<f64, ProbError> {
        let l1 = self.log_marginal_likelihood(counts)?;
        let l0 = baseline.log_marginal_likelihood(counts)?;
        Ok(libm::exp(l1 - l0))
    }
}
