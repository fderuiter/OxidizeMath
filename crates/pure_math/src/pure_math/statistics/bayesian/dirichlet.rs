use super::intervals_and_bf::{
    BayesFactor, CredibleInterval, equal_tailed_credible_interval, validate_hyperparam,
};
use crate::error::BayesianError;
use crate::pure_math::statistics::distributions::BetaDist;
use crate::pure_math::statistics::distributions::special::sample_gamma_unit_scale;
use oxidize_core::rng::OxidizeRng;
use statrs::function::gamma;

/// Conjugate Dirichlet-Multinomial model $\mathbf{p} \sim \mathrm{Dirichlet}(\boldsymbol{\alpha})$
/// over the $K$-simplex ($K \ge 2$) with zero-allocation fixed-size arrays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirichletMultinomial<const K: usize> {
    alphas: [f64; K],
}

impl<const K: usize> DirichletMultinomial<K> {
    /// Constructs $\mathrm{Dirichlet}(\alpha_1, \dots, \alpha_K)$ verifying $K \ge 2$ and $\alpha_k > 0$.
    #[verified_engine::verified]
    pub fn new(alphas: [f64; K]) -> Result<Self, BayesianError> {
        if K < 2 {
            return Err(BayesianError::InvalidHyperparameter {
                name: "dimension_k",
                value: K as f64,
            });
        }
        for &a in &alphas {
            validate_hyperparam("alpha", a)?;
        }
        Ok(Self { alphas })
    }

    /// Constructs a symmetric $\mathrm{Dirichlet}(\alpha, \dots, \alpha)$ model.
    #[verified_engine::verified]
    pub fn symmetric(alpha: f64) -> Result<Self, BayesianError> {
        validate_hyperparam("alpha", alpha)?;
        Self::new([alpha; K])
    }

    /// Returns concentration hyperparameters $\boldsymbol{\alpha}$.
    #[verified_engine::verified]
    pub fn alphas(&self) -> &[f64; K] {
        &self.alphas
    }

    /// Returns total concentration $\alpha_0 = \sum_{k=1}^K \alpha_k$.
    #[verified_engine::verified]
    pub fn concentration(&self) -> f64 {
        self.alphas.iter().sum()
    }

    /// Conjugate update given observed category counts `counts`.
    #[verified_engine::verified]
    pub fn update(&self, counts: &[u64; K]) -> Result<Self, BayesianError> {
        let mut next = self.alphas;
        for i in 0..K {
            next[i] += counts[i] as f64;
        }
        Self::new(next)
    }

    /// Posterior mean vector $E(p_k) = \alpha_k / \alpha_0$ on the $K$-simplex.
    #[verified_engine::verified]
    pub fn posterior_mean(&self) -> [f64; K] {
        let sum = self.concentration();
        let mut means = [0.0_f64; K];
        for i in 0..K {
            means[i] = self.alphas[i] / sum;
        }
        means
    }

    /// Exact marginal [`BetaDist`] $\mathrm{Beta}(\alpha_k, \alpha_0 - \alpha_k)$ for category `index`.
    #[verified_engine::verified]
    pub fn marginal_beta(&self, index: usize) -> Result<BetaDist, BayesianError> {
        if index >= K {
            return Err(BayesianError::CategoryOutOfBounds {
                index,
                dimension: K,
            });
        }
        let ak = self.alphas[index];
        let rest = (self.concentration() - ak).max(1e-15);
        BetaDist::new(ak, rest).map_err(|_| BayesianError::NumericalInstability {
            reason: "invalid marginal Beta parameters",
        })
    }

    /// Equal-tailed marginal credible interval for category `index` at mass `level`.
    #[verified_engine::verified]
    pub fn marginal_credible_interval(
        &self,
        index: usize,
        level: f64,
    ) -> Result<CredibleInterval, BayesianError> {
        equal_tailed_credible_interval(&self.marginal_beta(index)?, level)
    }

    /// Draws a probability vector on the $K$-simplex via normalized Gamma variates with zero heap allocations.
    #[verified_engine::verified]
    pub fn sample_simplex(&self, rng: &mut OxidizeRng) -> [f64; K] {
        let mut draws = [0.0_f64; K];
        let mut total = 0.0;
        for i in 0..K {
            let g = sample_gamma_unit_scale(self.alphas[i], rng);
            draws[i] = g;
            total += g;
        }
        if total > 0.0 {
            for v in &mut draws {
                *v /= total;
            }
            draws
        } else {
            self.posterior_mean()
        }
    }

    /// Exact Dirichlet-Multinomial compound log marginal likelihood $\ln P(\mathbf{c} \mid \boldsymbol{\alpha})$.
    #[verified_engine::verified]
    pub fn ln_marginal_likelihood(&self, counts: &[u64; K]) -> Result<f64, BayesianError> {
        let n_total: u64 = counts.iter().copied().sum();
        let nf = n_total as f64;
        let a0 = self.concentration();

        let mut ln_multinomial = gamma::ln_gamma(nf + 1.0);
        let mut ln_dirichlet_ratio = gamma::ln_gamma(a0) - gamma::ln_gamma(a0 + nf);
        for i in 0..K {
            let ci = counts[i] as f64;
            ln_multinomial -= gamma::ln_gamma(ci + 1.0);
            ln_dirichlet_ratio +=
                gamma::ln_gamma(self.alphas[i] + ci) - gamma::ln_gamma(self.alphas[i]);
        }
        Ok(ln_multinomial + ln_dirichlet_ratio)
    }

    /// Bayes factor $BF_{10}$ comparing $M_1: \mathbf{p} \sim \mathrm{Dirichlet}(\boldsymbol{\alpha})$
    /// against fixed categorical probabilities $M_0: \mathbf{p} = \mathbf{p}_0$.
    #[verified_engine::verified]
    pub fn bayes_factor_vs_probs(
        &self,
        null_probs: &[f64; K],
        counts: &[u64; K],
    ) -> Result<BayesFactor, BayesianError> {
        let mut sum_p = 0.0;
        for &p in null_probs {
            if !p.is_finite() || p <= 0.0 || p >= 1.0 {
                return Err(BayesianError::InvalidHyperparameter {
                    name: "null_prob",
                    value: p,
                });
            }
            sum_p += p;
        }
        if (sum_p - 1.0).abs() > 1e-6 {
            return Err(BayesianError::InvalidHyperparameter {
                name: "null_probs_sum",
                value: sum_p,
            });
        }
        let ln_m1 = self.ln_marginal_likelihood(counts)?;
        let nf = counts.iter().copied().sum::<u64>() as f64;
        let mut ln_m0 = gamma::ln_gamma(nf + 1.0);
        for i in 0..K {
            let ci = counts[i] as f64;
            ln_m0 += ci * null_probs[i].ln() - gamma::ln_gamma(ci + 1.0);
        }
        BayesFactor::from_ln_marginals(ln_m1, ln_m0)
    }
}

// [cite:clinical_trials_statistics]
