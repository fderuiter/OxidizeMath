#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use serde::{Deserialize, Serialize};

/// Supported parametric distribution families in `DistributionsExplorerTool`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistFamily {
    /// Gaussian $\mathcal{N}(\mu, \sigma^2)$.
    Normal,
    /// Shape-rate $\operatorname{Gamma}(\alpha, \beta)$.
    Gamma,
    /// Bounded $\operatorname{Beta}(\alpha, \beta)$ on $[0, 1]$.
    Beta,
    /// Student's $t(\nu)$.
    StudentsT,
    /// Chi-squared $\chi^2(\nu)$.
    ChiSquared,
    /// Rate-parameterized $\operatorname{Exp}(\lambda)$.
    Exponential,
    /// Continuous $\operatorname{Uniform}(a, a + w)$.
    Uniform,
    /// Discrete $\operatorname{Poisson}(\lambda)$.
    Poisson,
    /// Discrete $\operatorname{Binomial}(n, p)$.
    Binomial,
    /// Binary $\operatorname{Bernoulli}(p)$.
    Bernoulli,
}

pub(super) type SliderSpec = (&'static str, std::ops::RangeInclusive<f64>);

impl DistFamily {
    pub(super) const ALL: [Self; 10] = [
        Self::Normal,
        Self::Gamma,
        Self::Beta,
        Self::StudentsT,
        Self::ChiSquared,
        Self::Exponential,
        Self::Uniform,
        Self::Poisson,
        Self::Binomial,
        Self::Bernoulli,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            Self::Normal => "Normal (μ, σ)",
            Self::Gamma => "Gamma (α, β)",
            Self::Beta => "Beta (α, β)",
            Self::StudentsT => "Student's t (ν)",
            Self::ChiSquared => "Chi-Squared (ν)",
            Self::Exponential => "Exponential (λ)",
            Self::Uniform => "Uniform (a, width)",
            Self::Poisson => "Poisson (λ)",
            Self::Binomial => "Binomial (n, p)",
            Self::Bernoulli => "Bernoulli (p)",
        }
    }

    pub(super) fn is_discrete(self) -> bool {
        matches!(self, Self::Poisson | Self::Binomial | Self::Bernoulli)
    }

    pub(super) fn slider_specs(self) -> (SliderSpec, Option<SliderSpec>) {
        match self {
            Self::Normal => (("Mean μ", -10.0..=10.0), Some(("Std σ", 0.1..=8.0))),
            Self::Gamma => (("Shape α", 0.3..=15.0), Some(("Rate β", 0.1..=5.0))),
            Self::Beta => (("Alpha α", 0.3..=15.0), Some(("Beta β", 0.3..=15.0))),
            Self::StudentsT | Self::ChiSquared => (("DoF ν", 2.1..=25.0), None),
            Self::Exponential | Self::Poisson => (("Rate λ", 0.2..=20.0), None),
            Self::Uniform => (("Lower a", -10.0..=10.0), Some(("Width w", 0.2..=12.0))),
            Self::Binomial => (("Trials n", 1.0..=40.0), Some(("Prob p", 0.02..=0.98))),
            Self::Bernoulli => (("Prob p", 0.02..=0.98), None),
        }
    }
}

/// Bounded serializable configuration for `DistributionsExplorerTool`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionsState {
    /// Selected distribution family.
    pub family: DistFamily,
    /// Primary parameter ($\mu$, $\alpha$, $\nu$, $\lambda$, $a$, $n$, or $p$).
    pub param_a: f64,
    /// Secondary parameter ($\sigma$, $\beta$, width, or $p$).
    pub param_b: f64,
    /// Target quantile level $q \in [0.001, 0.999]$.
    pub quantile_prob: f64,
    /// Deterministic Monte Carlo sample size.
    pub sample_count: usize,
    /// Deterministic `OxidizeRng` seed.
    pub seed: u64,
    /// Overlay analytical CDF curve on plot.
    pub show_cdf: bool,
}

impl Default for DistributionsState {
    fn default() -> Self {
        Self {
            family: DistFamily::Beta,
            param_a: 2.5,
            param_b: 5.0,
            quantile_prob: 0.90,
            sample_count: 512,
            seed: 42,
            show_cdf: true,
        }
    }
}

impl DistributionsState {
    pub(super) fn validate(&self) -> Result<(), String> {
        let ok = self.param_a.is_finite()
            && (-10.0..=40.0).contains(&self.param_a)
            && self.param_b.is_finite()
            && (0.05..=20.0).contains(&self.param_b)
            && self.quantile_prob.is_finite()
            && (0.001..=0.999).contains(&self.quantile_prob)
            && (64..=4096).contains(&self.sample_count);
        if !ok {
            return Err("DistributionsState parameter out of bounds".into());
        }
        Ok(())
    }
}

/// Summary moments, quantile, and Monte Carlo diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct DistMetrics {
    /// Theoretical mean $\mathbb{E}\[X\]$.
    pub mean: f64,
    /// Theoretical variance $\operatorname{Var}(X)$.
    pub variance: f64,
    /// Exact inverse-CDF quantile $F^{-1}(q)$.
    pub quantile_x: f64,
    /// Empirical mean of deterministic `OxidizeRng` samples.
    pub sample_mean: f64,
    /// 1D Wasserstein distance $W_1$ between independent split-sample halves.
    pub split_w1: f64,
}
