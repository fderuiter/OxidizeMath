//! Karlis-Ntzoufras (2003) bivariate Poisson score-grid model with overflow-free log-space recurrence.

use super::grid::{DEFAULT_MAX_GOALS, MAX_GOALS_LIMIT, ScoreGrid};
use super::poisson::{log_factorial, poisson_pmf};
use crate::error::MatchModelError;
use serde::{Deserialize, Deserializer, Serialize};

/// Validated Karlis-Ntzoufras bivariate Poisson parameters $(\lambda_1, \lambda_2, \lambda_3 \ge 0)$.
///
/// Represents $X = U_1 + U_3$ and $Y = U_2 + U_3$ with independent $U_k \sim \text{Poisson}(\lambda_k)$,
/// yielding marginal expectations $\mathbb{E}(X) = \lambda_1 + \lambda_3$,
/// $\mathbb{E}(Y) = \lambda_2 + \lambda_3$, and shared covariance $\text{Cov}(X, Y) = \lambda_3$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct BivariatePoissonParams {
    lambda_1: f64,
    lambda_2: f64,
    lambda_3: f64,
}

impl BivariatePoissonParams {
    /// Additive shared-shock constructor taking $(\lambda_1, \lambda_2, \lambda_3 \ge 0)$ directly.
    ///
    /// Adding shared shock $\lambda_3 > 0$ increases both marginal means to $\lambda_1 + \lambda_3$
    /// and $\lambda_2 + \lambda_3$.
    #[verified_engine::verified]
    pub fn from_components(
        lambda_1: f64,
        lambda_2: f64,
        lambda_3: f64,
    ) -> Result<Self, MatchModelError> {
        for rate in [lambda_1, lambda_2, lambda_3] {
            if rate < 0.0 || !rate.is_finite() {
                return Err(MatchModelError::InvalidSharedRate { value: rate });
            }
        }
        Ok(Self {
            lambda_1,
            lambda_2,
            lambda_3,
        })
    }

    /// Alias for [`Self::from_components`].
    #[verified_engine::verified]
    pub fn new(lambda_1: f64, lambda_2: f64, lambda_3: f64) -> Result<Self, MatchModelError> {
        Self::from_components(lambda_1, lambda_2, lambda_3)
    }

    /// Marginal-preserving constructor taking target marginal means $(\lambda_{\text{home}} > 0, \mu_{\text{away}} > 0)$
    /// and shared covariance $\lambda_3 \in [0, \min(\lambda_{\text{home}}, \mu_{\text{away}})]$,
    /// setting $\lambda_1 = \lambda_{\text{home}} - \lambda_3$ and $\lambda_2 = \mu_{\text{away}} - \lambda_3$.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::match_models::BivariatePoissonParams;
    ///
    /// let params = BivariatePoissonParams::from_marginals(1.35, 1.15, 0.15).unwrap();
    /// assert!((params.lambda_1() - 1.20).abs() < 1e-12);
    /// assert!((params.lambda_2() - 1.00).abs() < 1e-12);
    /// assert!((params.lambda_3() - 0.15).abs() < 1e-12);
    /// ```
    #[verified_engine::verified]
    pub fn from_marginals(
        lambda_home: f64,
        mu_away: f64,
        covariance: f64,
    ) -> Result<Self, MatchModelError> {
        if lambda_home <= 0.0 || !lambda_home.is_finite() {
            return Err(MatchModelError::InvalidGoalRate { value: lambda_home });
        }
        if mu_away <= 0.0 || !mu_away.is_finite() {
            return Err(MatchModelError::InvalidGoalRate { value: mu_away });
        }
        let max_cov = lambda_home.min(mu_away);
        if covariance < 0.0 || covariance > max_cov || !covariance.is_finite() {
            return Err(MatchModelError::InvalidCovariance {
                covariance,
                max_allowed: max_cov,
            });
        }
        Self::from_components(lambda_home - covariance, mu_away - covariance, covariance)
    }

    /// Independent home goal component $\lambda_1$.
    #[verified_engine::verified]
    pub fn lambda_1(&self) -> f64 {
        self.lambda_1
    }

    /// Independent away goal component $\lambda_2$.
    #[verified_engine::verified]
    pub fn lambda_2(&self) -> f64 {
        self.lambda_2
    }

    /// Shared goal covariance component $\lambda_3 = \text{Cov}(X, Y)$.
    #[verified_engine::verified]
    pub fn lambda_3(&self) -> f64 {
        self.lambda_3
    }

    /// Marginal home mean $\mathbb{E}(X) = \lambda_1 + \lambda_3$.
    #[verified_engine::verified]
    pub fn marginal_home(&self) -> f64 {
        self.lambda_1 + self.lambda_3
    }

    /// Marginal away mean $\mathbb{E}(Y) = \lambda_2 + \lambda_3$.
    #[verified_engine::verified]
    pub fn marginal_away(&self) -> f64 {
        self.lambda_2 + self.lambda_3
    }

    /// Evaluates the bounded [`ScoreGrid`] with `DEFAULT_MAX_GOALS = 15` without renormalization.
    #[verified_engine::verified]
    pub fn default_grid(&self) -> Result<ScoreGrid, MatchModelError> {
        bivariate_poisson_grid(self, DEFAULT_MAX_GOALS, false)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBivariatePoissonParams {
    lambda_1: f64,
    lambda_2: f64,
    lambda_3: f64,
}

impl<'de> Deserialize<'de> for BivariatePoissonParams {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawBivariatePoissonParams::deserialize(deserializer)?;
        Self::from_components(raw.lambda_1, raw.lambda_2, raw.lambda_3)
            .map_err(serde::de::Error::custom)
    }
}

#[verified_engine::verified]
fn degenerate_bivariate_pmf(l1: f64, l2: f64, l3: f64, home: usize, away: usize) -> f64 {
    if l3 == 0.0 {
        poisson_pmf(l1, home) * poisson_pmf(l2, away)
    } else if l1 == 0.0 {
        if home <= away {
            poisson_pmf(l3, home) * poisson_pmf(l2, away - home)
        } else {
            0.0
        }
    } else if away <= home {
        poisson_pmf(l3, away) * poisson_pmf(l1, home - away)
    } else {
        0.0
    }
}

/// Evaluates the Karlis-Ntzoufras joint PMF $P(X = \text{home}, Y = \text{away})$ using
/// log-space recurrence $\ln T_{k+1} = \ln T_k + \ln\frac{(x-k)(y-k)}{k+1} + \ln\frac{\lambda_3}{\lambda_1\lambda_2}$.
#[verified_engine::verified]
pub fn bivariate_poisson_pmf(params: &BivariatePoissonParams, home: usize, away: usize) -> f64 {
    let (l1, l2, l3) = (params.lambda_1, params.lambda_2, params.lambda_3);
    if l1 == 0.0 || l2 == 0.0 || l3 == 0.0 {
        return degenerate_bivariate_pmf(l1, l2, l3, home, away);
    }
    let mut log_term = -(l1 + l2 + l3) + (home as f64) * l1.ln() - log_factorial(home)
        + (away as f64) * l2.ln()
        - log_factorial(away);
    let log_ratio = l3.ln() - l1.ln() - l2.ln();
    let mut total = log_term.exp();
    let k_max = home.min(away);
    for k in 0..k_max {
        let num = ((home - k) * (away - k)) as f64;
        log_term += num.ln() - ((k + 1) as f64).ln() + log_ratio;
        total += log_term.exp();
    }
    total
}

/// Builds a bounded 2D Karlis-Ntzoufras bivariate Poisson [`ScoreGrid`] with zero heap allocations.
#[verified_engine::verified]
pub fn bivariate_poisson_grid(
    params: &BivariatePoissonParams,
    max_goals: usize,
    normalize: bool,
) -> Result<ScoreGrid, MatchModelError> {
    if max_goals > MAX_GOALS_LIMIT {
        return Err(MatchModelError::InvalidMaxGoals {
            max_goals,
            limit: MAX_GOALS_LIMIT,
        });
    }
    ScoreGrid::from_fn(max_goals, normalize, |home, away| {
        bivariate_poisson_pmf(params, home, away)
    })
}
