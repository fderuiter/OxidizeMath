//! Independent two-team Maher (1982) Poisson score-grid model evaluated in log-space via `ln_gamma`.

use super::grid::{DEFAULT_MAX_GOALS, GRID_STRIDE, MAX_GOALS_LIMIT, ScoreGrid};
use crate::error::MatchModelError;
use serde::{Deserialize, Deserializer, Serialize};
use statrs::function::gamma::ln_gamma;

/// Validated expected-goal parameters $(\lambda, \mu)$ for an independent two-team Poisson match model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct IndependentPoissonParams {
    lambda: f64,
    mu: f64,
}

impl IndependentPoissonParams {
    /// Creates validated independent Poisson parameters (`lambda > 0`, `mu > 0`).
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::match_models::IndependentPoissonParams;
    ///
    /// let params = IndependentPoissonParams::new(1.35, 1.15).unwrap();
    /// assert_eq!(params.lambda(), 1.35);
    /// assert_eq!(params.mu(), 1.15);
    /// ```
    #[verified_engine::verified]
    pub fn new(lambda: f64, mu: f64) -> Result<Self, MatchModelError> {
        if lambda <= 0.0 || !lambda.is_finite() {
            return Err(MatchModelError::InvalidGoalRate { value: lambda });
        }
        if mu <= 0.0 || !mu.is_finite() {
            return Err(MatchModelError::InvalidGoalRate { value: mu });
        }
        Ok(Self { lambda, mu })
    }

    /// Expected home scoring rate $\lambda$.
    #[verified_engine::verified]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Expected away scoring rate $\mu$.
    #[verified_engine::verified]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Evaluates the bounded [`ScoreGrid`] with `DEFAULT_MAX_GOALS = 15` without renormalization.
    #[verified_engine::verified]
    pub fn default_grid(&self) -> Result<ScoreGrid, MatchModelError> {
        independent_poisson_grid(self, DEFAULT_MAX_GOALS, false)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawIndependentPoissonParams {
    lambda: f64,
    mu: f64,
}

impl<'de> Deserialize<'de> for IndependentPoissonParams {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawIndependentPoissonParams::deserialize(deserializer)?;
        Self::new(raw.lambda, raw.mu).map_err(serde::de::Error::custom)
    }
}

/// Evaluates $\ln(k!)$ using `statrs::function::gamma::ln_gamma`.
#[inline]
#[verified_engine::verified]
pub fn log_factorial(k: usize) -> f64 {
    if k <= 1 {
        0.0
    } else {
        ln_gamma((k + 1) as f64)
    }
}

/// Evaluates log-space univariate Poisson mass $\ln P(X = k \mid \lambda)$ for $\lambda \ge 0$.
#[inline]
#[verified_engine::verified]
pub fn poisson_log_pmf(lambda: f64, k: usize) -> f64 {
    if lambda == 0.0 {
        if k == 0 { 0.0 } else { f64::NEG_INFINITY }
    } else if k == 0 {
        -lambda
    } else {
        (k as f64) * lambda.ln() - lambda - log_factorial(k)
    }
}

/// Evaluates univariate Poisson probability mass $P(X = k \mid \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}$
/// in log-space without factorial overflow.
#[inline]
#[verified_engine::verified]
pub fn poisson_pmf(lambda: f64, k: usize) -> f64 {
    poisson_log_pmf(lambda, k).exp()
}

/// Populates stack marginal PMF table `out[0..=max_goals]` for a given Poisson rate `rate`.
#[verified_engine::verified]
pub(crate) fn fill_poisson_marginals(rate: f64, max_goals: usize, out: &mut [f64; GRID_STRIDE]) {
    for (k, slot) in out.iter_mut().enumerate().take(max_goals + 1) {
        *slot = poisson_pmf(rate, k);
    }
}

/// Builds a bounded 2D independent Poisson [`ScoreGrid`] with zero heap allocations.
#[verified_engine::verified]
pub fn independent_poisson_grid(
    params: &IndependentPoissonParams,
    max_goals: usize,
    normalize: bool,
) -> Result<ScoreGrid, MatchModelError> {
    if max_goals > MAX_GOALS_LIMIT {
        return Err(MatchModelError::InvalidMaxGoals {
            max_goals,
            limit: MAX_GOALS_LIMIT,
        });
    }
    let mut home_pmf = [0.0_f64; GRID_STRIDE];
    let mut away_pmf = [0.0_f64; GRID_STRIDE];
    fill_poisson_marginals(params.lambda, max_goals, &mut home_pmf);
    fill_poisson_marginals(params.mu, max_goals, &mut away_pmf);
    ScoreGrid::from_fn(max_goals, normalize, |home, away| {
        home_pmf[home] * away_pmf[away]
    })
}
