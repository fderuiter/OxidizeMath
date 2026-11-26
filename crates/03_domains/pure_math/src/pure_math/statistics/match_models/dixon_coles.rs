//! Dixon-Coles (1997) low-score correlation adjustment on `(0,0)`, `(1,0)`, `(0,1)`, and `(1,1)`.

use super::grid::{DEFAULT_MAX_GOALS, GRID_STRIDE, MAX_GOALS_LIMIT, ScoreGrid};
use super::poisson::fill_poisson_marginals;
use crate::error::MatchModelError;
use serde::{Deserialize, Deserializer, Serialize};

/// Computes the structural Dixon-Coles correlation bounds
/// `max(-1.0 / (lambda * mu), -1.0) <= rho <= min(1.0 / lambda, 1.0 / mu)`.
#[verified_engine::verified]
pub fn dixon_coles_rho_bounds(lambda: f64, mu: f64) -> Result<(f64, f64), MatchModelError> {
    if lambda <= 0.0 || !lambda.is_finite() {
        return Err(MatchModelError::InvalidGoalRate { value: lambda });
    }
    if mu <= 0.0 || !mu.is_finite() {
        return Err(MatchModelError::InvalidGoalRate { value: mu });
    }
    let min_rho = (-1.0 / (lambda * mu)).max(-1.0);
    let max_rho = (1.0 / lambda).min(1.0 / mu);
    Ok((min_rho, max_rho))
}

/// Evaluates the Dixon-Coles low-score dependence multiplier $\tau_{\lambda,\mu}(x, y)$.
///
/// - `(0, 0)` -> `1.0 - lambda * mu * rho`
/// - `(0, 1)` -> `1.0 + lambda * rho`
/// - `(1, 0)` -> `1.0 + mu * rho`
/// - `(1, 1)` -> `1.0 - rho`
/// - All other scores -> `1.0`
#[inline]
#[verified_engine::verified]
pub fn dixon_coles_tau(home: usize, away: usize, lambda: f64, mu: f64, rho: f64) -> f64 {
    match (home, away) {
        (0, 0) => 1.0 - lambda * mu * rho,
        (0, 1) => 1.0 + lambda * rho,
        (1, 0) => 1.0 + mu * rho,
        (1, 1) => 1.0 - rho,
        _ => 1.0,
    }
}

/// Validated Dixon-Coles match parameters $(\lambda, \mu, \rho)$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct DixonColesParams {
    lambda: f64,
    mu: f64,
    rho: f64,
}

impl DixonColesParams {
    /// Creates validated Dixon-Coles parameters enforcing `lambda > 0`, `mu > 0`,
    /// `max(-1.0 / (lambda * mu), -1.0) <= rho <= min(1.0 / lambda, 1.0 / mu)`,
    /// and non-negative $\tau_{\lambda,\mu}$ across all four low-score cells.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::match_models::DixonColesParams;
    ///
    /// let dc = DixonColesParams::new(1.35, 1.15, -0.13).unwrap();
    /// let grid = dc.default_grid().unwrap();
    /// assert!((grid.prob(0, 0) - 0.098652).abs() < 1e-6);
    /// assert!((grid.prob(1, 1) - 0.144004).abs() < 1e-6);
    /// ```
    #[verified_engine::verified]
    pub fn new(lambda: f64, mu: f64, rho: f64) -> Result<Self, MatchModelError> {
        let (min_rho, max_rho) = dixon_coles_rho_bounds(lambda, mu)?;
        if !rho.is_finite() || rho < min_rho || rho > max_rho {
            return Err(MatchModelError::InvalidCorrelation {
                rho,
                min: min_rho,
                max: max_rho,
            });
        }
        let tau_min = (-1.0 / lambda).max(-1.0 / mu);
        let tau_max = (1.0 / (lambda * mu)).min(1.0);
        if rho < tau_min || rho > tau_max {
            return Err(MatchModelError::InvalidCorrelation {
                rho,
                min: min_rho.max(tau_min),
                max: max_rho.min(tau_max),
            });
        }
        Ok(Self { lambda, mu, rho })
    }

    /// Expected home goal rate $\lambda$.
    #[verified_engine::verified]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// Expected away goal rate $\mu$.
    #[verified_engine::verified]
    pub fn mu(&self) -> f64 {
        self.mu
    }

    /// Low-score dependence parameter $\rho$.
    #[verified_engine::verified]
    pub fn rho(&self) -> f64 {
        self.rho
    }

    /// Evaluates the bounded [`ScoreGrid`] with `DEFAULT_MAX_GOALS = 15` without renormalization.
    #[verified_engine::verified]
    pub fn default_grid(&self) -> Result<ScoreGrid, MatchModelError> {
        dixon_coles_grid(self, DEFAULT_MAX_GOALS, false)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDixonColesParams {
    lambda: f64,
    mu: f64,
    rho: f64,
}

impl<'de> Deserialize<'de> for DixonColesParams {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let raw = RawDixonColesParams::deserialize(deserializer)?;
        Self::new(raw.lambda, raw.mu, raw.rho).map_err(serde::de::Error::custom)
    }
}

/// Builds a bounded 2D Dixon-Coles [`ScoreGrid`] with zero heap allocations.
#[verified_engine::verified]
pub fn dixon_coles_grid(
    params: &DixonColesParams,
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
        let tau = dixon_coles_tau(home, away, params.lambda, params.mu, params.rho);
        (tau * home_pmf[home] * away_pmf[away]).max(0.0)
    })
}
