//! Flat row-major slice adapters and [`UnitInterval`] probability aggregators for
//! Independent Poisson, Dixon-Coles (1997), and Karlis-Ntzoufras (2003) score grids.

use super::bivariate_poisson::{BivariatePoissonParams, bivariate_poisson_pmf};
use super::dixon_coles::dixon_coles_tau;
use super::poisson::poisson_pmf;
use crate::error::{CopulaError, ZipError};
use crate::pure_math::statistics::zip_regression::PoissonRate;
use math_commons::error::MathError;
use math_commons::math_kernel::types::{Dimension, flatten_2d_index};
use math_commons::primitives::UnitInterval;
use serde::{Deserialize, Serialize};

/// Default match-model family selector for UI and pricing pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoreModelKind {
    /// Independent Poisson marginal goal rates.
    IndependentPoisson,
    /// Dixon & Coles (1997) low-score $\tau$-adjusted Poisson model.
    DixonColes,
    /// Karlis & Ntzoufras (2003) shared-rate bivariate Poisson model.
    BivariatePoisson,
}

/// Low-score dependence parameter $\rho \in [-0.99, 0.99]$ in the Dixon-Coles (1997) model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DixonColesRho(f64);

impl DixonColesRho {
    /// Validates a Dixon-Coles dependence parameter $\rho \in [-0.99, 0.99]$.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, CopulaError> {
        if !(-0.99..=0.99).contains(&value) || !value.is_finite() {
            return Err(CopulaError::InvalidCorrelation { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw $\rho$ value.
    #[verified_engine::verified]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for DixonColesRho {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(-0.10)
    }
}

/// Shared covariance goal rate $\lambda_3 \ge 0$ in the Karlis-Ntzoufras (2003) bivariate Poisson model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SharedGoalRate(f64);

impl SharedGoalRate {
    /// Creates a non-negative finite shared scoring intensity $\lambda_3 \ge 0$.
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, ZipError> {
        if value < 0.0 || !value.is_finite() {
            return Err(ZipError::InvalidRate { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw $\lambda_3$ value.
    #[verified_engine::verified]
    pub fn value(self) -> f64 {
        self.0
    }
}

impl Default for SharedGoalRate {
    #[verified_engine::verified]
    fn default() -> Self {
        Self(0.15)
    }
}

/// Three-way (1X2) match outcome probabilities bounded in [`UnitInterval`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchOutcomeProbs {
    /// Probability that Home scores strictly more goals than Away ($H > A$).
    pub home_win: UnitInterval,
    /// Probability of a level scoreline ($H = A$).
    pub draw: UnitInterval,
    /// Probability that Away scores strictly more goals than Home ($A > H$).
    pub away_win: UnitInterval,
}

#[verified_engine::verified]
fn validate_matrix_slice(matrix_len: usize, max_goals: usize) -> Result<usize, MathError> {
    let cols = max_goals
        .checked_add(1)
        .ok_or(MathError::DimensionMismatch {
            expected: Dimension(usize::MAX),
            actual: Dimension(matrix_len),
        })?;
    let expected = cols.checked_mul(cols).ok_or(MathError::DimensionMismatch {
        expected: Dimension(usize::MAX),
        actual: Dimension(matrix_len),
    })?;
    if matrix_len != expected {
        return Err(MathError::DimensionMismatch {
            expected: Dimension(expected),
            actual: Dimension(matrix_len),
        });
    }
    Ok(cols)
}

#[verified_engine::verified]
fn normalize_grid(out_matrix: &mut [f64]) {
    let mut sum = 0.0;
    for val in out_matrix.iter() {
        sum += *val;
    }
    if sum > 0.0 && sum.is_finite() {
        let inv = 1.0 / sum;
        for val in out_matrix.iter_mut() {
            *val *= inv;
        }
    }
}

/// Populates a normalized `(max_goals + 1) x (max_goals + 1)` row-major score matrix under
/// independent Poisson goal processes without heap allocation.
#[verified_engine::verified]
pub fn independent_poisson_matrix_into(
    lambda_home: PoissonRate,
    lambda_away: PoissonRate,
    max_goals: usize,
    out_matrix: &mut [f64],
) -> Result<(), MathError> {
    let cols = validate_matrix_slice(out_matrix.len(), max_goals)?;
    let lh = lambda_home.value();
    let la = lambda_away.value();
    for home in 0..cols {
        let ph = poisson_pmf(lh, home);
        for away in 0..cols {
            let pa = poisson_pmf(la, away);
            out_matrix[flatten_2d_index(away, home, cols)] = ph * pa;
        }
    }
    normalize_grid(out_matrix);
    Ok(())
}

/// Populates a normalized `(max_goals + 1) x (max_goals + 1)` row-major score matrix under the
/// Dixon-Coles (1997) low-score correlation model without heap allocation.
#[verified_engine::verified]
pub fn dixon_coles_matrix_into(
    lambda_home: PoissonRate,
    lambda_away: PoissonRate,
    rho: DixonColesRho,
    max_goals: usize,
    out_matrix: &mut [f64],
) -> Result<(), MathError> {
    let cols = validate_matrix_slice(out_matrix.len(), max_goals)?;
    let lh = lambda_home.value();
    let la = lambda_away.value();
    let r = rho.value();
    for home in 0..cols {
        let ph = poisson_pmf(lh, home);
        for away in 0..cols {
            let pa = poisson_pmf(la, away);
            let tau = dixon_coles_tau(home, away, lh, la, r).max(0.0);
            out_matrix[flatten_2d_index(away, home, cols)] = ph * pa * tau;
        }
    }
    normalize_grid(out_matrix);
    Ok(())
}

/// Populates a normalized `(max_goals + 1) x (max_goals + 1)` row-major score matrix under the
/// Karlis-Ntzoufras (2003) bivariate Poisson model without heap allocation.
#[verified_engine::verified]
pub fn bivariate_poisson_matrix_into(
    lambda_1: PoissonRate,
    lambda_2: PoissonRate,
    lambda_3: SharedGoalRate,
    max_goals: usize,
    out_matrix: &mut [f64],
) -> Result<(), MathError> {
    let cols = validate_matrix_slice(out_matrix.len(), max_goals)?;
    let Ok(params) = BivariatePoissonParams::from_components(
        lambda_1.value(),
        lambda_2.value(),
        lambda_3.value(),
    ) else {
        return Err(MathError::OutOfBounds {
            value: lambda_3.value(),
            min: 0.0,
            max: f64::MAX,
        });
    };
    for home in 0..cols {
        for away in 0..cols {
            out_matrix[flatten_2d_index(away, home, cols)] =
                bivariate_poisson_pmf(&params, home, away);
        }
    }
    normalize_grid(out_matrix);
    Ok(())
}

/// Aggregates three-way (1X2) probabilities from a normalized `(max_goals + 1)^2` score matrix.
#[verified_engine::verified]
pub fn match_1x2_probabilities(
    matrix: &[f64],
    max_goals: usize,
) -> Result<MatchOutcomeProbs, MathError> {
    let cols = validate_matrix_slice(matrix.len(), max_goals)?;
    let mut home_win = 0.0;
    let mut draw = 0.0;
    let mut away_win = 0.0;
    for home in 0..cols {
        for away in 0..cols {
            let p = matrix[flatten_2d_index(away, home, cols)];
            if home > away {
                home_win += p;
            } else if home == away {
                draw += p;
            } else {
                away_win += p;
            }
        }
    }
    Ok(MatchOutcomeProbs {
        home_win: UnitInterval::new_clamped(home_win),
        draw: UnitInterval::new_clamped(draw),
        away_win: UnitInterval::new_clamped(away_win),
    })
}

/// Computes `(over_probability, under_probability)` for total goals $H + A$ relative to `total_line`
/// (e.g. `2.5`). For integer lines, pushes ($H + A == \text{line}$) are excluded from both tails.
#[verified_engine::verified]
pub fn over_under_probabilities(
    matrix: &[f64],
    max_goals: usize,
    total_line: f64,
) -> Result<(UnitInterval, UnitInterval), MathError> {
    let cols = validate_matrix_slice(matrix.len(), max_goals)?;
    let mut over = 0.0;
    let mut under = 0.0;
    for home in 0..cols {
        for away in 0..cols {
            let total = (home + away) as f64;
            let p = matrix[flatten_2d_index(away, home, cols)];
            if total > total_line + 1e-9 {
                over += p;
            } else if total + 1e-9 < total_line {
                under += p;
            }
        }
    }
    Ok((
        UnitInterval::new_clamped(over),
        UnitInterval::new_clamped(under),
    ))
}

/// Computes Both-Teams-To-Score ("BTTS Yes", $H \ge 1 \land A \ge 1$) probability as [`UnitInterval`].
#[verified_engine::verified]
pub fn both_teams_to_score_probability(
    matrix: &[f64],
    max_goals: usize,
) -> Result<UnitInterval, MathError> {
    let cols = validate_matrix_slice(matrix.len(), max_goals)?;
    let mut btts_yes = 0.0;
    for home in 1..cols {
        for away in 1..cols {
            btts_yes += matrix[flatten_2d_index(away, home, cols)];
        }
    }
    Ok(UnitInterval::new_clamped(btts_yes))
}

/// Computes the push-adjusted Asian Handicap cover probability for the Home side given `home_handicap`
/// (e.g. `-0.5`, `-1.0`, `+0.5`), returning $P(\text{win}) / (P(\text{win}) + P(\text{loss}))$.
#[verified_engine::verified]
pub fn asian_handicap_home_cover(
    matrix: &[f64],
    max_goals: usize,
    home_handicap: f64,
) -> Result<UnitInterval, MathError> {
    let cols = validate_matrix_slice(matrix.len(), max_goals)?;
    let mut win = 0.0;
    let mut loss = 0.0;
    for home in 0..cols {
        for away in 0..cols {
            let adj_diff = (home as f64) + home_handicap - (away as f64);
            let p = matrix[flatten_2d_index(away, home, cols)];
            if adj_diff > 1e-9 {
                win += p;
            } else if adj_diff < -1e-9 {
                loss += p;
            }
        }
    }
    let active = win + loss;
    if active <= 0.0 {
        return Ok(UnitInterval::new_clamped(0.5));
    }
    Ok(UnitInterval::new_clamped(win / active))
}
