//! Derivative match-market probability aggregators (1X2, Over/Under totals, BTTS, and Asian Handicap)
//! operating on flat row-major score matrices indexed via [`flatten_2d_index`].

use super::score_matrix::validate_matrix_slice;
use math_commons::error::MathError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::primitives::UnitInterval;

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
    let mut btts = 0.0;
    for home in 1..cols {
        for away in 1..cols {
            btts += matrix[flatten_2d_index(away, home, cols)];
        }
    }
    Ok(UnitInterval::new_clamped(btts))
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
