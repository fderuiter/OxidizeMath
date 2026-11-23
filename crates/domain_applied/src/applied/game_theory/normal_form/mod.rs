//! # Finite Normal-Form (Matrix & Bimatrix) Game Solvers
//!
//! Zero-allocation `f64` solvers for finite two-player strategic-form games:
//! - Iterated strict (`IESDS`) and weak (`IEWDS`) pure-strategy dominance (`iterated_dominance`)
//! - Two-player zero-sum minimax via shifted primal-dual Simplex (`solve_zero_sum_minimax`)
//! - Exact-support enumeration capped at `8x8` (`support_enumeration`)
//! - General bimatrix Nash equilibria via lexicographically perturbed Lemke-Howson (`lemke_howson_equilibrium`)
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_applied::__doctest_facade as math_explorer;
//! use math_explorer::applied::game_theory::normal_form::{
//!     BimatrixGameView, LemkeHowsonWorkspace, lemke_howson_equilibrium,
//! };
//!
//! let row_payoffs = [1.0, -1.0, -1.0, 1.0];
//! let col_payoffs = [-1.0, 1.0, 1.0, -1.0];
//! let game = BimatrixGameView::new(&row_payoffs, &col_payoffs, 2, 2).unwrap();
//! let mut ws = LemkeHowsonWorkspace::new(2, 2);
//! let mut p = [0.0; 2];
//! let mut q = [0.0; 2];
//! let eq = lemke_howson_equilibrium(game, 0, &mut ws, &mut p, &mut q).unwrap();
//! assert!((p[0] - 0.5).abs() < 1e-9 && (q[0] - 0.5).abs() < 1e-9);
//! assert!(eq.row_payoff.abs() < 1e-9);
//! ```

mod dominance;
mod lemke_howson;
mod minimax;
mod support_enum;

pub use dominance::{DominanceMode, DominanceResult, DominanceWorkspace, iterated_dominance};
pub use lemke_howson::{BimatrixPayoffs, LemkeHowsonWorkspace, lemke_howson_equilibrium};
pub use minimax::{MinimaxResult, MinimaxWorkspace, solve_zero_sum_minimax};
pub use support_enum::{MAX_SUPPORT_ENUM_DIM, SupportEquilibrium, support_enumeration};

use crate::error::GameTheoryError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH};

/// Borrowed view of a finite `rows x cols` two-player bimatrix game in row-major order.
#[derive(Debug, Clone, Copy)]
pub struct BimatrixGameView<'a> {
    /// Row player's payoff matrix $A$ (`rows * cols`, row-major).
    pub row_payoffs: &'a [f64],
    /// Column player's payoff matrix $B$ (`rows * cols`, row-major).
    pub col_payoffs: &'a [f64],
    /// Number of pure strategies for the row player.
    pub rows: usize,
    /// Number of pure strategies for the column player.
    pub cols: usize,
}

impl<'a> BimatrixGameView<'a> {
    /// Validates dimensions and payoff finiteness for a `rows x cols` bimatrix game.
    #[verified_engine::verified]
    pub fn new(
        row_payoffs: &'a [f64],
        col_payoffs: &'a [f64],
        rows: usize,
        cols: usize,
    ) -> Result<Self, GameTheoryError> {
        if rows == 0 || cols == 0 {
            return Err(GameTheoryError::EmptyStrategies);
        }
        let expected = rows * cols;
        if row_payoffs.len() != expected {
            return Err(GameTheoryError::DimensionMismatch {
                expected,
                actual: row_payoffs.len(),
            });
        }
        if col_payoffs.len() != expected {
            return Err(GameTheoryError::DimensionMismatch {
                expected,
                actual: col_payoffs.len(),
            });
        }
        for r in 0..rows {
            for c in 0..cols {
                let idx = flatten_2d_index(c, r, cols);
                let a = row_payoffs[idx];
                if !a.is_finite() {
                    return Err(GameTheoryError::InvalidPayoff {
                        row: r,
                        col: c,
                        value: a,
                    });
                }
                let b = col_payoffs[idx];
                if !b.is_finite() {
                    return Err(GameTheoryError::InvalidPayoff {
                        row: r,
                        col: c,
                        value: b,
                    });
                }
            }
        }
        Ok(Self {
            row_payoffs,
            col_payoffs,
            rows,
            cols,
        })
    }

    /// Reads $(A_{r, c}, B_{r, c})$ at `(row, col)`.
    #[inline]
    #[must_use]
    #[verified_engine::verified]
    pub fn payoff_pair(&self, row: usize, col: usize) -> (f64, f64) {
        let idx = flatten_2d_index(col, row, self.cols);
        (self.row_payoffs[idx], self.col_payoffs[idx])
    }
}

#[verified_engine::verified]
fn validate_mixed_strategy(probs: &[f64], expected_len: usize) -> Result<(), GameTheoryError> {
    if probs.len() != expected_len {
        return Err(GameTheoryError::DimensionMismatch {
            expected: expected_len,
            actual: probs.len(),
        });
    }
    let mut sum = 0.0;
    for (idx, &p) in probs.iter().enumerate() {
        if !p.is_finite() || p < -TOLERANCE_HIGH || p > 1.0 + TOLERANCE_FAST {
            return Err(GameTheoryError::InvalidPayoff {
                row: idx,
                col: 0,
                value: p,
            });
        }
        sum += p.max(0.0);
    }
    if (sum - 1.0).abs() > TOLERANCE_FAST {
        return Err(GameTheoryError::InvalidPayoff {
            row: 0,
            col: 0,
            value: sum,
        });
    }
    Ok(())
}

/// Computes `(row_payoff, col_payoff, max_unilateral_regret)` for a mixed strategy profile $(p, q)$.
#[verified_engine::verified]
pub fn evaluate_bimatrix_profile(
    game: BimatrixGameView<'_>,
    row_strategy: &[f64],
    col_strategy: &[f64],
) -> Result<(f64, f64, f64), GameTheoryError> {
    validate_mixed_strategy(row_strategy, game.rows)?;
    validate_mixed_strategy(col_strategy, game.cols)?;

    let mut row_payoff = 0.0;
    let mut best_row_dev = f64::NEG_INFINITY;
    for r in 0..game.rows {
        let mut expected_r = 0.0;
        for c in 0..game.cols {
            expected_r += game.row_payoffs[flatten_2d_index(c, r, game.cols)] * col_strategy[c];
        }
        row_payoff += row_strategy[r] * expected_r;
        best_row_dev = best_row_dev.max(expected_r);
    }

    let mut col_payoff = 0.0;
    let mut best_col_dev = f64::NEG_INFINITY;
    for c in 0..game.cols {
        let mut expected_c = 0.0;
        for r in 0..game.rows {
            expected_c += game.col_payoffs[flatten_2d_index(c, r, game.cols)] * row_strategy[r];
        }
        col_payoff += col_strategy[c] * expected_c;
        best_col_dev = best_col_dev.max(expected_c);
    }

    let max_regret = (best_row_dev - row_payoff)
        .max(best_col_dev - col_payoff)
        .max(0.0);
    Ok((row_payoff, col_payoff, max_regret))
}

#[cfg(test)]
mod tests;

// [cite:graph_parameters_rust]
