//! Exact-support Nash equilibrium enumeration for bimatrix games up to `8x8`.

use super::BimatrixGameView;
use crate::error::GameTheoryError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::{TOLERANCE_FAST, TOLERANCE_HIGH, TOLERANCE_STANDARD};

/// Maximum pure strategy count per player permitted by [`support_enumeration`].
pub const MAX_SUPPORT_ENUM_DIM: usize = 8;

const MAX_SYS_DIM: usize = MAX_SUPPORT_ENUM_DIM + 1;

/// Stack-inline Nash equilibrium returned by [`support_enumeration`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SupportEquilibrium {
    /// Number of row pure strategies in the game (`<= 8`).
    pub rows: usize,
    /// Number of column pure strategies in the game (`<= 8`).
    pub cols: usize,
    /// Row player's mixed strategy probabilities in `0..rows`.
    pub row_strategy: [f64; MAX_SUPPORT_ENUM_DIM],
    /// Column player's mixed strategy probabilities in `0..cols`.
    pub col_strategy: [f64; MAX_SUPPORT_ENUM_DIM],
    /// Row player's equilibrium expected payoff.
    pub row_payoff: f64,
    /// Column player's equilibrium expected payoff.
    pub col_payoff: f64,
}

impl SupportEquilibrium {
    /// Returns the active slice of row mixed-strategy probabilities.
    #[inline]
    #[must_use]
    pub fn row_probs(&self) -> &[f64] {
        &self.row_strategy[..self.rows]
    }

    /// Returns the active slice of column mixed-strategy probabilities.
    #[inline]
    #[must_use]
    pub fn col_probs(&self) -> &[f64] {
        &self.col_strategy[..self.cols]
    }
}

#[verified_engine::verified]
fn solve_small_linear_system(
    mat: &mut [f64; MAX_SYS_DIM * MAX_SYS_DIM],
    rhs: &mut [f64; MAX_SYS_DIM],
    dim: usize,
    sol: &mut [f64; MAX_SYS_DIM],
) -> bool {
    for col in 0..dim {
        let mut piv_row = col;
        let mut max_abs = mat[flatten_2d_index(col, col, dim)].abs();
        for r in (col + 1)..dim {
            let cand = mat[flatten_2d_index(col, r, dim)].abs();
            if cand > max_abs {
                max_abs = cand;
                piv_row = r;
            }
        }
        if max_abs <= TOLERANCE_HIGH {
            return false;
        }
        if piv_row != col {
            for c in col..dim {
                mat.swap(
                    flatten_2d_index(c, col, dim),
                    flatten_2d_index(c, piv_row, dim),
                );
            }
            rhs.swap(col, piv_row);
        }
        let inv_piv = 1.0 / mat[flatten_2d_index(col, col, dim)];
        for r in (col + 1)..dim {
            let factor = mat[flatten_2d_index(col, r, dim)] * inv_piv;
            if factor.abs() <= 0.0 {
                continue;
            }
            for c in col..dim {
                let top = mat[flatten_2d_index(c, col, dim)];
                mat[flatten_2d_index(c, r, dim)] -= factor * top;
            }
            rhs[r] -= factor * rhs[col];
        }
    }

    for r in (0..dim).rev() {
        let mut acc = rhs[r];
        for c in (r + 1)..dim {
            acc -= mat[flatten_2d_index(c, r, dim)] * sol[c];
        }
        sol[r] = acc / mat[flatten_2d_index(r, r, dim)];
        if !sol[r].is_finite() {
            return false;
        }
    }
    true
}

#[verified_engine::verified]
fn extract_mask_indices(
    mask: usize,
    limit: usize,
    out: &mut [usize; MAX_SUPPORT_ENUM_DIM],
) -> usize {
    let mut len = 0usize;
    for idx in 0..limit {
        if (mask & (1usize << idx)) != 0 {
            out[len] = idx;
            len += 1;
        }
    }
    len
}

#[verified_engine::verified]
fn solve_indifference_pair(
    game: BimatrixGameView<'_>,
    r_idx: &[usize],
    c_idx: &[usize],
    row_strategy: &mut [f64; MAX_SUPPORT_ENUM_DIM],
    col_strategy: &mut [f64; MAX_SUPPORT_ENUM_DIM],
) -> Option<(f64, f64)> {
    let k = r_idx.len();
    let dim = k + 1;
    let mut mat = [0.0; MAX_SYS_DIM * MAX_SYS_DIM];
    let mut rhs = [0.0; MAX_SYS_DIM];
    let mut sol = [0.0; MAX_SYS_DIM];

    // Solve A_{S1, S2} q - u * 1 = 0, sum(q) = 1
    for a in 0..k {
        for b in 0..k {
            mat[flatten_2d_index(b, a, dim)] =
                game.row_payoffs[flatten_2d_index(c_idx[b], r_idx[a], game.cols)];
        }
        mat[flatten_2d_index(k, a, dim)] = -1.0;
        rhs[a] = 0.0;
    }
    for b in 0..k {
        mat[flatten_2d_index(b, k, dim)] = 1.0;
    }
    rhs[k] = 1.0;

    if !solve_small_linear_system(&mut mat, &mut rhs, dim, &mut sol) {
        return None;
    }
    let row_val = sol[k];
    col_strategy.fill(0.0);
    let mut q_sum = 0.0;
    for b in 0..k {
        if sol[b] <= TOLERANCE_STANDARD {
            return None;
        }
        col_strategy[c_idx[b]] = sol[b];
        q_sum += sol[b];
    }
    for qc in col_strategy.iter_mut().take(game.cols) {
        *qc /= q_sum;
    }

    // Solve B_{S1, S2}^T p - v * 1 = 0, sum(p) = 1
    mat.fill(0.0);
    rhs.fill(0.0);
    for b in 0..k {
        for a in 0..k {
            mat[flatten_2d_index(a, b, dim)] =
                game.col_payoffs[flatten_2d_index(c_idx[b], r_idx[a], game.cols)];
        }
        mat[flatten_2d_index(k, b, dim)] = -1.0;
    }
    for a in 0..k {
        mat[flatten_2d_index(a, k, dim)] = 1.0;
    }
    rhs[k] = 1.0;

    if !solve_small_linear_system(&mut mat, &mut rhs, dim, &mut sol) {
        return None;
    }
    let col_val = sol[k];
    row_strategy.fill(0.0);
    let mut p_sum = 0.0;
    for a in 0..k {
        if sol[a] <= TOLERANCE_STANDARD {
            return None;
        }
        row_strategy[r_idx[a]] = sol[a];
        p_sum += sol[a];
    }
    for pr in row_strategy.iter_mut().take(game.rows) {
        *pr /= p_sum;
    }
    Some((row_val, col_val))
}

#[verified_engine::verified]
fn is_best_response_pair(
    game: BimatrixGameView<'_>,
    row_strategy: &[f64; MAX_SUPPORT_ENUM_DIM],
    col_strategy: &[f64; MAX_SUPPORT_ENUM_DIM],
    row_val: f64,
    col_val: f64,
) -> bool {
    for r in 0..game.rows {
        let mut u_r = 0.0;
        for (c, &qc) in col_strategy.iter().enumerate().take(game.cols) {
            u_r += game.row_payoffs[flatten_2d_index(c, r, game.cols)] * qc;
        }
        if u_r > row_val + TOLERANCE_FAST {
            return false;
        }
    }
    for c in 0..game.cols {
        let mut v_c = 0.0;
        for (r, &pr) in row_strategy.iter().enumerate().take(game.rows) {
            v_c += game.col_payoffs[flatten_2d_index(c, r, game.cols)] * pr;
        }
        if v_c > col_val + TOLERANCE_FAST {
            return false;
        }
    }
    true
}

#[verified_engine::verified]
fn is_duplicate_equilibrium(existing: &[SupportEquilibrium], cand: &SupportEquilibrium) -> bool {
    for eq in existing {
        let same_p = eq
            .row_probs()
            .iter()
            .zip(cand.row_probs())
            .all(|(a, b)| (a - b).abs() <= TOLERANCE_FAST);
        let same_q = eq
            .col_probs()
            .iter()
            .zip(cand.col_probs())
            .all(|(a, b)| (a - b).abs() <= TOLERANCE_FAST);
        if same_p && same_q {
            return true;
        }
    }
    false
}

/// Enumerates non-degenerate extreme Nash equilibria of a bimatrix game (`rows, cols <= 8`)
/// into `equilibria_out`, returning the number of equilibria written.
#[verified_engine::verified]
pub fn support_enumeration(
    game: BimatrixGameView<'_>,
    equilibria_out: &mut [SupportEquilibrium],
) -> Result<usize, GameTheoryError> {
    if game.rows > MAX_SUPPORT_ENUM_DIM || game.cols > MAX_SUPPORT_ENUM_DIM {
        return Err(GameTheoryError::SupportEnumerationCapExceeded {
            rows: game.rows,
            cols: game.cols,
            max_dim: MAX_SUPPORT_ENUM_DIM,
        });
    }
    if equilibria_out.is_empty() {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: 1,
            actual: 0,
        });
    }

    let max_k = game.rows.min(game.cols);
    let mut found = 0usize;
    let mut r_idx = [0usize; MAX_SUPPORT_ENUM_DIM];
    let mut c_idx = [0usize; MAX_SUPPORT_ENUM_DIM];
    let mut row_strategy = [0.0; MAX_SUPPORT_ENUM_DIM];
    let mut col_strategy = [0.0; MAX_SUPPORT_ENUM_DIM];

    for k in 1..=max_k {
        for r_mask in 1usize..(1usize << game.rows) {
            if r_mask.count_ones() as usize != k {
                continue;
            }
            let _ = extract_mask_indices(r_mask, game.rows, &mut r_idx);
            for c_mask in 1usize..(1usize << game.cols) {
                if c_mask.count_ones() as usize != k {
                    continue;
                }
                let _ = extract_mask_indices(c_mask, game.cols, &mut c_idx);
                let Some((u_val, v_val)) = solve_indifference_pair(
                    game,
                    &r_idx[..k],
                    &c_idx[..k],
                    &mut row_strategy,
                    &mut col_strategy,
                ) else {
                    continue;
                };
                if !is_best_response_pair(game, &row_strategy, &col_strategy, u_val, v_val) {
                    continue;
                }
                let cand = SupportEquilibrium {
                    rows: game.rows,
                    cols: game.cols,
                    row_strategy,
                    col_strategy,
                    row_payoff: u_val,
                    col_payoff: v_val,
                };
                if !is_duplicate_equilibrium(&equilibria_out[..found], &cand) {
                    equilibria_out[found] = cand;
                    found += 1;
                    if found == equilibria_out.len() {
                        return Ok(found);
                    }
                }
            }
        }
    }

    if found == 0 {
        return Err(GameTheoryError::NoEquilibriumFound);
    }
    Ok(found)
}

// [cite:graph_parameters_rust]
