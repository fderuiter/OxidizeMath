//! Lexicographically perturbed Lemke-Howson complementary pivoting for bimatrix Nash equilibria.

use super::BimatrixGameView;
use crate::error::GameTheoryError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::TOLERANCE_HIGH;

/// Equilibrium payoff summary returned by [`lemke_howson_equilibrium`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BimatrixPayoffs {
    /// Row player's expected payoff at the computed Nash equilibrium.
    pub row_payoff: f64,
    /// Column player's expected payoff at the computed Nash equilibrium.
    pub col_payoff: f64,
    /// Number of complementary Lemke-Howson pivots executed.
    pub pivots: usize,
}

/// Preallocated complementary-pivoting tableau workspace for [`lemke_howson_equilibrium`].
#[derive(Clone, Debug)]
pub struct LemkeHowsonWorkspace {
    tableau: Vec<f64>,
    basis: Vec<usize>,
}

impl LemkeHowsonWorkspace {
    /// Allocates an `N x (2N + 1)` tableau (`N = max_rows + max_cols`) for Lemke-Howson pivoting.
    #[must_use]
    pub fn new(max_rows: usize, max_cols: usize) -> Self {
        let n = max_rows + max_cols;
        let width = 2 * n + 1;
        Self {
            tableau: vec![0.0; n * width],
            basis: vec![0; n],
        }
    }

    /// Expands tableau storage if needed outside verified zero-allocation regions.
    pub fn ensure_capacity(&mut self, rows: usize, cols: usize) {
        let n = rows + cols;
        let req_tab = n * (2 * n + 1);
        if self.tableau.len() < req_tab {
            self.tableau.resize(req_tab, 0.0);
        }
        if self.basis.len() < n {
            self.basis.resize(n, 0);
        }
    }
}

#[verified_engine::verified]
fn positive_payoff_shift(game: BimatrixGameView<'_>) -> f64 {
    let mut min_val = f64::INFINITY;
    for &a in game.row_payoffs {
        min_val = min_val.min(a);
    }
    for &b in game.col_payoffs {
        min_val = min_val.min(b);
    }
    if min_val < 1.0 { 1.0 - min_val } else { 0.0 }
}

#[verified_engine::verified]
fn init_lcp_tableau(game: BimatrixGameView<'_>, shift: f64, tab: &mut [f64], basis: &mut [usize]) {
    let m = game.rows;
    let n_cols = game.cols;
    let total = m + n_cols;
    let width = 2 * total + 1;

    tab.fill(0.0);
    for idx in 0..total {
        tab[flatten_2d_index(idx, idx, width)] = 1.0;
        tab[flatten_2d_index(2 * total, idx, width)] = 1.0;
        basis[idx] = idx;
    }

    // Rows 0..m: w_r + sum_c A'_{r,c} y_c = 1
    for r in 0..m {
        for c in 0..n_cols {
            let a_shifted = game.row_payoffs[flatten_2d_index(c, r, n_cols)] + shift;
            tab[flatten_2d_index(total + m + c, r, width)] = a_shifted;
        }
    }

    // Rows m..m+n: w_{m+c} + sum_r B'_{r,c} x_r = 1
    for c in 0..n_cols {
        for r in 0..m {
            let b_shifted = game.col_payoffs[flatten_2d_index(c, r, n_cols)] + shift;
            tab[flatten_2d_index(total + r, m + c, width)] = b_shifted;
        }
    }
}

#[verified_engine::verified]
fn select_lexicographic_pivot_row(tab: &[f64], total: usize, enter: usize) -> Option<usize> {
    let width = 2 * total + 1;
    let rhs_col = 2 * total;
    let mut best_row: Option<usize> = None;

    for r in 0..total {
        let a_r = tab[flatten_2d_index(enter, r, width)];
        if a_r <= TOLERANCE_HIGH {
            continue;
        }
        let Some(curr) = best_row else {
            best_row = Some(r);
            continue;
        };
        let a_best = tab[flatten_2d_index(enter, curr, width)];
        let ratio_r = tab[flatten_2d_index(rhs_col, r, width)] / a_r;
        let ratio_best = tab[flatten_2d_index(rhs_col, curr, width)] / a_best;
        if (ratio_r - ratio_best).abs() > TOLERANCE_HIGH {
            if ratio_r < ratio_best {
                best_row = Some(r);
            }
            continue;
        }

        // Columns 0..total hold B^{-1}, providing exact lexicographic perturbation coefficients.
        for pert_col in 0..total {
            let lex_r = tab[flatten_2d_index(pert_col, r, width)] / a_r;
            let lex_best = tab[flatten_2d_index(pert_col, curr, width)] / a_best;
            if (lex_r - lex_best).abs() > TOLERANCE_HIGH {
                if lex_r < lex_best {
                    best_row = Some(r);
                }
                break;
            }
        }
    }
    best_row
}

#[verified_engine::verified]
fn pivot_lcp_row(tab: &mut [f64], total: usize, leave: usize, enter: usize) {
    let width = 2 * total + 1;
    let inv_piv = 1.0 / tab[flatten_2d_index(enter, leave, width)];
    for col in 0..width {
        tab[flatten_2d_index(col, leave, width)] *= inv_piv;
    }
    tab[flatten_2d_index(enter, leave, width)] = 1.0;

    for r in 0..total {
        if r == leave {
            continue;
        }
        let factor = tab[flatten_2d_index(enter, r, width)];
        if factor.abs() <= 0.0 {
            continue;
        }
        for col in 0..width {
            let lv = tab[flatten_2d_index(col, leave, width)];
            tab[flatten_2d_index(col, r, width)] -= factor * lv;
        }
        tab[flatten_2d_index(enter, r, width)] = 0.0;
    }
}

#[verified_engine::verified]
fn extract_strategies_and_payoffs(
    game: BimatrixGameView<'_>,
    tab: &[f64],
    basis: &[usize],
    pivots: usize,
    row_out: &mut [f64],
    col_out: &mut [f64],
) -> Result<BimatrixPayoffs, GameTheoryError> {
    let m = game.rows;
    let n_cols = game.cols;
    let total = m + n_cols;
    let width = 2 * total + 1;
    let rhs_col = 2 * total;

    let p = &mut row_out[..m];
    let q = &mut col_out[..n_cols];
    p.fill(0.0);
    q.fill(0.0);

    let mut p_sum = 0.0;
    let mut q_sum = 0.0;
    for (r, &var) in basis.iter().enumerate() {
        if var >= total {
            let z_idx = var - total;
            let val = tab[flatten_2d_index(rhs_col, r, width)].max(0.0);
            if z_idx < m {
                p[z_idx] = val;
                p_sum += val;
            } else {
                q[z_idx - m] = val;
                q_sum += val;
            }
        }
    }

    if p_sum <= TOLERANCE_HIGH || q_sum <= TOLERANCE_HIGH {
        return Err(GameTheoryError::NoEquilibriumFound);
    }
    for pr in p.iter_mut() {
        *pr /= p_sum;
    }
    for qc in q.iter_mut() {
        *qc /= q_sum;
    }

    let mut row_payoff = 0.0;
    let mut col_payoff = 0.0;
    for (r, &pr) in p.iter().enumerate() {
        for (c, &qc) in q.iter().enumerate() {
            let idx = flatten_2d_index(c, r, n_cols);
            let weight = pr * qc;
            row_payoff += weight * game.row_payoffs[idx];
            col_payoff += weight * game.col_payoffs[idx];
        }
    }

    Ok(BimatrixPayoffs {
        row_payoff,
        col_payoff,
        pivots,
    })
}

/// Computes a Nash equilibrium of a finite `rows x cols` bimatrix game via lexicographically
/// perturbed Lemke-Howson pivoting starting from `initial_label` (`0..rows + cols`).
#[verified_engine::verified]
pub fn lemke_howson_equilibrium(
    game: BimatrixGameView<'_>,
    initial_label: usize,
    workspace: &mut LemkeHowsonWorkspace,
    row_strategy_out: &mut [f64],
    col_strategy_out: &mut [f64],
) -> Result<BimatrixPayoffs, GameTheoryError> {
    let total = game.rows + game.cols;
    let req_tab = total * (2 * total + 1);
    if initial_label >= total {
        return Err(GameTheoryError::DimensionMismatch {
            expected: total,
            actual: initial_label,
        });
    }
    if workspace.tableau.len() < req_tab {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: req_tab,
            actual: workspace.tableau.len(),
        });
    }
    if workspace.basis.len() < total {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: total,
            actual: workspace.basis.len(),
        });
    }
    if row_strategy_out.len() < game.rows {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: game.rows,
            actual: row_strategy_out.len(),
        });
    }
    if col_strategy_out.len() < game.cols {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: game.cols,
            actual: col_strategy_out.len(),
        });
    }

    let shift = positive_payoff_shift(game);
    let tab = &mut workspace.tableau[..req_tab];
    let basis = &mut workspace.basis[..total];
    init_lcp_tableau(game, shift, tab, basis);

    let max_pivots = 64 * (total + 1) * (total + 1);
    let mut enter = total + initial_label;
    let mut pivots = 0usize;

    loop {
        if pivots >= max_pivots {
            return Err(GameTheoryError::PivotLimitExceeded { max_pivots });
        }
        let leave_row = select_lexicographic_pivot_row(tab, total, enter)
            .ok_or(GameTheoryError::NoEquilibriumFound)?;
        let leaving_var = basis[leave_row];
        pivot_lcp_row(tab, total, leave_row, enter);
        basis[leave_row] = enter;
        pivots += 1;

        let dropped_or_recovered_label = leaving_var % total;
        if dropped_or_recovered_label == initial_label {
            break;
        }
        enter = if leaving_var < total {
            leaving_var + total
        } else {
            leaving_var - total
        };
    }

    extract_strategies_and_payoffs(game, tab, basis, pivots, row_strategy_out, col_strategy_out)
}

// [cite:graph_parameters_rust]
