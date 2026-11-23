//! Zero-allocation primal-dual Simplex minimax solver for finite two-player zero-sum games.

use crate::error::GameTheoryError;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::registry::TOLERANCE_HIGH;

/// Summary returned by [`solve_zero_sum_minimax`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MinimaxResult {
    /// Unique saddle-point value $v^*$ of the zero-sum game (row player's expected payoff).
    pub game_value: f64,
    /// Number of Simplex pivots executed.
    pub pivots: usize,
}

/// Preallocated Simplex tableau workspace for [`solve_zero_sum_minimax`].
#[derive(Clone, Debug)]
pub struct MinimaxWorkspace {
    tableau: Vec<f64>,
    basis: Vec<usize>,
}

impl MinimaxWorkspace {
    /// Allocates tableau storage supporting zero-sum games up to `max_rows x max_cols`.
    #[must_use]
    pub fn new(max_rows: usize, max_cols: usize) -> Self {
        let tab_rows = max_rows + 1;
        let tab_cols = max_rows + max_cols + 1;
        Self {
            tableau: vec![0.0; tab_rows * tab_cols],
            basis: vec![0; max_rows],
        }
    }

    /// Expands tableau storage if needed outside verified zero-allocation regions.
    pub fn ensure_capacity(&mut self, rows: usize, cols: usize) {
        let req_tab = (rows + 1) * (rows + cols + 1);
        if self.tableau.len() < req_tab {
            self.tableau.resize(req_tab, 0.0);
        }
        if self.basis.len() < rows {
            self.basis.resize(rows, 0);
        }
    }
}

#[verified_engine::verified]
fn validate_zero_sum_inputs(
    payoffs: &[f64],
    rows: usize,
    cols: usize,
    workspace: &MinimaxWorkspace,
    row_out: &[f64],
    col_out: &[f64],
) -> Result<f64, GameTheoryError> {
    if rows == 0 || cols == 0 {
        return Err(GameTheoryError::EmptyStrategies);
    }
    let expected = rows * cols;
    if payoffs.len() != expected {
        return Err(GameTheoryError::DimensionMismatch {
            expected,
            actual: payoffs.len(),
        });
    }
    let req_tab = (rows + 1) * (rows + cols + 1);
    if workspace.tableau.len() < req_tab {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: req_tab,
            actual: workspace.tableau.len(),
        });
    }
    if workspace.basis.len() < rows {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: rows,
            actual: workspace.basis.len(),
        });
    }
    if row_out.len() < rows {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: rows,
            actual: row_out.len(),
        });
    }
    if col_out.len() < cols {
        return Err(GameTheoryError::WorkspaceTooSmall {
            required: cols,
            actual: col_out.len(),
        });
    }

    let mut min_val = f64::INFINITY;
    for r in 0..rows {
        for c in 0..cols {
            let v = payoffs[flatten_2d_index(c, r, cols)];
            if !v.is_finite() {
                return Err(GameTheoryError::InvalidPayoff {
                    row: r,
                    col: c,
                    value: v,
                });
            }
            min_val = min_val.min(v);
        }
    }
    Ok(if min_val < 1.0 { 1.0 - min_val } else { 0.0 })
}

#[verified_engine::verified]
fn init_minimax_tableau(
    payoffs: &[f64],
    rows: usize,
    cols: usize,
    shift: f64,
    tab: &mut [f64],
    basis: &mut [usize],
) {
    let width = rows + cols + 1;
    tab.fill(0.0);
    for r in 0..rows {
        for c in 0..cols {
            let shifted = payoffs[flatten_2d_index(c, r, cols)] + shift;
            tab[flatten_2d_index(c, r, width)] = shifted;
        }
        tab[flatten_2d_index(cols + r, r, width)] = 1.0;
        tab[flatten_2d_index(width - 1, r, width)] = 1.0;
        basis[r] = cols + r;
    }
    for c in 0..cols {
        tab[flatten_2d_index(c, rows, width)] = -1.0;
    }
}

#[verified_engine::verified]
fn select_lexicographic_leaving_row(
    tab: &[f64],
    rows: usize,
    cols: usize,
    enter: usize,
) -> Option<usize> {
    let width = rows + cols + 1;
    let rhs_col = width - 1;
    let mut best_row: Option<usize> = None;

    for r in 0..rows {
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
        for slack_idx in 0..rows {
            let col = cols + slack_idx;
            let lex_r = tab[flatten_2d_index(col, r, width)] / a_r;
            let lex_best = tab[flatten_2d_index(col, curr, width)] / a_best;
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
fn pivot_tableau(tab: &mut [f64], total_rows: usize, width: usize, leave: usize, enter: usize) {
    let inv_piv = 1.0 / tab[flatten_2d_index(enter, leave, width)];
    for col in 0..width {
        tab[flatten_2d_index(col, leave, width)] *= inv_piv;
    }
    tab[flatten_2d_index(enter, leave, width)] = 1.0;

    for r in 0..total_rows {
        if r == leave {
            continue;
        }
        let factor = tab[flatten_2d_index(enter, r, width)];
        if factor.abs() <= 0.0 {
            continue;
        }
        for col in 0..width {
            let leave_val = tab[flatten_2d_index(col, leave, width)];
            tab[flatten_2d_index(col, r, width)] -= factor * leave_val;
        }
        tab[flatten_2d_index(enter, r, width)] = 0.0;
    }
}

/// Solves a finite `rows x cols` zero-sum matrix game (row maximizes, column minimizes) via Simplex.
#[verified_engine::verified]
pub fn solve_zero_sum_minimax(
    payoffs: &[f64],
    rows: usize,
    cols: usize,
    workspace: &mut MinimaxWorkspace,
    row_strategy_out: &mut [f64],
    col_strategy_out: &mut [f64],
) -> Result<MinimaxResult, GameTheoryError> {
    let shift = validate_zero_sum_inputs(
        payoffs,
        rows,
        cols,
        workspace,
        row_strategy_out,
        col_strategy_out,
    )?;

    let total_rows = rows + 1;
    let width = rows + cols + 1;
    let tab = &mut workspace.tableau[..total_rows * width];
    let basis = &mut workspace.basis[..rows];
    init_minimax_tableau(payoffs, rows, cols, shift, tab, basis);

    let max_pivots = 64 * (rows + cols + 1) * (rows + cols + 1);
    let mut pivots = 0usize;
    loop {
        let mut enter_opt = None;
        for col in 0..(rows + cols) {
            if tab[flatten_2d_index(col, rows, width)] < -TOLERANCE_HIGH {
                enter_opt = Some(col);
                break;
            }
        }
        let Some(enter) = enter_opt else {
            break;
        };
        if pivots >= max_pivots {
            return Err(GameTheoryError::PivotLimitExceeded { max_pivots });
        }
        let leave = select_lexicographic_leaving_row(tab, rows, cols, enter)
            .ok_or(GameTheoryError::NoEquilibriumFound)?;
        pivot_tableau(tab, total_rows, width, leave, enter);
        basis[leave] = enter;
        pivots += 1;
    }

    let obj = tab[flatten_2d_index(width - 1, rows, width)];
    if obj <= TOLERANCE_HIGH {
        return Err(GameTheoryError::NoEquilibriumFound);
    }
    let shifted_value = 1.0 / obj;

    let row_s = &mut row_strategy_out[..rows];
    let col_s = &mut col_strategy_out[..cols];
    row_s.fill(0.0);
    col_s.fill(0.0);

    let mut row_sum = 0.0;
    for (r, p_r) in row_s.iter_mut().enumerate() {
        let y_r = tab[flatten_2d_index(cols + r, rows, width)].max(0.0);
        *p_r = y_r * shifted_value;
        row_sum += *p_r;
    }
    if row_sum > 0.0 {
        for p_r in row_s.iter_mut() {
            *p_r /= row_sum;
        }
    }

    let mut col_sum = 0.0;
    for (r, &var) in basis.iter().enumerate() {
        if var < cols {
            let x_c = tab[flatten_2d_index(width - 1, r, width)].max(0.0);
            col_s[var] = x_c * shifted_value;
            col_sum += col_s[var];
        }
    }
    if col_sum > 0.0 {
        for q_c in col_s.iter_mut() {
            *q_c /= col_sum;
        }
    }

    Ok(MinimaxResult {
        game_value: shifted_value - shift,
        pivots,
    })
}

// [cite:graph_parameters_rust]
