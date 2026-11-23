//! Internal tableau pivoting and Bland's anti-cycling helpers for Two-Phase Simplex.

use super::linear_programming::{ConstraintSense, LinearProgram, LpSense, SimplexWorkspace};
use nalgebra::DMatrix;

#[verified_engine::verified]
pub(super) fn pivot_tableau(
    tab: &mut DMatrix<f64>,
    basis: &mut [usize],
    p_row: usize,
    p_col: usize,
    cols: usize,
) {
    let inv_piv = 1.0 / tab[(p_row, p_col)];
    for c in 0..cols {
        tab[(p_row, c)] *= inv_piv;
    }
    tab[(p_row, p_col)] = 1.0;
    let rows = tab.nrows();
    for r in 0..rows {
        if r == p_row {
            continue;
        }
        let factor = tab[(r, p_col)];
        if factor.abs() > 0.0 {
            for c in 0..cols {
                tab[(r, c)] -= factor * tab[(p_row, c)];
            }
            tab[(r, p_col)] = 0.0;
        }
    }
    basis[p_row] = p_col;
}

pub(super) fn bland_entering_col(
    tab: &DMatrix<f64>,
    active: &[bool],
    obj_row: usize,
    tol: f64,
) -> Option<usize> {
    (0..active.len()).find(|&c| active[c] && tab[(obj_row, c)] < -tol)
}

pub(super) fn bland_leaving_row(
    tab: &DMatrix<f64>,
    basis: &[usize],
    m: usize,
    enter_col: usize,
    rhs_col: usize,
    tol: f64,
) -> Option<usize> {
    let mut best_row: Option<usize> = None;
    let mut best_ratio = f64::INFINITY;
    for r in 0..m {
        let a_rc = tab[(r, enter_col)];
        if a_rc > tol {
            let ratio = tab[(r, rhs_col)].max(0.0) / a_rc;
            let update = match best_row {
                None => true,
                Some(prev) => {
                    if (ratio - best_ratio).abs() <= tol * (1.0 + best_ratio) {
                        basis[r] < basis[prev]
                    } else {
                        ratio < best_ratio
                    }
                }
            };
            if update {
                best_ratio = ratio;
                best_row = Some(r);
            }
        }
    }
    best_row
}

pub(super) fn record_primal_vertex(ws: &mut SimplexWorkspace, m: usize, n: usize, rhs_col: usize) {
    ws.primal.fill(0.0);
    for r in 0..m {
        let var = ws.basis[r];
        if var < n {
            ws.primal[var] = ws.tableau[(r, rhs_col)].max(0.0);
        }
    }
    if ws.pivot_path.len() < ws.pivot_path.capacity() {
        let x0 = if n >= 1 { ws.primal[0] } else { 0.0 };
        let x1 = if n >= 2 { ws.primal[1] } else { 0.0 };
        let pt = [x0, x1];
        let duplicate = ws
            .pivot_path
            .last()
            .is_some_and(|prev| (prev[0] - pt[0]).hypot(prev[1] - pt[1]) < 1e-11);
        if !duplicate {
            ws.pivot_path.push(pt);
        }
    }
}

pub(super) fn setup_phase_one(lp: &LinearProgram, ws: &mut SimplexWorkspace) -> (usize, usize) {
    let n = lp.c.len();
    let m = lp.b.len();
    let total_cols = n + 2 * m + 1;
    let rhs_col = n + 2 * m;
    ws.tableau.fill(0.0);
    ws.col_active.fill(false);
    ws.col_active[..n].fill(true);
    for i in 0..m {
        let sign = if lp.b[i] < 0.0 { -1.0 } else { 1.0 };
        ws.row_signs[i] = sign;
        for j in 0..n {
            ws.tableau[(i, j)] = sign * lp.a[(i, j)];
        }
        match lp.senses[i] {
            ConstraintSense::LessEqual => {
                ws.tableau[(i, n + i)] = sign;
                ws.col_active[n + i] = true;
            }
            ConstraintSense::GreaterEqual => {
                ws.tableau[(i, n + i)] = -sign;
                ws.col_active[n + i] = true;
            }
            ConstraintSense::Equal => {}
        }
        let art_col = n + m + i;
        ws.tableau[(i, art_col)] = 1.0;
        ws.tableau[(i, rhs_col)] = sign * lp.b[i];
        let slack_is_unit = ws.col_active[n + i] && ws.tableau[(i, n + i)] > 0.5;
        if slack_is_unit {
            ws.basis[i] = n + i;
        } else {
            ws.basis[i] = art_col;
            ws.col_active[art_col] = true;
            for c in 0..=rhs_col {
                ws.tableau[(m, c)] -= ws.tableau[(i, c)];
            }
            ws.tableau[(m, art_col)] = 0.0;
        }
    }
    (total_cols, rhs_col)
}

pub(super) fn drive_artificials_out(
    ws: &mut SimplexWorkspace,
    n: usize,
    m: usize,
    total_cols: usize,
    tol: f64,
) {
    let art_start = n + m;
    for r in 0..m {
        if ws.basis[r] >= art_start
            && let Some(col) =
                (0..art_start).find(|&c| ws.col_active[c] && ws.tableau[(r, c)].abs() > tol)
        {
            pivot_tableau(&mut ws.tableau, &mut ws.basis, r, col, total_cols);
        }
    }
    ws.col_active[art_start..(n + 2 * m)].fill(false);
}

pub(super) fn setup_phase_two_objective(
    lp: &LinearProgram,
    ws: &mut SimplexWorkspace,
    total_cols: usize,
) {
    let n = lp.c.len();
    let m = lp.b.len();
    for c in 0..total_cols {
        ws.tableau[(m, c)] = 0.0;
    }
    let dir = match lp.sense {
        LpSense::Maximize => 1.0,
        LpSense::Minimize => -1.0,
    };
    for j in 0..n {
        ws.tableau[(m, j)] = -dir * lp.c[j];
    }
    for r in 0..m {
        let bv = ws.basis[r];
        if bv < n {
            let coeff = ws.tableau[(m, bv)];
            if coeff.abs() > 0.0 {
                for c in 0..total_cols {
                    ws.tableau[(m, c)] -= coeff * ws.tableau[(r, c)];
                }
                ws.tableau[(m, bv)] = 0.0;
            }
        }
    }
}
