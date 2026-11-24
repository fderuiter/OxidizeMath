//! Zero-allocation Householder QR constrained stationary/steady-state linear solver.
// [cite:math_commons]

use super::{SolverWorkspace, tarjan};
use crate::math_kernel::types::flatten_2d_index;

const NEGATIVE_CLAMP_TOL: f64 = -1e-10;
const RESIDUAL_REL_TOL: f64 = 1e-12;
const DIAG_PIVOT_TOL: f64 = 1e-14;

/// Solves for the unique stationary/steady-state distribution on a slice `matrix` ($n \times n$).
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn solve_stationary_slice(
    ws: &mut SolverWorkspace,
    matrix: &[f64],
    n: usize,
    is_generator: bool,
    out_pi: &mut [f64],
) -> bool {
    debug_assert!(matrix.len() >= n * n && out_pi.len() == n);
    if !matrix.iter().all(|x| x.is_finite()) {
        return false;
    }
    let summary = tarjan::decompose_tarjan(ws, matrix, n);
    if summary.num_sink_components != 1 || summary.unique_sink_size == 0 {
        return false;
    }
    solve_on_unique_sink(
        ws,
        matrix,
        n,
        summary.unique_sink_size,
        is_generator,
        out_pi,
    )
}

/// Solves for the unique stationary/steady-state distribution on `ws.operator_buf[..n*n]`.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn solve_staged_stationary(
    ws: &mut SolverWorkspace,
    n: usize,
    is_generator: bool,
    out_pi: &mut [f64],
) -> bool {
    debug_assert!(ws.operator_buf.len() >= n * n && out_pi.len() == n);
    if !ws.operator_buf[..n * n].iter().all(|x| x.is_finite()) {
        return false;
    }
    // Temporarily swap out operator_buf slice reference via indices without allocating.
    let summary = decompose_staged(ws, n);
    if summary.num_sink_components != 1 || summary.unique_sink_size == 0 {
        return false;
    }
    let m = summary.unique_sink_size;
    out_pi.fill(0.0);
    if m == 1 {
        let s = ws.sink_members[0];
        out_pi[s] = 1.0;
        return verify_residual(&ws.operator_buf[..n * n], n, is_generator, 1.0, out_pi);
    }
    let Some(scale) = assemble_qr_from_buf(
        &ws.operator_buf[..n * n],
        n,
        &ws.sink_members[..m],
        is_generator,
        &mut ws.qr_matrix[..(m + 1) * m],
        &mut ws.qr_rhs[..m + 1],
    ) else {
        return false;
    };
    if !householder_qr_solve(
        &mut ws.qr_matrix[..(m + 1) * m],
        &mut ws.qr_rhs[..m + 1],
        &mut ws.qr_sol[..m],
        m + 1,
        m,
    ) {
        return false;
    }
    if !normalize_and_scatter(&ws.sink_members[..m], &ws.qr_sol[..m], out_pi) {
        return false;
    }
    verify_residual(&ws.operator_buf[..n * n], n, is_generator, scale, out_pi)
}

fn decompose_staged(ws: &mut SolverWorkspace, n: usize) -> super::SccSummary {
    let mut buf = core::mem::take(&mut ws.operator_buf);
    let summary = tarjan::decompose_tarjan(ws, &buf[..n * n], n);
    ws.operator_buf = core::mem::take(&mut buf);
    summary
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn solve_on_unique_sink(
    ws: &mut SolverWorkspace,
    matrix: &[f64],
    n: usize,
    m: usize,
    is_generator: bool,
    out_pi: &mut [f64],
) -> bool {
    debug_assert!(m >= 1 && m <= n);
    out_pi.fill(0.0);
    if m == 1 {
        let s = ws.sink_members[0];
        out_pi[s] = 1.0;
        return verify_residual(matrix, n, is_generator, 1.0, out_pi);
    }

    let Some(scale) = assemble_qr_from_buf(
        matrix,
        n,
        &ws.sink_members[..m],
        is_generator,
        &mut ws.qr_matrix[..(m + 1) * m],
        &mut ws.qr_rhs[..m + 1],
    ) else {
        return false;
    };

    if !householder_qr_solve(
        &mut ws.qr_matrix[..(m + 1) * m],
        &mut ws.qr_rhs[..m + 1],
        &mut ws.qr_sol[..m],
        m + 1,
        m,
    ) {
        return false;
    }

    if !normalize_and_scatter(&ws.sink_members[..m], &ws.qr_sol[..m], out_pi) {
        return false;
    }

    verify_residual(matrix, n, is_generator, scale, out_pi)
}

/// Assembles the $(m + 1) \times m$ scale-normalized constrained system:
/// rows $0..m$ hold $(P_C^T - I) / \alpha$ or $G_C^T / \alpha$ with RHS $0$, and row $m$ holds
/// $[1, \dots, 1]$ with RHS $1$.
#[cfg_attr(feature = "std", verified_engine::verified)]
fn assemble_qr_from_buf(
    matrix: &[f64],
    n: usize,
    sink: &[usize],
    is_generator: bool,
    qr_mat: &mut [f64],
    qr_rhs: &mut [f64],
) -> Option<f64> {
    let m = sink.len();
    debug_assert!(m >= 2 && qr_mat.len() >= (m + 1) * m && qr_rhs.len() > m);

    let mut max_abs = 0.0_f64;
    for (r, &sr) in sink.iter().enumerate() {
        for (c, &sc) in sink.iter().enumerate() {
            let mut val = matrix[flatten_2d_index(sr, sc, n)];
            if !is_generator && r == c {
                val -= 1.0;
            }
            qr_mat[flatten_2d_index(c, r, m)] = val;
            let a = libm::fabs(val);
            if a > max_abs {
                max_abs = a;
            }
        }
        qr_rhs[r] = 0.0;
    }

    if !max_abs.is_finite() || max_abs == 0.0 {
        return None;
    }

    let inv_scale = 1.0 / max_abs;
    for val in &mut qr_mat[..m * m] {
        *val *= inv_scale;
    }
    qr_mat[m * m..(m + 1) * m].fill(1.0);
    qr_rhs[m] = 1.0;
    Some(max_abs)
}

/// Solves a full-column-rank overdetermined `rows x cols` linear least-squares system in-place
/// via Householder reflections followed by upper-triangular back-substitution.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn householder_qr_solve(
    mat: &mut [f64],
    rhs: &mut [f64],
    sol: &mut [f64],
    rows: usize,
    cols: usize,
) -> bool {
    debug_assert!(
        rows >= cols && mat.len() >= rows * cols && rhs.len() >= rows && sol.len() >= cols
    );

    for k in 0..cols {
        if !apply_householder_column(mat, rhs, rows, cols, k) {
            return false;
        }
    }

    for i in (0..cols).rev() {
        let diag = mat[flatten_2d_index(i, i, cols)];
        if !diag.is_finite() || libm::fabs(diag) <= DIAG_PIVOT_TOL {
            return false;
        }
        let mut sum = rhs[i];
        for j in (i + 1)..cols {
            sum -= mat[flatten_2d_index(j, i, cols)] * sol[j];
        }
        sol[i] = sum / diag;
    }
    true
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn apply_householder_column(
    mat: &mut [f64],
    rhs: &mut [f64],
    rows: usize,
    cols: usize,
    k: usize,
) -> bool {
    debug_assert!(k < cols && cols <= rows);
    let mut tail_sq = 0.0_f64;
    for i in (k + 1)..rows {
        let v = mat[flatten_2d_index(k, i, cols)];
        tail_sq += v * v;
    }
    let x0 = mat[flatten_2d_index(k, k, cols)];
    let norm_x = libm::hypot(x0, libm::sqrt(tail_sq));
    if !norm_x.is_finite() || norm_x <= DIAG_PIVOT_TOL {
        return false;
    }
    if tail_sq == 0.0 {
        return true;
    }

    let alpha = if x0 >= 0.0 { -norm_x } else { norm_x };
    let v0 = x0 - alpha;
    let v_norm_sq = v0 * v0 + tail_sq;
    if v_norm_sq <= 0.0 || !v_norm_sq.is_finite() {
        return false;
    }
    let beta = 2.0 / v_norm_sq;

    for j in (k + 1)..cols {
        let kj = flatten_2d_index(j, k, cols);
        let mut dot = v0 * mat[kj];
        for i in (k + 1)..rows {
            dot += mat[flatten_2d_index(k, i, cols)] * mat[flatten_2d_index(j, i, cols)];
        }
        let factor = beta * dot;
        mat[kj] -= factor * v0;
        for i in (k + 1)..rows {
            let ik = mat[flatten_2d_index(k, i, cols)];
            mat[flatten_2d_index(j, i, cols)] -= factor * ik;
        }
    }

    let mut dot_b = v0 * rhs[k];
    for i in (k + 1)..rows {
        dot_b += mat[flatten_2d_index(k, i, cols)] * rhs[i];
    }
    let factor_b = beta * dot_b;
    rhs[k] -= factor_b * v0;
    for i in (k + 1)..rows {
        rhs[i] -= factor_b * mat[flatten_2d_index(k, i, cols)];
    }

    mat[flatten_2d_index(k, k, cols)] = alpha;
    true
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn normalize_and_scatter(sink: &[usize], sol: &[f64], out_pi: &mut [f64]) -> bool {
    debug_assert_eq!(sink.len(), sol.len());
    let mut sum = 0.0_f64;
    for &w in sol {
        if !w.is_finite() || w < NEGATIVE_CLAMP_TOL {
            return false;
        }
        if w > 0.0 {
            sum += w;
        }
    }
    if !sum.is_finite() || sum <= 0.0 {
        return false;
    }
    let inv_sum = 1.0 / sum;
    for (&state, &w) in sink.iter().enumerate().map(|(idx, s)| (s, &sol[idx])) {
        out_pi[state] = if w > 0.0 { w * inv_sum } else { 0.0 };
    }
    true
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn verify_residual(
    matrix: &[f64],
    n: usize,
    is_generator: bool,
    operator_scale: f64,
    pi: &[f64],
) -> bool {
    debug_assert!(pi.len() == n && matrix.len() >= n * n);
    let mut res_sq = 0.0_f64;
    let mut full_scale = operator_scale;
    for col in 0..n {
        let mut dot = 0.0_f64;
        for row in 0..n {
            let entry = matrix[flatten_2d_index(col, row, n)];
            let mag = libm::fabs(if !is_generator && row == col {
                entry - 1.0
            } else {
                entry
            });
            if mag > full_scale {
                full_scale = mag;
            }
            dot += pi[row] * entry;
        }
        let r = if is_generator { dot } else { dot - pi[col] };
        res_sq += r * r;
    }
    if full_scale == 0.0 {
        return res_sq <= RESIDUAL_REL_TOL * RESIDUAL_REL_TOL;
    }
    (libm::sqrt(res_sq) / full_scale) < RESIDUAL_REL_TOL
}
