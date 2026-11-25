//! Zero-allocation Householder QR constrained stationary/steady-state linear solver.
// [cite:math_commons]

use super::{SolverWorkspace, tarjan};
use crate::math_kernel::types::flatten_2d_index;
use math_scalars::RealField;

#[inline]
fn negative_clamp_tol<T: RealField>() -> T {
    -((T::EPSILON * T::from_f64(65_536.0)).max(T::from_f64(1e-10)))
}

#[inline]
fn residual_rel_tol<T: RealField>() -> T {
    (T::EPSILON * T::from_f64(4096.0)).max(T::from_f64(1e-12))
}

#[inline]
fn diag_pivot_tol<T: RealField>() -> T {
    (T::EPSILON * T::from_f64(64.0)).max(T::from_f64(1e-14))
}

/// Solves for the unique stationary/steady-state distribution on a slice `matrix` ($n \times n$).
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn solve_stationary_slice<T: RealField>(
    ws: &mut SolverWorkspace<T>,
    matrix: &[T],
    n: usize,
    is_generator: bool,
    out_pi: &mut [T],
) -> bool {
    debug_assert!(matrix.len() >= n * n && out_pi.len() == n);
    if !matrix.iter().copied().all(T::is_finite) {
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
pub(super) fn solve_staged_stationary<T: RealField>(
    ws: &mut SolverWorkspace<T>,
    n: usize,
    is_generator: bool,
    out_pi: &mut [T],
) -> bool {
    debug_assert!(ws.operator_buf.len() >= n * n && out_pi.len() == n);
    if !ws.operator_buf[..n * n].iter().copied().all(T::is_finite) {
        return false;
    }
    let summary = decompose_staged(ws, n);
    if summary.num_sink_components != 1 || summary.unique_sink_size == 0 {
        return false;
    }
    let m = summary.unique_sink_size;
    out_pi.fill(T::ZERO);
    if m == 1 {
        let s = ws.sink_members[0];
        out_pi[s] = T::ONE;
        return verify_residual(&ws.operator_buf[..n * n], n, is_generator, T::ONE, out_pi);
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

fn decompose_staged<T: RealField>(ws: &mut SolverWorkspace<T>, n: usize) -> super::SccSummary {
    let mut buf = core::mem::take(&mut ws.operator_buf);
    let summary = tarjan::decompose_tarjan(ws, &buf[..n * n], n);
    ws.operator_buf = core::mem::take(&mut buf);
    summary
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn solve_on_unique_sink<T: RealField>(
    ws: &mut SolverWorkspace<T>,
    matrix: &[T],
    n: usize,
    m: usize,
    is_generator: bool,
    out_pi: &mut [T],
) -> bool {
    debug_assert!(m >= 1 && m <= n);
    out_pi.fill(T::ZERO);
    if m == 1 {
        let s = ws.sink_members[0];
        out_pi[s] = T::ONE;
        return verify_residual(matrix, n, is_generator, T::ONE, out_pi);
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
fn assemble_qr_from_buf<T: RealField>(
    matrix: &[T],
    n: usize,
    sink: &[usize],
    is_generator: bool,
    qr_mat: &mut [T],
    qr_rhs: &mut [T],
) -> Option<T> {
    let m = sink.len();
    debug_assert!(m >= 2 && qr_mat.len() >= (m + 1) * m && qr_rhs.len() > m);

    let mut max_abs = T::ZERO;
    for (r, &sr) in sink.iter().enumerate() {
        for (c, &sc) in sink.iter().enumerate() {
            let mut val = matrix[flatten_2d_index(sr, sc, n)];
            if !is_generator && r == c {
                val -= T::ONE;
            }
            qr_mat[flatten_2d_index(c, r, m)] = val;
            let a = val.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        qr_rhs[r] = T::ZERO;
    }

    if !max_abs.is_finite() || max_abs == T::ZERO {
        return None;
    }

    let inv_scale = T::ONE / max_abs;
    for val in &mut qr_mat[..m * m] {
        *val *= inv_scale;
    }
    qr_mat[m * m..(m + 1) * m].fill(T::ONE);
    qr_rhs[m] = T::ONE;
    Some(max_abs)
}

/// Solves a full-column-rank overdetermined `rows x cols` linear least-squares system in-place
/// via Householder reflections followed by upper-triangular back-substitution.
#[cfg_attr(feature = "std", verified_engine::verified)]
pub(super) fn householder_qr_solve<T: RealField>(
    mat: &mut [T],
    rhs: &mut [T],
    sol: &mut [T],
    rows: usize,
    cols: usize,
) -> bool {
    debug_assert!(
        rows >= cols && mat.len() >= rows * cols && rhs.len() >= rows && sol.len() >= cols
    );
    let piv_tol = diag_pivot_tol::<T>();

    for k in 0..cols {
        if !apply_householder_column(mat, rhs, rows, cols, k) {
            return false;
        }
    }

    for i in (0..cols).rev() {
        let diag = mat[flatten_2d_index(i, i, cols)];
        if !diag.is_finite() || diag.abs() <= piv_tol {
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
fn apply_householder_column<T: RealField>(
    mat: &mut [T],
    rhs: &mut [T],
    rows: usize,
    cols: usize,
    k: usize,
) -> bool {
    debug_assert!(k < cols && cols <= rows);
    let mut tail_sq = T::ZERO;
    for i in (k + 1)..rows {
        let v = mat[flatten_2d_index(k, i, cols)];
        tail_sq += v * v;
    }
    let x0 = mat[flatten_2d_index(k, k, cols)];
    let norm_x = x0.hypot(tail_sq.sqrt());
    if !norm_x.is_finite() || norm_x <= diag_pivot_tol::<T>() {
        return false;
    }
    if tail_sq == T::ZERO {
        return true;
    }

    let alpha = if x0 >= T::ZERO { -norm_x } else { norm_x };
    let v0 = x0 - alpha;
    let v_norm_sq = v0 * v0 + tail_sq;
    if v_norm_sq <= T::ZERO || !v_norm_sq.is_finite() {
        return false;
    }
    let beta = T::from_f64(2.0) / v_norm_sq;

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
fn normalize_and_scatter<T: RealField>(sink: &[usize], sol: &[T], out_pi: &mut [T]) -> bool {
    debug_assert_eq!(sink.len(), sol.len());
    let clamp_tol = negative_clamp_tol::<T>();
    let mut sum = T::ZERO;
    for &w in sol {
        if !w.is_finite() || w < clamp_tol {
            return false;
        }
        if w > T::ZERO {
            sum += w;
        }
    }
    if !sum.is_finite() || sum <= T::ZERO {
        return false;
    }
    let inv_sum = T::ONE / sum;
    for (&state, &w) in sink.iter().enumerate().map(|(idx, s)| (s, &sol[idx])) {
        out_pi[state] = if w > T::ZERO { w * inv_sum } else { T::ZERO };
    }
    true
}

#[cfg_attr(feature = "std", verified_engine::verified)]
fn verify_residual<T: RealField>(
    matrix: &[T],
    n: usize,
    is_generator: bool,
    operator_scale: T,
    pi: &[T],
) -> bool {
    debug_assert!(pi.len() == n && matrix.len() >= n * n);
    let rel_tol = residual_rel_tol::<T>();
    let mut res_sq = T::ZERO;
    let mut full_scale = operator_scale;
    for col in 0..n {
        let mut dot = T::ZERO;
        for row in 0..n {
            let entry = matrix[flatten_2d_index(col, row, n)];
            let mag = (if !is_generator && row == col {
                entry - T::ONE
            } else {
                entry
            })
            .abs();
            if mag > full_scale {
                full_scale = mag;
            }
            dot += pi[row] * entry;
        }
        let r = if is_generator { dot } else { dot - pi[col] };
        res_sq += r * r;
    }
    if full_scale == T::ZERO {
        return res_sq <= rel_tol * rel_tol;
    }
    (res_sq.sqrt() / full_scale) < rel_tol
}
