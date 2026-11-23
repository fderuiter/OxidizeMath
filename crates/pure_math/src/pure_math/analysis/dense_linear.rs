//! Shared zero-allocation dense linear system solvers for nonlinear root-finding
//! and implicit/Rosenbrock ODE integrators.

use math_commons::math_kernel::types::flatten_2d_index;

/// Normalizes IEEE-754 negative zero (`-0.0`) to canonical positive zero (`0.0`).
#[verified_engine::verified]
#[inline]
pub(crate) fn canonical_zero(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// Checks whether `a` and `b` are finite and have strictly opposite signs without calling `signum`.
#[verified_engine::verified]
#[inline]
pub(crate) fn has_strict_sign_change(a: f64, b: f64) -> bool {
    (a < 0.0 && b > 0.0) || (a > 0.0 && b < 0.0)
}

/// Solves the `n x n` row-major linear system `mat * x = rhs` in-place via Gaussian elimination
/// with partial pivoting, writing the solution vector into `rhs[..n]`.
///
/// Returns `false` if `n == 0`, any entry is non-finite, or a pivot magnitude falls below `pivot_tol`.
#[verified_engine::verified]
pub(crate) fn solve_linear_in_place(
    mat: &mut [f64],
    rhs: &mut [f64],
    n: usize,
    pivot_tol: f64,
) -> bool {
    if n == 0 || mat.len() < n * n || rhs.len() < n {
        return false;
    }
    for k in 0..n {
        let mut pivot_row = k;
        let mut max_abs = mat[flatten_2d_index(k, k, n)].abs();
        if !max_abs.is_finite() {
            return false;
        }
        for i in (k + 1)..n {
            let v = mat[flatten_2d_index(k, i, n)].abs();
            if !v.is_finite() {
                return false;
            }
            if v > max_abs {
                max_abs = v;
                pivot_row = i;
            }
        }
        if max_abs <= pivot_tol {
            return false;
        }
        if pivot_row != k {
            swap_matrix_rows(mat, n, k, pivot_row, k);
            rhs.swap(k, pivot_row);
        }
        if !eliminate_below_pivot(mat, rhs, n, k) {
            return false;
        }
    }
    back_substitute_upper(mat, rhs, n)
}

#[verified_engine::verified]
fn swap_matrix_rows(mat: &mut [f64], n: usize, r1: usize, r2: usize, start_col: usize) {
    for col in start_col..n {
        mat.swap(flatten_2d_index(col, r1, n), flatten_2d_index(col, r2, n));
    }
}

#[verified_engine::verified]
fn eliminate_below_pivot(mat: &mut [f64], rhs: &mut [f64], n: usize, k: usize) -> bool {
    let pivot = mat[flatten_2d_index(k, k, n)];
    for i in (k + 1)..n {
        let idx_ik = flatten_2d_index(k, i, n);
        let factor = mat[idx_ik] / pivot;
        mat[idx_ik] = 0.0;
        for j in (k + 1)..n {
            let idx_ij = flatten_2d_index(j, i, n);
            mat[idx_ij] -= factor * mat[flatten_2d_index(j, k, n)];
        }
        rhs[i] -= factor * rhs[k];
        if !rhs[i].is_finite() {
            return false;
        }
    }
    true
}

#[verified_engine::verified]
fn back_substitute_upper(mat: &[f64], rhs: &mut [f64], n: usize) -> bool {
    for idx in 0..n {
        let i = n - 1 - idx;
        let mut sum = rhs[i];
        for (j, &xj) in rhs.iter().enumerate().take(n).skip(i + 1) {
            sum -= mat[flatten_2d_index(j, i, n)] * xj;
        }
        let x_i = sum / mat[flatten_2d_index(i, i, n)];
        if !x_i.is_finite() {
            return false;
        }
        rhs[i] = x_i;
    }
    true
}

/// Computes an in-place `PA = LU` factorization of the `n x n` row-major matrix `lu`,
/// recording row permutations in `pivots[..n]`.
#[verified_engine::verified]
pub(crate) fn lu_factor_in_place(
    lu: &mut [f64],
    pivots: &mut [usize],
    n: usize,
    pivot_tol: f64,
) -> bool {
    if n == 0 || lu.len() < n * n || pivots.len() < n {
        return false;
    }
    for (idx, p) in pivots.iter_mut().take(n).enumerate() {
        *p = idx;
    }
    for k in 0..n {
        let mut pivot_row = k;
        let mut max_abs = lu[flatten_2d_index(k, k, n)].abs();
        if !max_abs.is_finite() {
            return false;
        }
        for i in (k + 1)..n {
            let v = lu[flatten_2d_index(k, i, n)].abs();
            if !v.is_finite() {
                return false;
            }
            if v > max_abs {
                max_abs = v;
                pivot_row = i;
            }
        }
        if max_abs <= pivot_tol {
            return false;
        }
        if pivot_row != k {
            swap_matrix_rows(lu, n, k, pivot_row, 0);
            pivots.swap(k, pivot_row);
        }
        let diag = lu[flatten_2d_index(k, k, n)];
        for i in (k + 1)..n {
            let idx_ik = flatten_2d_index(k, i, n);
            let mult = lu[idx_ik] / diag;
            lu[idx_ik] = mult;
            for j in (k + 1)..n {
                let idx_ij = flatten_2d_index(j, i, n);
                lu[idx_ij] -= mult * lu[flatten_2d_index(j, k, n)];
            }
        }
    }
    true
}

/// Solves `A * x = b` in-place using a pre-factored `PA = LU` decomposition from [`lu_factor_in_place`].
///
/// `scratch[..n]` stores the permuted vector `P * b` before forward/backward substitution writes
/// the solution into `rhs[..n]`.
#[verified_engine::verified]
pub(crate) fn lu_solve_in_place(
    lu: &[f64],
    pivots: &[usize],
    rhs: &mut [f64],
    scratch: &mut [f64],
    n: usize,
) -> bool {
    if n == 0 || lu.len() < n * n || pivots.len() < n || rhs.len() < n || scratch.len() < n {
        return false;
    }
    for i in 0..n {
        scratch[i] = rhs[pivots[i]];
    }
    for (i, &si) in scratch.iter().enumerate().take(n) {
        let mut sum = si;
        for (j, &xj) in rhs.iter().enumerate().take(i) {
            sum -= lu[flatten_2d_index(j, i, n)] * xj;
        }
        if !sum.is_finite() {
            return false;
        }
        rhs[i] = sum;
    }
    back_substitute_upper(lu, rhs, n)
}

// [cite:advanced_linear_algebra]
