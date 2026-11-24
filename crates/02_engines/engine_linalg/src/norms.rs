#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Matrix norms (`L1`, `Linf`, Frobenius, spectral) and 1-norm / spectral condition-number estimators.

use crate::lu::{lu_factor_in_place, lu_solve_factored, lu_solve_transpose_factored};
use crate::scalar::Scalar;
use crate::svd_bidiag::{svd_bidiagonal, svd_bidiagonal_workspace_len};
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`norm_spectral`] and [`cond_spectral`] for an `m x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn spectral_workspace_len(m: usize, n: usize) -> usize {
    let k = if m < n { m } else { n };
    m * k + k + k * n + svd_bidiagonal_workspace_len(m, n)
}

/// Workspace scalar capacity required by [`cond_l1`] for an `n x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn cond_l1_workspace_len(n: usize) -> usize {
    n * n + 4 * n
}

/// Computes the induced 1-norm $\|A\|_1 = \max_j \sum_i |A_{ij}|$.
/// `[cite:engine_linalg]`
pub fn norm_l1<T: Scalar>(a: &[T], m: usize, n: usize) -> Result<T, LinalgError> {
    if m.checked_mul(n) != Some(a.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let mut best = T::ZERO;
    for c in 0..n {
        let mut col_sum = T::ZERO;
        for r in 0..m {
            col_sum += a[mat_idx(r, c, n)].abs();
        }
        best = best.max(col_sum);
    }
    Ok(best)
}

/// Computes the induced infinity-norm $\|A\|_\infty = \max_i \sum_j |A_{ij}|$.
/// `[cite:engine_linalg]`
pub fn norm_linf<T: Scalar>(a: &[T], m: usize, n: usize) -> Result<T, LinalgError> {
    if m.checked_mul(n) != Some(a.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let mut best = T::ZERO;
    for r in 0..m {
        let mut row_sum = T::ZERO;
        for c in 0..n {
            row_sum += a[mat_idx(r, c, n)].abs();
        }
        best = best.max(row_sum);
    }
    Ok(best)
}

/// Computes the scaled overflow-safe Frobenius norm $\|A\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$.
/// `[cite:engine_linalg]`
pub fn norm_frobenius<T: Scalar>(a: &[T], m: usize, n: usize) -> Result<T, LinalgError> {
    if m.checked_mul(n) != Some(a.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    for &val in a {
        let av = val.abs();
        if av > T::ZERO {
            if scale < av {
                let ratio = scale / av;
                ssq = T::ONE + ssq * ratio * ratio;
                scale = av;
            } else {
                let ratio = av / scale;
                ssq += ratio * ratio;
            }
        }
    }
    Ok(scale * ssq.sqrt())
}

/// Computes the spectral 2-norm $\|A\|_2 = \sigma_{\max}(A)$ via [`svd_bidiagonal`].
/// `[cite:advanced_linear_algebra]`
pub fn norm_spectral<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<T, LinalgError> {
    let k = m.min(n);
    if m.checked_mul(n) != Some(a.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    if k == 0 {
        return Ok(T::ZERO);
    }
    let (u, mut r1) = ws.try_split(m * k)?;
    let (s, mut r2) = r1.try_split(k)?;
    let (vt, mut inner_ws) = r2.try_split(k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;
    Ok(s[0])
}

/// Computes the spectral 2-norm condition number $\kappa_2(A) = \sigma_{\max}(A) / \sigma_{\min}(A)$.
/// `[cite:advanced_linear_algebra]`
pub fn cond_spectral<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<T, LinalgError> {
    let k = m.min(n);
    if k == 0 || m.checked_mul(n) != Some(a.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let (u, mut r1) = ws.try_split(m * k)?;
    let (s, mut r2) = r1.try_split(k)?;
    let (vt, mut inner_ws) = r2.try_split(k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;
    let s_max = s[0];
    let s_min = s[k - 1];
    let tol = T::from_usize(m.max(n)) * T::EPSILON * s_max;
    if s_min <= tol || s_min == T::ZERO {
        return Err(LinalgError::SingularMatrix);
    }
    Ok(s_max / s_min)
}

/// Estimates the 1-norm condition number $\kappa_1(A) = \|A\|_1 \|A^{-1}\|_1$ via Hager-Higham iteration over LU.
/// `[cite:advanced_linear_algebra]`
pub fn cond_l1<T: Scalar>(
    a: &[T],
    n: usize,
    pivots: &mut [usize],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<T, LinalgError> {
    if n == 0 || n.checked_mul(n) != Some(a.len()) || pivots.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let a_norm1 = norm_l1(a, n, n)?;
    if a_norm1 == T::ZERO {
        return Err(LinalgError::SingularMatrix);
    }

    let (lu, mut r1) = ws.try_split(n * n)?;
    let (x, mut r2) = r1.try_split(n)?;
    let (xi, mut r3) = r2.try_split(n)?;
    let (z, mut r4) = r3.try_split(n)?;
    let (work, _) = r4.try_split(n)?;

    lu.copy_from_slice(a);
    lu_factor_in_place(lu, n, pivots, T::ZERO)?;

    let inv_n = T::ONE / T::from_usize(n);
    x.fill(inv_n);
    let mut gamma = T::ZERO;

    for step in 0..5 {
        lu_solve_factored(lu, n, pivots, x, z)?;
        let mut z_l1 = T::ZERO;
        for i in 0..n {
            z_l1 += z[i].abs();
            xi[i] = if z[i] >= T::ZERO { T::ONE } else { -T::ONE };
        }
        if z_l1 <= gamma && step > 0 {
            break;
        }
        gamma = z_l1;
        lu_solve_transpose_factored(lu, n, pivots, xi, z, work)?;

        let mut best_j = 0;
        let mut best_z = z[0].abs();
        let mut z_dot_x = T::ZERO;
        for j in 0..n {
            let az = z[j].abs();
            if az > best_z {
                best_z = az;
                best_j = j;
            }
            z_dot_x += z[j] * x[j];
        }
        if step > 0 && best_z <= z_dot_x {
            break;
        }
        x.fill(T::ZERO);
        x[best_j] = T::ONE;
    }

    Ok(a_norm1 * gamma)
}
