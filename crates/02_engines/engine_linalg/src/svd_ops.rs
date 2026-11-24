#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! SVD-backed pseudoinverse, low-rank approximation, effective rank, and QR-with-SVD-fallback least-squares solvers.

use crate::qr::{qr_least_squares_workspace_len, qr_solve_least_squares};
use crate::scalar::Scalar;
use crate::svd_bidiag::{svd_bidiagonal, svd_bidiagonal_workspace_len};
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`svd_pseudoinverse`], [`svd_low_rank_approx`], and [`svd_solve_least_squares`].
/// `[cite:engine_linalg]`
#[inline]
pub const fn svd_ops_workspace_len(m: usize, n: usize) -> usize {
    let k = if m < n { m } else { n };
    m * k + k + k * n + svd_bidiagonal_workspace_len(m, n)
}

/// Workspace scalar capacity required by [`qr_or_svd_solve`].
/// `[cite:engine_linalg]`
#[inline]
pub const fn qr_or_svd_workspace_len(m: usize, n: usize) -> usize {
    let qr_len = qr_least_squares_workspace_len(m, n);
    let svd_len = svd_ops_workspace_len(m, n);
    if qr_len > svd_len { qr_len } else { svd_len }
}

/// Counts singular values strictly exceeding `tol` via [`svd_bidiagonal`].
/// `[cite:advanced_linear_algebra]`
pub fn svd_effective_rank<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    tol: T,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<usize, LinalgError> {
    if !tol.is_finite() || tol < T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    let k = m.min(n);
    if m.checked_mul(n) != Some(a.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    if k == 0 {
        return Ok(0);
    }
    let (u, mut r1) = ws.try_split(m * k)?;
    let (s, mut r2) = r1.try_split(k)?;
    let (vt, mut inner_ws) = r2.try_split(k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;
    Ok(s.iter().filter(|&&sigma| sigma > tol).count())
}

/// Computes the Moore-Penrose pseudoinverse $A^+ = V \Sigma^+ U^T \in \mathbb{R}^{n \times m}$ with cutoff `epsilon`.
/// `[cite:advanced_linear_algebra]`
pub fn svd_pseudoinverse<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    epsilon: T,
    pinv: &mut [T],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    if !epsilon.is_finite() || epsilon < T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    if m.checked_mul(n) != Some(a.len()) || n.checked_mul(m) != Some(pinv.len()) {
        return Err(LinalgError::DimensionMismatch);
    }
    let k = m.min(n);
    pinv.fill(T::ZERO);
    if k == 0 {
        return Ok(());
    }

    let (u, mut r1) = ws.try_split(m * k)?;
    let (s, mut r2) = r1.try_split(k)?;
    let (vt, mut inner_ws) = r2.try_split(k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;

    for idx in 0..k {
        let sigma = s[idx];
        if sigma <= epsilon {
            continue;
        }
        let inv_sigma = T::ONE / sigma;
        for r in 0..n {
            let v_r_idx = vt[mat_idx(idx, r, n)] * inv_sigma;
            for c in 0..m {
                pinv[mat_idx(r, c, m)] += v_r_idx * u[mat_idx(c, idx, k)];
            }
        }
    }
    Ok(())
}

/// Computes the Eckart-Young rank-`target_rank` approximation $\sum_{i=0}^{r-1} \sigma_i u_i v_i^T$.
/// `[cite:advanced_linear_algebra]`
pub fn svd_low_rank_approx<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    target_rank: usize,
    approx: &mut [T],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    if m.checked_mul(n) != Some(a.len()) || approx.len() != a.len() {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let k = m.min(n);
    approx.fill(T::ZERO);
    let r_lim = target_rank.min(k);
    if r_lim == 0 {
        return Ok(());
    }

    let (u, mut r1) = ws.try_split(m * k)?;
    let (s, mut r2) = r1.try_split(k)?;
    let (vt, mut inner_ws) = r2.try_split(k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;

    for idx in 0..r_lim {
        let sigma = s[idx];
        for r in 0..m {
            let scaled_u = sigma * u[mat_idx(r, idx, k)];
            for c in 0..n {
                approx[mat_idx(r, c, n)] += scaled_u * vt[mat_idx(idx, c, n)];
            }
        }
    }
    Ok(())
}

/// Computes the minimum-norm least-squares solution $x = V \Sigma^+ U^T b$ via [`svd_bidiagonal`].
/// `[cite:advanced_linear_algebra]`
pub fn svd_solve_least_squares<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    b: &[T],
    tol: T,
    x: &mut [T],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<usize, LinalgError> {
    if !tol.is_finite() || tol < T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    if m.checked_mul(n) != Some(a.len()) || b.len() != m || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(b)?;
    let k = m.min(n);
    x.fill(T::ZERO);
    if k == 0 {
        return Ok(0);
    }

    let (u, mut r1) = ws.try_split(m * k)?;
    let (s, mut r2) = r1.try_split(k)?;
    let (vt, mut inner_ws) = r2.try_split(k * n)?;
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;

    let cutoff = tol.max(T::from_usize(m.max(n)) * T::EPSILON * s[0]);
    let mut rank = 0;
    for idx in 0..k {
        let sigma = s[idx];
        if sigma <= cutoff || sigma == T::ZERO {
            continue;
        }
        rank += 1;
        let mut ut_b = T::ZERO;
        for r in 0..m {
            ut_b += u[mat_idx(r, idx, k)] * b[r];
        }
        let coeff = ut_b / sigma;
        for c in 0..n {
            x[c] += coeff * vt[mat_idx(idx, c, n)];
        }
    }
    Ok(rank)
}

/// Solves $\min_x \|A x - b\|_2$ via Householder QR when full column rank, falling back to SVD when rank-deficient.
/// `[cite:advanced_linear_algebra]`
pub fn qr_or_svd_solve<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    b: &[T],
    x: &mut [T],
    col_perm: &mut [usize],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<usize, LinalgError> {
    if m.checked_mul(n) != Some(a.len()) || b.len() != m || x.len() != n || col_perm.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    if m >= n {
        let rank = qr_solve_least_squares(a, m, n, b, x, col_perm, ws)?;
        if rank == n {
            return Ok(rank);
        }
    }
    svd_solve_least_squares(a, m, n, b, T::ZERO, x, ws)
}
