#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! SVD-backed pseudoinverse, low-rank approximation, effective rank, and QR-with-SVD-fallback least-squares solvers.

use crate::qr::{qr_least_squares_workspace_len, qr_solve_least_squares};
use crate::scalar::Scalar;
use crate::svd_bidiag::{svd_bidiagonal, svd_bidiagonal_workspace_len};
use crate::types::{
    ByteArena, LinalgError, SolverWorkspace, carve_scalars, check_all_finite, mat_idx,
};

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
    ws: &mut (impl SolverWorkspace + ?Sized),
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
    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let u = carve_scalars::<T>(&mut cur, m * k)?;
    let s = carve_scalars::<T>(&mut cur, k)?;
    let vt = carve_scalars::<T>(&mut cur, k * n)?;
    let mut inner_ws = ByteArena::new(cur.into_bytes());
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
    ws: &mut (impl SolverWorkspace + ?Sized),
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

    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let u = carve_scalars::<T>(&mut cur, m * k)?;
    let s = carve_scalars::<T>(&mut cur, k)?;
    let vt = carve_scalars::<T>(&mut cur, k * n)?;
    let mut inner_ws = ByteArena::new(cur.into_bytes());
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;

    for (idx, &sigma) in s.iter().enumerate().take(k) {
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
    ws: &mut (impl SolverWorkspace + ?Sized),
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

    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let u = carve_scalars::<T>(&mut cur, m * k)?;
    let s = carve_scalars::<T>(&mut cur, k)?;
    let vt = carve_scalars::<T>(&mut cur, k * n)?;
    let mut inner_ws = ByteArena::new(cur.into_bytes());
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;

    for (idx, &sigma) in s.iter().enumerate().take(r_lim) {
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
///
/// Reference: Golub & Kahan (1965), DOI: 10.1137/0702016;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:advanced_linear_algebra]`
///
/// # Stability
/// Orthogonal bidiagonalization and implicit Wilkinson-shift QR sweeps achieve backward error
/// $O(\max(m,n)\varepsilon_{\text{mach}}\|A\|_2)$ while zeroing singular modes $\sigma_i \le \max(\text{tol}, \max(m,n)\varepsilon_{\text{mach}}\sigma_1)$.
///
/// # Domain of Validity
/// Requires finite `a` (`m * n`), finite `b` (`m`), finite `tol >= 0`, `x` (`n`), and `ws`
/// capacity at least [`svd_ops_workspace_len`]`(m, n)`.
pub fn svd_solve_least_squares<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    b: &[T],
    tol: T,
    x: &mut [T],
    ws: &mut (impl SolverWorkspace + ?Sized),
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

    let mut sub = ws.reborrow();
    let mut cur = sub.cursor();
    let u = carve_scalars::<T>(&mut cur, m * k)?;
    let s = carve_scalars::<T>(&mut cur, k)?;
    let vt = carve_scalars::<T>(&mut cur, k * n)?;
    let mut inner_ws = ByteArena::new(cur.into_bytes());
    svd_bidiagonal(a, m, n, u, s, vt, &mut inner_ws)?;

    let cutoff = tol.max(T::from_usize(m.max(n)) * T::EPSILON * s[0]);
    let mut rank = 0;
    for (idx, &sigma) in s.iter().enumerate().take(k) {
        if sigma <= cutoff || sigma == T::ZERO {
            continue;
        }
        rank += 1;
        let mut ut_b = T::ZERO;
        for (r, &br) in b.iter().enumerate().take(m) {
            ut_b += u[mat_idx(r, idx, k)] * br;
        }
        let coeff = ut_b / sigma;
        for (c, xc) in x.iter_mut().enumerate().take(n) {
            *xc += coeff * vt[mat_idx(idx, c, n)];
        }
    }
    Ok(rank)
}

/// Solves $\min_x \|A x - b\|_2$ via Householder QR when full column rank, falling back to minimum-norm SVD when rank-deficient.
///
/// Reference: Golub & Van Loan (2013), DOI: 10.56021/9781421407944;
/// [`docs/math_specs/engine_linalg.md`](../../../../docs/math_specs/engine_linalg.md). `[cite:advanced_linear_algebra]`
///
/// # Stability
/// Combines fast $O(m n^2)$ orthogonal Householder QR on well-conditioned full-rank systems with
/// minimum-2-norm Moore-Penrose SVD regularization whenever $m < n$ or numerical rank $< n$.
///
/// # Domain of Validity
/// Requires finite `a` (`m * n`), finite `b` (`m`), `x` (`n`), `col_perm` (`n`), and `ws`
/// capacity at least [`qr_or_svd_workspace_len`]`(m, n)`.
pub fn qr_or_svd_solve<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    b: &[T],
    x: &mut [T],
    col_perm: &mut [usize],
    ws: &mut (impl SolverWorkspace + ?Sized),
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
