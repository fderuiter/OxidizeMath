#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Symmetric Jacobi eigensolver and one-sided Hestenes-Jacobi Singular Value Decomposition.

use crate::scalar::Scalar;
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`symmetric_eigen_jacobi`] for an `n x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn symmetric_eigen_workspace_len(n: usize) -> usize {
    n * n
}

/// Workspace scalar capacity required by [`svd_jacobi`] for an `m x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn svd_jacobi_workspace_len(m: usize, n: usize) -> usize {
    if m >= n {
        m * n + n * n
    } else {
        2 * m * n + m * m + n * m
    }
}

fn rotate_cols<T: Scalar>(
    mat: &mut [T],
    nrows: usize,
    ncols: usize,
    p: usize,
    q: usize,
    c: T,
    s: T,
) {
    for r in 0..nrows {
        let mp = mat[mat_idx(r, p, ncols)];
        let mq = mat[mat_idx(r, q, ncols)];
        mat[mat_idx(r, p, ncols)] = c * mp - s * mq;
        mat[mat_idx(r, q, ncols)] = s * mp + c * mq;
    }
}

fn jacobi_sym_sweep<T: Scalar>(work: &mut [T], eigvecs: &mut [T], n: usize, tol: T) -> T {
    let mut max_off = T::ZERO;
    for p in 0..n {
        for q in (p + 1)..n {
            let apq = work[mat_idx(p, q, n)];
            max_off = max_off.max(apq.abs());
            if apq.abs() <= tol {
                continue;
            }
            let app = work[mat_idx(p, p, n)];
            let aqq = work[mat_idx(q, q, n)];
            let tau = (aqq - app) / (T::TWO * apq);
            let t = if tau >= T::ZERO {
                T::ONE / (tau + T::ONE.hypot(tau))
            } else {
                -T::ONE / (-tau + T::ONE.hypot(tau))
            };
            let c = T::ONE / T::ONE.hypot(t);
            let s = t * c;

            work[mat_idx(p, p, n)] = app - t * apq;
            work[mat_idx(q, q, n)] = aqq + t * apq;
            work[mat_idx(p, q, n)] = T::ZERO;
            work[mat_idx(q, p, n)] = T::ZERO;
            for r in 0..n {
                if r != p && r != q {
                    let arp = work[mat_idx(r, p, n)];
                    let arq = work[mat_idx(r, q, n)];
                    work[mat_idx(r, p, n)] = c * arp - s * arq;
                    work[mat_idx(p, r, n)] = work[mat_idx(r, p, n)];
                    work[mat_idx(r, q, n)] = s * arp + c * arq;
                    work[mat_idx(q, r, n)] = work[mat_idx(r, q, n)];
                }
            }
            rotate_cols(eigvecs, n, n, p, q, c, s);
        }
    }
    max_off
}

/// Computes all eigenvalues and orthonormal eigenvectors of a symmetric `n x n` matrix via cyclic Jacobi rotations.
/// `[cite:advanced_linear_algebra]`
pub fn symmetric_eigen_jacobi<T: Scalar>(
    a: &[T],
    n: usize,
    eigvals: &mut [T],
    eigvecs: &mut [T],
    max_sweeps: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(a.len()) || eigvals.len() != n || eigvecs.len() != a.len() {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let (work, _) = ws.try_split(symmetric_eigen_workspace_len(n))?;
    work.copy_from_slice(a);

    eigvecs.fill(T::ZERO);
    for i in 0..n {
        eigvecs[mat_idx(i, i, n)] = T::ONE;
    }
    if n <= 1 {
        if n == 1 {
            eigvals[0] = work[0];
        }
        return Ok(());
    }

    let mut scale = T::ZERO;
    for &v in work.iter() {
        scale = scale.max(v.abs());
    }
    let tol = T::DEFAULT_TOL_SCALE * T::EPSILON * scale.max(T::MIN_POSITIVE);

    let mut converged = false;
    for _ in 0..max_sweeps {
        if jacobi_sym_sweep(work, eigvecs, n, tol) <= tol {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NonConvergence);
    }
    for (i, slot) in eigvals.iter_mut().enumerate().take(n) {
        *slot = work[mat_idx(i, i, n)];
    }
    Ok(())
}

fn fill_orthogonal_zero_col<T: Scalar>(u: &mut [T], m: usize, n: usize, col: usize) {
    for basis in 0..m {
        for r in 0..m {
            u[mat_idx(r, col, n)] = if r == basis { T::ONE } else { T::ZERO };
        }
        for prev in 0..col {
            let mut dot = T::ZERO;
            for r in 0..m {
                dot += u[mat_idx(r, prev, n)] * u[mat_idx(r, col, n)];
            }
            for r in 0..m {
                let up = u[mat_idx(r, prev, n)];
                u[mat_idx(r, col, n)] -= dot * up;
            }
        }
        let mut norm_sq = T::ZERO;
        for r in 0..m {
            let val = u[mat_idx(r, col, n)];
            norm_sq += val * val;
        }
        let norm = norm_sq.sqrt();
        if norm > T::DEFAULT_TOL_SCALE * T::EPSILON {
            for r in 0..m {
                u[mat_idx(r, col, n)] /= norm;
            }
            return;
        }
    }
}

pub(crate) fn sort_thin_svd<T: Scalar>(
    u: &mut [T],
    m: usize,
    k: usize,
    s: &mut [T],
    v: &mut [T],
    n: usize,
    vt: &mut [T],
) {
    for i in 0..k {
        let mut best = i;
        for j in (i + 1)..k {
            if s[j] > s[best] {
                best = j;
            }
        }
        if best != i {
            s.swap(i, best);
            for r in 0..m {
                u.swap(mat_idx(r, i, k), mat_idx(r, best, k));
            }
            for r in 0..n {
                v.swap(mat_idx(r, i, k), mat_idx(r, best, k));
            }
        }
    }
    for i in 0..k {
        for j in 0..n {
            vt[mat_idx(i, j, n)] = v[mat_idx(j, i, k)];
        }
    }
}

fn hestenes_jacobi_sweep<T: Scalar>(w: &mut [T], v: &mut [T], m: usize, n: usize) -> bool {
    let mut clean_sweep = true;
    for p in 0..n {
        for q in (p + 1)..n {
            let mut alpha = T::ZERO;
            let mut beta = T::ZERO;
            let mut gamma = T::ZERO;
            for r in 0..m {
                let wp = w[mat_idx(r, p, n)];
                let wq = w[mat_idx(r, q, n)];
                alpha += wp * wp;
                beta += wq * wq;
                gamma += wp * wq;
            }
            let limit = T::DEFAULT_TOL_SCALE * T::EPSILON * (alpha * beta).sqrt();
            if gamma.abs() <= limit {
                continue;
            }
            clean_sweep = false;
            let zeta = (beta - alpha) / (T::TWO * gamma);
            let t = if zeta >= T::ZERO {
                T::ONE / (zeta + T::ONE.hypot(zeta))
            } else {
                -T::ONE / (-zeta + T::ONE.hypot(zeta))
            };
            let c = T::ONE / T::ONE.hypot(t);
            let sn = t * c;
            rotate_cols(w, m, n, p, q, c, sn);
            rotate_cols(v, n, n, p, q, c, sn);
        }
    }
    clean_sweep
}

fn svd_jacobi_tall<T: Scalar>(
    a: &[T],
    dims: (usize, usize),
    u: &mut [T],
    s: &mut [T],
    vt: &mut [T],
    max_sweeps: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    let (m, n) = dims;
    let (w, mut rest) = ws.try_split(m * n)?;
    let (v, _) = rest.try_split(n * n)?;
    w.copy_from_slice(a);
    v.fill(T::ZERO);
    for i in 0..n {
        v[mat_idx(i, i, n)] = T::ONE;
    }

    let mut converged = n <= 1;
    for _ in 0..max_sweeps {
        if converged || hestenes_jacobi_sweep(w, v, m, n) {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(LinalgError::NonConvergence);
    }

    let mut max_sigma = T::ZERO;
    for (j, sj) in s.iter_mut().enumerate().take(n) {
        let mut norm_sq = T::ZERO;
        for r in 0..m {
            let val = w[mat_idx(r, j, n)];
            norm_sq += val * val;
        }
        *sj = norm_sq.sqrt();
        max_sigma = max_sigma.max(*sj);
    }
    let zero_tol = T::from_usize(m.max(n)) * T::EPSILON * max_sigma;
    for (j, sj) in s.iter_mut().enumerate().take(n) {
        if *sj > zero_tol && *sj > T::ZERO {
            for r in 0..m {
                u[mat_idx(r, j, n)] = w[mat_idx(r, j, n)] / *sj;
            }
        } else {
            *sj = T::ZERO;
            fill_orthogonal_zero_col(u, m, n, j);
        }
    }
    sort_thin_svd(u, m, n, s, v, n, vt);
    Ok(())
}

/// Computes the thin Singular Value Decomposition $A = U \operatorname{diag}(S) V^T$ via one-sided Hestenes-Jacobi rotations.
///
/// Output dimensions for `dims = (m, n)` and $k = \min(m, n)$: `u` has shape `m x k`, `s` has
/// length `k` (non-increasing), and `vt` has shape `k x n`.
/// `[cite:advanced_linear_algebra]`
pub fn svd_jacobi<T: Scalar>(
    a: &[T],
    dims: (usize, usize),
    u: &mut [T],
    s: &mut [T],
    vt: &mut [T],
    max_sweeps: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    let (m, n) = dims;
    let k = m.min(n);
    if m.checked_mul(n) != Some(a.len())
        || m.checked_mul(k) != Some(u.len())
        || s.len() != k
        || k.checked_mul(n) != Some(vt.len())
    {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    if k == 0 {
        return Ok(());
    }
    if m >= n {
        return svd_jacobi_tall(a, (m, n), u, s, vt, max_sweeps, ws);
    }

    let (at, mut rest1) = ws.try_split(n * m)?;
    let (u_at, mut rest2) = rest1.try_split(n * m)?;
    let (vt_at, mut inner_ws) = rest2.try_split(m * m)?;
    for r in 0..m {
        for c in 0..n {
            at[mat_idx(c, r, m)] = a[mat_idx(r, c, n)];
        }
    }
    svd_jacobi_tall(at, (n, m), u_at, s, vt_at, max_sweeps, &mut inner_ws)?;
    for r in 0..m {
        for c in 0..m {
            u[mat_idx(r, c, m)] = vt_at[mat_idx(c, r, m)];
        }
    }
    for r in 0..m {
        for c in 0..n {
            vt[mat_idx(r, c, n)] = u_at[mat_idx(c, r, m)];
        }
    }
    Ok(())
}
