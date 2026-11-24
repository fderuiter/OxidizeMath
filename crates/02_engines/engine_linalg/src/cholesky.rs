#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Strict symmetric positive-definite ($L L^T$) and positive-semidefinite Cholesky factorizations.

use crate::scalar::Scalar;
use crate::svd_jacobi::{symmetric_eigen_jacobi, symmetric_eigen_workspace_len};
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`cholesky_solve`] for an `n x n` SPD matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn cholesky_workspace_len(n: usize) -> usize {
    n * n
}

/// Workspace scalar capacity required by [`cholesky_semidefinite`] for an `n x n` PSD matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn cholesky_semidefinite_workspace_len(n: usize) -> usize {
    n + n * n + symmetric_eigen_workspace_len(n)
}

fn check_symmetric<T: Scalar>(a: &[T], n: usize, sym_tol: T) -> Result<(), LinalgError> {
    for i in 0..n {
        for j in (i + 1)..n {
            let diff = (a[mat_idx(i, j, n)] - a[mat_idx(j, i, n)]).abs();
            if diff > sym_tol {
                return Err(LinalgError::NotPositiveDefinite);
            }
        }
    }
    Ok(())
}

/// Computes the strict lower-triangular Cholesky factor $A = L L^T$ of a symmetric positive-definite matrix.
/// `[cite:engine_linalg]`
pub fn cholesky_factor<T: Scalar>(a: &[T], l: &mut [T], n: usize) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(a.len()) || l.len() != a.len() {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(a)?;
    let mut max_diag = T::ONE;
    for i in 0..n {
        max_diag = max_diag.max(a[mat_idx(i, i, n)].abs());
    }
    let sym_tol = T::DEFAULT_TOL_SCALE * T::EPSILON * max_diag;
    check_symmetric(a, n, sym_tol)?;

    l.fill(T::ZERO);
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[mat_idx(i, j, n)];
            for k in 0..j {
                sum -= l[mat_idx(i, k, n)] * l[mat_idx(j, k, n)];
            }
            if i == j {
                if sum <= T::ZERO || !sum.is_finite() {
                    return Err(LinalgError::NotPositiveDefinite);
                }
                l[mat_idx(i, i, n)] = sum.sqrt();
            } else {
                let diag = l[mat_idx(j, j, n)];
                l[mat_idx(i, j, n)] = sum / diag;
            }
        }
    }
    Ok(())
}

/// Solves $L L^T x = b$ given a lower-triangular Cholesky factor `l` from [`cholesky_factor`].
/// `[cite:engine_linalg]`
pub fn cholesky_solve_factored<T: Scalar>(
    l: &[T],
    b: &[T],
    x: &mut [T],
    n: usize,
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(l.len()) || b.len() != n || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(b)?;

    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[mat_idx(i, k, n)] * x[k];
        }
        let diag = l[mat_idx(i, i, n)];
        if diag <= T::ZERO {
            return Err(LinalgError::NotPositiveDefinite);
        }
        x[i] = sum / diag;
    }

    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= l[mat_idx(k, i, n)] * x[k];
        }
        x[i] = sum / l[mat_idx(i, i, n)];
    }
    Ok(())
}

/// Solves a symmetric positive-definite linear system $A x = b$ via strict Cholesky in `ws`.
/// `[cite:engine_linalg]`
pub fn cholesky_solve<T: Scalar>(
    a: &[T],
    b: &[T],
    x: &mut [T],
    n: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(a.len()) || b.len() != n || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let (l_buf, _) = ws.try_split(cholesky_workspace_len(n))?;
    cholesky_factor(a, l_buf, n)?;
    cholesky_solve_factored(l_buf, b, x, n)
}

/// Computes a positive-semidefinite lower-triangular Cholesky factor $A \approx L L^T$.
///
/// Validates symmetry within `sym_tol`, checks via symmetric Jacobi eigenvalues that
/// $\lambda_{\min}(A) \ge -\text{psd\_tol}$, and snaps rank-deficient diagonal residuals in
/// $[-\text{psd\_tol}, \text{psd\_tol}]$ to `0` while zeroing dependent sub-columns.
/// `[cite:engine_linalg]`
pub fn cholesky_semidefinite<T: Scalar>(
    a: &[T],
    l: &mut [T],
    n: usize,
    sym_tol: T,
    psd_tol: T,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<(), LinalgError> {
    if n.checked_mul(n) != Some(a.len()) || l.len() != a.len() {
        return Err(LinalgError::DimensionMismatch);
    }
    if sym_tol < T::ZERO || psd_tol < T::ZERO || !sym_tol.is_finite() || !psd_tol.is_finite() {
        return Err(LinalgError::InvalidInput);
    }
    check_all_finite(a)?;
    check_symmetric(a, n, sym_tol)?;

    let (eigvals, mut rest) = ws.try_split(n)?;
    let (eigvecs, mut eig_ws) = rest.try_split(n * n)?;
    symmetric_eigen_jacobi(a, n, eigvals, eigvecs, 64, &mut eig_ws)?;
    for &ev in eigvals.iter() {
        if ev < -psd_tol {
            return Err(LinalgError::NotPositiveDefinite);
        }
    }

    let snap_tol = psd_tol.min(T::from_f64(1e-12));
    l.fill(T::ZERO);
    for j in 0..n {
        let mut diag = a[mat_idx(j, j, n)];
        for k in 0..j {
            let l_jk = l[mat_idx(j, k, n)];
            diag -= l_jk * l_jk;
        }
        if diag < -psd_tol {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let l_jj = if diag <= snap_tol {
            T::ZERO
        } else {
            diag.sqrt()
        };
        l[mat_idx(j, j, n)] = l_jj;
        if l_jj > T::ZERO {
            for i in (j + 1)..n {
                let mut sum = a[mat_idx(i, j, n)];
                for k in 0..j {
                    sum -= l[mat_idx(i, k, n)] * l[mat_idx(j, k, n)];
                }
                l[mat_idx(i, j, n)] = sum / l_jj;
            }
        }
    }
    Ok(())
}
