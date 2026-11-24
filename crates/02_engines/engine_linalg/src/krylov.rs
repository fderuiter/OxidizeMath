#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Preallocated zero-allocation Conjugate Gradient (CG) and restarted GMRES Krylov solvers.

use crate::scalar::Scalar;
use crate::sparse::CsrMatrixView;
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Convergence summary returned by [`cg_solve`] and [`gmres_solve`].
#[derive(Copy, Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KrylovReport<T> {
    /// Total inner Krylov iterations executed.
    pub iterations: usize,
    /// Final Euclidean residual norm $\|b - A x\|_2$.
    pub residual_norm: T,
}

/// Workspace scalar capacity required by [`cg_solve`] for an `n x n` SPD CSR matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn cg_workspace_len(n: usize) -> usize {
    3 * n
}

/// Workspace scalar capacity required by [`gmres_solve`] for an `n x n` CSR matrix and `restart`.
/// `[cite:engine_linalg]`
#[inline]
pub const fn gmres_workspace_len(n: usize, restart: usize) -> usize {
    (restart + 1) * n + (restart + 1) * restart + 4 * restart + 1 + n
}

#[inline]
fn dot_vec<T: Scalar>(a: &[T], b: &[T]) -> T {
    let mut sum = T::ZERO;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += x * y;
    }
    sum
}

#[inline]
fn norm2_vec<T: Scalar>(v: &[T]) -> T {
    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    for &x in v {
        let ax = x.abs();
        if ax > T::ZERO {
            if scale < ax {
                let ratio = scale / ax;
                ssq = T::ONE + ssq * ratio * ratio;
                scale = ax;
            } else {
                let ratio = ax / scale;
                ssq += ratio * ratio;
            }
        }
    }
    scale * ssq.sqrt()
}

/// Solves a symmetric positive-definite sparse system $A x = b$ via preallocated Conjugate Gradient.
/// `[cite:advanced_linear_algebra]`
pub fn cg_solve<T: Scalar>(
    a: &CsrMatrixView<'_, T>,
    b: &[T],
    x: &mut [T],
    tol: T,
    max_iter: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<KrylovReport<T>, LinalgError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    if !tol.is_finite() || tol <= T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    check_all_finite(b)?;
    check_all_finite(x)?;

    let (r, mut r1) = ws.try_split(n)?;
    let (p, mut r2) = r1.try_split(n)?;
    let (ap, _) = r2.try_split(n)?;

    a.spmv(x, ap)?;
    for i in 0..n {
        r[i] = b[i] - ap[i];
        p[i] = r[i];
    }

    let b_norm = norm2_vec(b).max(T::ONE);
    let stop = tol * b_norm;
    let mut rs_old = dot_vec(r, r);
    let mut res_norm = rs_old.sqrt();
    if res_norm <= stop {
        return Ok(KrylovReport {
            iterations: 0,
            residual_norm: res_norm,
        });
    }

    for iter in 1..=max_iter {
        a.spmv(p, ap)?;
        let p_ap = dot_vec(p, ap);
        if p_ap <= T::ZERO || !p_ap.is_finite() {
            return Err(LinalgError::NotPositiveDefinite);
        }
        let alpha = rs_old / p_ap;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * ap[i];
        }
        let rs_new = dot_vec(r, r);
        res_norm = rs_new.sqrt();
        if res_norm <= stop {
            return Ok(KrylovReport {
                iterations: iter,
                residual_norm: res_norm,
            });
        }
        let beta = rs_new / rs_old;
        for i in 0..n {
            p[i] = r[i] + beta * p[i];
        }
        rs_old = rs_new;
    }

    Err(LinalgError::NonConvergence)
}

fn arnoldi_mgs_step<T: Scalar>(
    a: &CsrMatrixView<'_, T>,
    v_basis: &mut [T],
    h_mat: &mut [T],
    w: &mut [T],
    n: usize,
    m: usize,
    j: usize,
) -> Result<(), LinalgError> {
    let vj_start = j * n;
    a.spmv(&v_basis[vj_start..(vj_start + n)], w)?;
    for i in 0..=j {
        let vi_start = i * n;
        let hij = dot_vec(&w[..n], &v_basis[vi_start..(vi_start + n)]);
        h_mat[mat_idx(i, j, m)] = hij;
        for k in 0..n {
            w[k] -= hij * v_basis[vi_start + k];
        }
    }
    let h_next = norm2_vec(&w[..n]);
    h_mat[mat_idx(j + 1, j, m)] = h_next;
    if h_next > T::ZERO {
        let next_start = (j + 1) * n;
        for k in 0..n {
            v_basis[next_start + k] = w[k] / h_next;
        }
    }
    Ok(())
}

fn apply_gmres_rotations<T: Scalar>(
    h_mat: &mut [T],
    cs: &mut [T],
    sn: &mut [T],
    g: &mut [T],
    m: usize,
    j: usize,
) {
    for i in 0..j {
        let h_ij = h_mat[mat_idx(i, j, m)];
        let h_next = h_mat[mat_idx(i + 1, j, m)];
        h_mat[mat_idx(i, j, m)] = cs[i] * h_ij + sn[i] * h_next;
        h_mat[mat_idx(i + 1, j, m)] = -sn[i] * h_ij + cs[i] * h_next;
    }
    let h_jj = h_mat[mat_idx(j, j, m)];
    let h_j1j = h_mat[mat_idx(j + 1, j, m)];
    let denom = h_jj.hypot(h_j1j);
    if denom == T::ZERO {
        cs[j] = T::ONE;
        sn[j] = T::ZERO;
    } else {
        cs[j] = h_jj / denom;
        sn[j] = h_j1j / denom;
    }
    h_mat[mat_idx(j, j, m)] = cs[j] * h_jj + sn[j] * h_j1j;
    h_mat[mat_idx(j + 1, j, m)] = T::ZERO;

    let gj = g[j];
    g[j] = cs[j] * gj;
    g[j + 1] = -sn[j] * gj;
}

fn update_gmres_x<T: Scalar>(
    h_mat: &[T],
    v_basis: &[T],
    g: &[T],
    y: &mut [T],
    x: &mut [T],
    dims: (usize, usize, usize),
) -> Result<(), LinalgError> {
    let (n, m, steps) = dims;
    for i in (0..steps).rev() {
        let mut sum = g[i];
        for (k, &yk) in y.iter().enumerate().take(steps).skip(i + 1) {
            sum -= h_mat[mat_idx(i, k, m)] * yk;
        }
        let diag = h_mat[mat_idx(i, i, m)];
        if diag == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        y[i] = sum / diag;
    }
    for (j, &yj) in y.iter().enumerate().take(steps) {
        let vj_start = j * n;
        for (k, xk) in x.iter_mut().enumerate().take(n) {
            *xk += yj * v_basis[vj_start + k];
        }
    }
    Ok(())
}

/// Solves a general nonsymmetric sparse linear system $A x = b$ via preallocated restarted $\text{GMRES}(m)$.
/// `[cite:advanced_linear_algebra]`
pub fn gmres_solve<T: Scalar>(
    a: &CsrMatrixView<'_, T>,
    b: &[T],
    x: &mut [T],
    restart: usize,
    tol: T,
    max_restarts: usize,
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<KrylovReport<T>, LinalgError> {
    let n = a.nrows();
    if a.ncols() != n || b.len() != n || x.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    if restart == 0 || !tol.is_finite() || tol <= T::ZERO {
        return Err(LinalgError::InvalidInput);
    }
    check_all_finite(b)?;
    check_all_finite(x)?;

    let m = restart;
    let (v_basis, mut r1) = ws.try_split((m + 1) * n)?;
    let (h_mat, mut r2) = r1.try_split((m + 1) * m)?;
    let (cs, mut r3) = r2.try_split(m)?;
    let (sn, mut r4) = r3.try_split(m)?;
    let (g, mut r5) = r4.try_split(m + 1)?;
    let (y, mut r6) = r5.try_split(m)?;
    let (w, _) = r6.try_split(n)?;

    let stop = tol * norm2_vec(b).max(T::ONE);
    let mut total_iters = 0;

    for _cycle in 0..max_restarts {
        a.spmv(x, w)?;
        for k in 0..n {
            w[k] = b[k] - w[k];
        }
        let beta = norm2_vec(&w[..n]);
        if beta <= stop {
            return Ok(KrylovReport {
                iterations: total_iters,
                residual_norm: beta,
            });
        }
        v_basis.fill(T::ZERO);
        h_mat.fill(T::ZERO);
        g.fill(T::ZERO);
        for k in 0..n {
            v_basis[k] = w[k] / beta;
        }
        g[0] = beta;

        let mut inner_steps = 0;
        for j in 0..m {
            arnoldi_mgs_step(a, v_basis, h_mat, w, n, m, j)?;
            apply_gmres_rotations(h_mat, cs, sn, g, m, j);
            total_iters += 1;
            inner_steps = j + 1;
            if g[j + 1].abs() <= stop {
                update_gmres_x(h_mat, v_basis, g, y, x, (n, m, inner_steps))?;
                return Ok(KrylovReport {
                    iterations: total_iters,
                    residual_norm: g[j + 1].abs(),
                });
            }
        }
        update_gmres_x(h_mat, v_basis, g, y, x, (n, m, inner_steps))?;
    }

    Err(LinalgError::NonConvergence)
}
