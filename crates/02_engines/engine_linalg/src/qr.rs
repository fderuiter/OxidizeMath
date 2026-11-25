#![cfg_attr(any(), verified(opt_out = "no_std_engine"))]
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]
//! Householder QR with column pivoting ($A P = Q R$), contrast variance solvers, and streaming Givens QR.

use crate::scalar::Scalar;
use crate::types::{LinalgError, SolverWorkspace, check_all_finite, mat_idx};

/// Workspace scalar capacity required by [`qr_factor_pivoted`] for an `m x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn qr_factor_workspace_len(n: usize) -> usize {
    n
}

/// Workspace scalar capacity required by [`qr_solve_least_squares`] for an `m x n` matrix.
/// `[cite:engine_linalg]`
#[inline]
pub const fn qr_least_squares_workspace_len(m: usize, n: usize) -> usize {
    let k = if m < n { m } else { n };
    m * n + k + m + n
}

fn swap_cols<T: Scalar>(mat: &mut [T], m: usize, n: usize, c1: usize, c2: usize) {
    if c1 == c2 {
        return;
    }
    for r in 0..m {
        mat.swap(mat_idx(r, c1, n), mat_idx(r, c2, n));
    }
}

fn col_norm_from_row<T: Scalar>(mat: &[T], m: usize, n: usize, col: usize, start_row: usize) -> T {
    let mut scale = T::ZERO;
    let mut ssq = T::ONE;
    for r in start_row..m {
        let v = mat[mat_idx(r, col, n)].abs();
        if v > T::ZERO {
            if scale < v {
                let ratio = scale / v;
                ssq = T::ONE + ssq * ratio * ratio;
                scale = v;
            } else {
                let ratio = v / scale;
                ssq += ratio * ratio;
            }
        }
    }
    scale * ssq.sqrt()
}

fn apply_householder_col<T: Scalar>(qr: &mut [T], m: usize, n: usize, k: usize, tau_k: T) {
    if tau_k == T::ZERO {
        return;
    }
    for j in (k + 1)..n {
        let mut dot = qr[mat_idx(k, j, n)];
        for i in (k + 1)..m {
            dot += qr[mat_idx(i, k, n)] * qr[mat_idx(i, j, n)];
        }
        let factor = tau_k * dot;
        qr[mat_idx(k, j, n)] -= factor;
        for i in (k + 1)..m {
            let v_i = qr[mat_idx(i, k, n)];
            qr[mat_idx(i, j, n)] -= factor * v_i;
        }
    }
}

fn qr_pivot_and_reflect<T: Scalar>(
    qr: &mut [T],
    m: usize,
    n: usize,
    k: usize,
    col_perm: &mut [usize],
    col_norms: &mut [T],
) -> T {
    let mut best_c = k;
    let mut best_norm = col_norms[k];
    for (c, &cn) in col_norms.iter().enumerate().take(n).skip(k + 1) {
        if cn > best_norm {
            best_norm = cn;
            best_c = c;
        }
    }
    if best_c != k {
        swap_cols(qr, m, n, k, best_c);
        col_perm.swap(k, best_c);
        col_norms.swap(k, best_c);
    }

    let tail_norm = if k + 1 < m {
        col_norm_from_row(qr, m, n, k, k + 1)
    } else {
        T::ZERO
    };
    let tau_k = if tail_norm == T::ZERO {
        T::ZERO
    } else {
        let norm_x = col_norm_from_row(qr, m, n, k, k);
        let x0 = qr[mat_idx(k, k, n)];
        let alpha = -norm_x.copysign(x0);
        let v0 = x0 - alpha;
        let tk = -v0 / alpha;
        qr[mat_idx(k, k, n)] = alpha;
        for i in (k + 1)..m {
            qr[mat_idx(i, k, n)] /= v0;
        }
        apply_householder_col(qr, m, n, k, tk);
        tk
    };

    for (c, slot) in col_norms.iter_mut().enumerate().take(n).skip(k + 1) {
        *slot = if k + 1 < m {
            col_norm_from_row(qr, m, n, c, k + 1)
        } else {
            T::ZERO
        };
    }
    tau_k
}

/// Factors row-major `qr` (`m x n`) in-place as $A P = Q R$ via Householder QR with column pivoting.
///
/// Stores implicit unit-lower Householder vectors below the diagonal of `qr`, upper-triangular $R$
/// on and above the diagonal, reflector scalars in `tau` (`min(m, n)`), and column permutation
/// indices in `col_perm` (`n`). Returns the numerical rank estimated from $|R_{ii}|$.
/// `[cite:engine_linalg]`
pub fn qr_factor_pivoted<T: Scalar>(
    qr: &mut [T],
    m: usize,
    n: usize,
    tau: &mut [T],
    col_perm: &mut [usize],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<usize, LinalgError> {
    let k_max = m.min(n);
    if m.checked_mul(n) != Some(qr.len()) || tau.len() < k_max || col_perm.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(qr)?;
    let (col_norms, _) = ws.try_split(qr_factor_workspace_len(n))?;
    for j in 0..n {
        col_perm[j] = j;
        col_norms[j] = col_norm_from_row(qr, m, n, j, 0);
    }
    for (k, tau_slot) in tau.iter_mut().enumerate().take(k_max) {
        *tau_slot = qr_pivot_and_reflect(qr, m, n, k, col_perm, col_norms);
    }

    if k_max == 0 {
        return Ok(0);
    }
    let r00 = qr[mat_idx(0, 0, n)].abs();
    if r00 == T::ZERO {
        return Ok(0);
    }
    let rank_tol = T::from_usize(m.max(n)) * T::EPSILON * r00;
    let mut rank = 0;
    while rank < k_max && qr[mat_idx(rank, rank, n)].abs() > rank_tol {
        rank += 1;
    }
    Ok(rank)
}

/// Applies $Q^T$ in-place to `y` (`len == m`) using reflectors from [`qr_factor_pivoted`].
/// `[cite:engine_linalg]`
pub fn qr_apply_qt<T: Scalar>(
    qr: &[T],
    m: usize,
    n: usize,
    tau: &[T],
    y: &mut [T],
) -> Result<(), LinalgError> {
    let k_max = m.min(n);
    if m.checked_mul(n) != Some(qr.len()) || tau.len() < k_max || y.len() != m {
        return Err(LinalgError::DimensionMismatch);
    }
    check_all_finite(y)?;

    for k in 0..k_max {
        let tau_k = tau[k];
        if tau_k == T::ZERO {
            continue;
        }
        let mut dot = y[k];
        for i in (k + 1)..m {
            dot += qr[mat_idx(i, k, n)] * y[i];
        }
        let factor = tau_k * dot;
        y[k] -= factor;
        for i in (k + 1)..m {
            y[i] -= factor * qr[mat_idx(i, k, n)];
        }
    }
    Ok(())
}

/// Solves $R_{0..\text{rank}, 0..\text{rank}} z = (Q^T b)_{0..\text{rank}}$ and unpermutes $x = P z$.
/// `[cite:engine_linalg]`
pub fn qr_solve_r_pivoted<T: Scalar>(
    qr: &[T],
    dims: (usize, usize),
    rank: usize,
    col_perm: &[usize],
    qt_b: &[T],
    x: &mut [T],
    z_work: &mut [T],
) -> Result<(), LinalgError> {
    let (m, n) = dims;
    if m.checked_mul(n) != Some(qr.len())
        || col_perm.len() != n
        || qt_b.len() < rank
        || x.len() != n
        || z_work.len() < n
        || rank > m.min(n)
    {
        return Err(LinalgError::DimensionMismatch);
    }
    z_work[..n].fill(T::ZERO);
    for i in (0..rank).rev() {
        let mut sum = qt_b[i];
        for (j, &zj) in z_work.iter().enumerate().take(rank).skip(i + 1) {
            sum -= qr[mat_idx(i, j, n)] * zj;
        }
        let diag = qr[mat_idx(i, i, n)];
        if diag == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        z_work[i] = sum / diag;
    }
    x.fill(T::ZERO);
    for j in 0..n {
        x[col_perm[j]] = z_work[j];
    }
    Ok(())
}

/// Computes $c^T (A^T A)^{-1} c = \| R^{-T} (P^T c) \|_2^2$ directly from pivoted QR without forming $A^T A$.
/// `[cite:engine_linalg]`
pub fn qr_solve_rt_permuted<T: Scalar>(
    qr: &[T],
    m: usize,
    n: usize,
    rank: usize,
    col_perm: &[usize],
    c: &[T],
    w_work: &mut [T],
) -> Result<T, LinalgError> {
    if m < n || m.checked_mul(n) != Some(qr.len()) || col_perm.len() != n || c.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    if w_work.len() < n {
        return Err(LinalgError::WorkspaceTooSmall {
            required: n,
            available: w_work.len(),
        });
    }
    check_all_finite(c)?;
    if rank < n {
        return Err(LinalgError::SingularMatrix);
    }

    let mut sq_norm = T::ZERO;
    for i in 0..n {
        let mut sum = c[col_perm[i]];
        for (j, &wj) in w_work.iter().enumerate().take(i) {
            sum -= qr[mat_idx(j, i, n)] * wj;
        }
        let diag = qr[mat_idx(i, i, n)];
        if diag == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        let wi = sum / diag;
        w_work[i] = wi;
        sq_norm += wi * wi;
    }
    Ok(sq_norm)
}

/// Solves $\min_x \|A x - b\|_2$ via Householder QR with column pivoting in `ws`, returning numerical rank.
/// `[cite:engine_linalg]`
pub fn qr_solve_least_squares<T: Scalar>(
    a: &[T],
    m: usize,
    n: usize,
    b: &[T],
    x: &mut [T],
    col_perm: &mut [usize],
    ws: &mut SolverWorkspace<'_, T>,
) -> Result<usize, LinalgError> {
    let k_max = m.min(n);
    if m.checked_mul(n) != Some(a.len()) || b.len() != m || x.len() != n || col_perm.len() != n {
        return Err(LinalgError::DimensionMismatch);
    }
    let (qr_buf, mut rest1) = ws.try_split(m * n)?;
    let (tau, mut rest2) = rest1.try_split(k_max)?;
    let (qt_b, mut rest3) = rest2.try_split(m)?;
    let (z_work, mut factor_ws) = rest3.try_split(n)?;

    qr_buf.copy_from_slice(a);
    let rank = qr_factor_pivoted(qr_buf, m, n, tau, col_perm, &mut factor_ws)?;
    qt_b.copy_from_slice(b);
    qr_apply_qt(qr_buf, m, n, tau, qt_b)?;
    qr_solve_r_pivoted(qr_buf, (m, n), rank, col_perm, qt_b, x, z_work)?;
    Ok(rank)
}

#[cfg(feature = "serde")]
#[path = "qr/serde_impl.rs"]
mod serde_impl;

/// Stack-allocated streaming orthogonal Givens QR accumulator for arbitrarily tall least-squares systems.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingQr<T: Scalar, const MAX_P: usize> {
    r: [[T; MAX_P]; MAX_P],
    qty: [T; MAX_P],
    ncols: usize,
    nrows: usize,
}

impl<T: Scalar, const MAX_P: usize> StreamingQr<T, MAX_P> {
    /// Creates an empty `ncols`-predictor upper-triangular QR accumulator (`ncols <= MAX_P`).
    pub fn new(ncols: usize) -> Result<Self, LinalgError> {
        if ncols == 0 || ncols > MAX_P {
            return Err(LinalgError::DimensionMismatch);
        }
        Ok(Self {
            r: [[T::ZERO; MAX_P]; MAX_P],
            qty: [T::ZERO; MAX_P],
            ncols,
            nrows: 0,
        })
    }

    /// Incorporates one observation row $(a_i, b_i)$ via orthogonal Givens plane rotations.
    pub fn push_row(&mut self, row: &[T], rhs: T) -> Result<(), LinalgError> {
        if row.len() != self.ncols {
            return Err(LinalgError::DimensionMismatch);
        }
        check_all_finite(row)?;
        if !rhs.is_finite() {
            return Err(LinalgError::InvalidInput);
        }

        let mut work = [T::ZERO; MAX_P];
        work[..self.ncols].copy_from_slice(row);
        let mut y = rhs;

        for j in 0..self.ncols {
            let xj = work[j];
            if xj == T::ZERO {
                continue;
            }
            let rjj = self.r[j][j];
            let hyp = rjj.hypot(xj);
            let c = rjj / hyp;
            let s = xj / hyp;
            self.r[j][j] = hyp;
            for (rk, wk) in self.r[j][(j + 1)..self.ncols]
                .iter_mut()
                .zip(&mut work[(j + 1)..self.ncols])
            {
                let old_r = *rk;
                let old_w = *wk;
                *rk = c * old_r + s * old_w;
                *wk = -s * old_r + c * old_w;
            }
            let qj = self.qty[j];
            self.qty[j] = c * qj + s * y;
            y = -s * qj + c * y;
        }
        self.nrows += 1;
        Ok(())
    }

    /// Back-substitutes $R x = (Q^T b)_{0..p}$, returning `Err(SingularMatrix)` if underdetermined or rank-deficient.
    pub fn solve(&self, rel_tol: T) -> Result<[T; MAX_P], LinalgError> {
        if self.nrows < self.ncols {
            return Err(LinalgError::SingularMatrix);
        }
        let mut max_diag = T::ZERO;
        for i in 0..self.ncols {
            max_diag = max_diag.max(self.r[i][i].abs());
        }
        if max_diag == T::ZERO {
            return Err(LinalgError::SingularMatrix);
        }
        let thresh = rel_tol.max(T::from_usize(self.nrows) * T::EPSILON) * max_diag;
        let mut sol = [T::ZERO; MAX_P];
        for i in (0..self.ncols).rev() {
            let diag = self.r[i][i];
            if diag.abs() <= thresh {
                return Err(LinalgError::SingularMatrix);
            }
            let mut sum = self.qty[i];
            for (&rij, &sj) in self.r[i][(i + 1)..self.ncols]
                .iter()
                .zip(&sol[(i + 1)..self.ncols])
            {
                sum -= rij * sj;
            }
            sol[i] = sum / diag;
        }
        Ok(sol)
    }
}
