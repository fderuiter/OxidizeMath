//! Inline zero-allocation stack micro-solvers (`N <= 8`) for `math_commons::fit`.
//!
//! Uses 2D stack arrays (`[[T; 8]; 8]`) with coupled Givens plane rotations for tall streaming QR
//! and small dense linear systems, plus square-root-free symmetric $L D L^\top$ factorization for
//! Marquardt-damped normal equations — avoiding heap allocation and Tier 2 `engine_linalg` coupling.
// [cite:math_commons]

use crate::error::FitError;
use math_scalars::RealField;

/// Hard stack dimension ceiling (`N <= 8`) for inline `math_commons::fit` micro-solvers.
pub const MAX_INLINE_SYSTEM_DIM: usize = 8;

#[inline(always)]
fn coupled_givens<T: RealField>(a: T, b: T) -> (T, T, T) {
    if b.abs() > a.abs() {
        let t = a / b;
        let u = (T::ONE + t * t).sqrt().copysign(b);
        let s = T::ONE / u;
        (s * t, s, b * u)
    } else {
        let t = b / a;
        let u = (T::ONE + t * t).sqrt().copysign(a);
        let c = T::ONE / u;
        (c, c * t, a * u)
    }
}

fn back_substitute_upper<T: RealField, const OUT: usize>(
    r: &[[T; MAX_INLINE_SYSTEM_DIM]; MAX_INLINE_SYSTEM_DIM],
    qty: &[T; MAX_INLINE_SYSTEM_DIM],
    ncols: usize,
    nrows: usize,
    rel_tol: T,
) -> Result<[T; OUT], FitError> {
    if ncols == 0 || ncols > OUT || ncols > MAX_INLINE_SYSTEM_DIM || nrows < ncols {
        return Err(FitError::SingularMatrix);
    }
    let mut max_diag = T::ZERO;
    for (i, row) in r.iter().enumerate().take(ncols) {
        max_diag = max_diag.max(row[i].abs());
    }
    if !max_diag.is_finite() || max_diag == T::ZERO {
        return Err(FitError::SingularMatrix);
    }
    let thresh = rel_tol.max(T::from_usize(nrows) * T::EPSILON) * max_diag;
    let mut sol = [T::ZERO; OUT];
    for i in (0..ncols).rev() {
        let diag = r[i][i];
        if !diag.is_finite() || diag.abs() <= thresh {
            return Err(FitError::SingularMatrix);
        }
        let mut sum = qty[i];
        for (&rij, &sj) in r[i][(i + 1)..ncols].iter().zip(&sol[(i + 1)..ncols]) {
            sum -= rij * sj;
        }
        let xi = sum / diag;
        if !xi.is_finite() {
            return Err(FitError::SingularMatrix);
        }
        sol[i] = xi;
    }
    Ok(sol)
}

/// Streaming orthogonal Givens QR accumulator on 2D stack arrays (`ncols <= 8`).
pub(super) struct StreamingGivensQr<T: RealField> {
    r: [[T; MAX_INLINE_SYSTEM_DIM]; MAX_INLINE_SYSTEM_DIM],
    qty: [T; MAX_INLINE_SYSTEM_DIM],
    ncols: usize,
    nrows: usize,
}

impl<T: RealField> StreamingGivensQr<T> {
    #[inline]
    pub(super) fn new(ncols: usize) -> Result<Self, FitError> {
        if ncols == 0 || ncols > MAX_INLINE_SYSTEM_DIM {
            return Err(FitError::SingularMatrix);
        }
        Ok(Self {
            r: [[T::ZERO; MAX_INLINE_SYSTEM_DIM]; MAX_INLINE_SYSTEM_DIM],
            qty: [T::ZERO; MAX_INLINE_SYSTEM_DIM],
            ncols,
            nrows: 0,
        })
    }

    #[inline]
    pub(super) fn push_row(&mut self, row: &[T], rhs: T) -> Result<(), FitError> {
        if row.len() != self.ncols || !rhs.is_finite() || !row.iter().all(|v| v.is_finite()) {
            return Err(FitError::SingularMatrix);
        }
        let mut work = [T::ZERO; MAX_INLINE_SYSTEM_DIM];
        work[..self.ncols].copy_from_slice(row);
        let mut y = rhs;
        for j in 0..self.ncols {
            let xj = work[j];
            if xj == T::ZERO {
                continue;
            }
            let (c, s, r_new) = coupled_givens(self.r[j][j], xj);
            self.r[j][j] = r_new;
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

    #[inline]
    pub(super) fn solve<const OUT: usize>(&self, rel_tol: T) -> Result<[T; OUT], FitError> {
        back_substitute_upper(&self.r, &self.qty, self.ncols, self.nrows, rel_tol)
    }
}

/// Solves an $n \times n$ ($1 \le n \le 8$) dense linear system $A x = b$ via stack Givens QR rotations.
pub(super) fn solve_givens_dense_small<T: RealField, const CAP: usize>(
    mat: &[[T; CAP]; CAP],
    rhs: &[T; CAP],
    n: usize,
) -> Result<[T; CAP], FitError> {
    if n == 0 || n > CAP || n > MAX_INLINE_SYSTEM_DIM {
        return Err(FitError::SingularMatrix);
    }
    let mut qr = StreamingGivensQr::<T>::new(n)?;
    for (row, &b_i) in mat.iter().zip(rhs.iter()).take(n) {
        qr.push_row(&row[..n], b_i)?;
    }
    qr.solve::<CAP>(T::from_f64(1e-12))
}

/// Solves $(J^\top J + \lambda \max(\operatorname{diag}(J^\top J), \text{floor})) \delta = J^\top r$
/// ($1 \le k \le 8$) on the stack via square-root-free symmetric $L D L^\top$ factorization.
pub(super) fn solve_marquardt_ldlt<T: RealField, const M: usize>(
    jtj: &[[T; M]; M],
    jtr: &[T; M],
    k: usize,
    lambda: T,
    diag_floor: T,
) -> Result<[T; M], FitError> {
    if k == 0 || k > M || k > MAX_INLINE_SYSTEM_DIM || !lambda.is_finite() || lambda < T::ZERO {
        return Err(FitError::SingularMatrix);
    }
    let mut l = [[T::ZERO; MAX_INLINE_SYSTEM_DIM]; MAX_INLINE_SYSTEM_DIM];
    let mut d = [T::ZERO; MAX_INLINE_SYSTEM_DIM];
    let mut max_diag = T::ZERO;

    for j in 0..k {
        let scale_j = jtj[j][j].abs().max(diag_floor);
        let mut dj = jtj[j][j] + lambda * scale_j;
        if !dj.is_finite() || !jtr[j].is_finite() {
            return Err(FitError::SingularMatrix);
        }
        max_diag = max_diag.max(dj.abs());
        for (p, &dp) in d.iter().enumerate().take(j) {
            let ljp = l[j][p];
            dj -= ljp * ljp * dp;
        }
        let piv_tol = T::EPSILON * max_diag.max(T::ONE);
        if !dj.is_finite() || dj <= piv_tol {
            return Err(FitError::SingularMatrix);
        }
        d[j] = dj;
        l[j][j] = T::ONE;
        for i in (j + 1)..k {
            let mut sum = jtj[i][j];
            if !sum.is_finite() {
                return Err(FitError::SingularMatrix);
            }
            for p in 0..j {
                sum -= l[i][p] * l[j][p] * d[p];
            }
            l[i][j] = sum / dj;
        }
    }

    let mut y = [T::ZERO; MAX_INLINE_SYSTEM_DIM];
    for i in 0..k {
        let mut sum = jtr[i];
        for (p, &yp) in y.iter().enumerate().take(i) {
            sum -= l[i][p] * yp;
        }
        y[i] = sum;
    }

    let mut delta = [T::ZERO; M];
    for i in (0..k).rev() {
        let mut sum = y[i] / d[i];
        for (j, &dj_sol) in delta.iter().enumerate().take(k).skip(i + 1) {
            sum -= l[j][i] * dj_sol;
        }
        if !sum.is_finite() {
            return Err(FitError::SingularMatrix);
        }
        delta[i] = sum;
    }
    Ok(delta)
}
