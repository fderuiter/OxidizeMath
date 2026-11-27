//! Pre-factored Householder QR General Linear Model (GLM) estimators and contrast `t_statistic` kernels.

use crate::error::EngineeringError;
use engine_linalg::{
    Scalar, SolverWorkspace, qr_apply_qt, qr_factor_pivoted, qr_factor_workspace_len,
    qr_solve_r_pivoted, qr_solve_rt_permuted,
};
use math_memory::{ArenaError, ArenaLayout, ArenaRegion, ByteCursor, OwnedByteArena};
use nalgebra::{DMatrix, DVector, RealField};

#[inline]
fn alloc_glm<'a, U: bytemuck::Pod>(
    cur: &mut ByteCursor<'a>,
    len: usize,
) -> Result<&'a mut [U], EngineeringError> {
    cur.alloc_slice::<U>(len)
        .map_err(|_| EngineeringError::SingularDesignMatrix)
}

/// Computes the byte capacity required to execute full zero-allocation GLM QR solves for `(nrows, ncols)`.
#[must_use]
#[verified_engine::verified]
pub fn glm_workspace_bytes<T: Scalar>(nrows: usize, ncols: usize) -> usize {
    let mut layout = ArenaLayout::new();
    let _ = layout.reserve::<T>(nrows.saturating_mul(ncols));
    let _ = layout.reserve::<T>(ncols);
    let _ = layout.reserve::<usize>(ncols);
    let _ = layout.reserve::<T>(nrows);
    let _ = layout.reserve::<T>(ncols);
    let _ = layout.reserve::<T>(qr_factor_workspace_len(ncols));
    layout.total_bytes().saturating_add(64)
}

fn copy_dmatrix_row_major<T: RealField + Scalar>(x: &DMatrix<T>, out: &mut [T]) {
    let (m, n) = x.shape();
    for r in 0..m {
        let row = &mut out[r * n..(r + 1) * n];
        for (c, slot) in row.iter_mut().enumerate() {
            *slot = x[(r, c)];
        }
    }
}

/// Pre-factored column-pivoted Householder QR decomposition ($X P = Q R$) of a GLM design matrix $X$,
/// backed by a single [`OwnedByteArena`].
#[derive(Debug, Clone)]
pub struct GlmDesignQr<T: RealField + Scalar> {
    arena: OwnedByteArena,
    qr: ArenaRegion<T>,
    tau: ArenaRegion<T>,
    col_perm: ArenaRegion<usize>,
    nrows: usize,
    ncols: usize,
    rank: usize,
}

impl<T: RealField + Scalar> GlmDesignQr<T> {
    /// Fallibly preallocates internal QR factor storage in a single [`OwnedByteArena`] for up to `max_rows x max_cols`.
    pub fn try_with_capacity(max_rows: usize, max_cols: usize) -> Result<Self, ArenaError> {
        let mut layout = ArenaLayout::new();
        let mn = max_rows
            .checked_mul(max_cols)
            .ok_or(ArenaError::ArithmeticOverflow)?;
        let qr = layout.reserve::<T>(mn)?;
        let tau = layout.reserve::<T>(max_cols)?;
        let col_perm = layout.reserve::<usize>(max_cols)?;
        Ok(Self {
            arena: OwnedByteArena::try_from_layout(&layout)?,
            qr,
            tau,
            col_perm,
            nrows: 0,
            ncols: 0,
            rank: 0,
        })
    }

    /// Preallocates internal QR factor storage in a single [`OwnedByteArena`] for up to `max_rows x max_cols`.
    #[must_use]
    pub fn with_capacity(max_rows: usize, max_cols: usize) -> Self {
        Self::try_with_capacity(max_rows, max_cols).expect("GlmDesignQr::with_capacity failed")
    }

    /// Factors `x` into `self` in-place using `workspace` for Householder QR scratch (`0` heap allocations).
    #[verified_engine::verified]
    pub fn factor_into_with_workspace(
        &mut self,
        x: &DMatrix<T>,
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Result<(), EngineeringError> {
        let (m, n) = x.shape();
        if m == 0 || n == 0 || m < n {
            return Err(EngineeringError::SingularDesignMatrix);
        }
        let mn = m
            .checked_mul(n)
            .ok_or(EngineeringError::SingularDesignMatrix)?;
        if self.qr.len() < mn || self.tau.len() < n || self.col_perm.len() < n {
            return Err(EngineeringError::SingularDesignMatrix);
        }
        let qr_reg = self.qr;
        let tau_reg = self.tau;
        let perm_reg = self.col_perm;
        let mut raw = self.arena.raw_cursor();
        let qr = &mut raw.region_mut(qr_reg)[..mn];
        let tau = &mut raw.region_mut(tau_reg)[..n];
        let col_perm = &mut raw.region_mut(perm_reg)[..n];
        copy_dmatrix_row_major(x, qr);
        let mut frame = workspace.reborrow();
        let rank = qr_factor_pivoted(qr, m, n, tau, col_perm, &mut frame)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        if rank < n {
            return Err(EngineeringError::SingularDesignMatrix);
        }
        self.nrows = m;
        self.ncols = n;
        self.rank = rank;
        Ok(())
    }

    /// Factors design matrix `x` (`subjects x predictors`) using caller-supplied `workspace` for QR scratch.
    #[verified_engine::verified]
    pub fn try_new_with_workspace(
        x: &DMatrix<T>,
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Result<Self, EngineeringError> {
        let (m, n) = x.shape();
        if m == 0 || n == 0 || m < n {
            return Err(EngineeringError::SingularDesignMatrix);
        }
        let mut design = Self::with_capacity(m, n);
        design.factor_into_with_workspace(x, workspace)?;
        Ok(design)
    }

    /// Factors design matrix `x` (`subjects x predictors`) via Householder QR with column pivoting.
    #[verified_engine::verified]
    pub fn try_new(x: &DMatrix<T>) -> Result<Self, EngineeringError> {
        let (m, n) = x.shape();
        let mut arena = OwnedByteArena::with_capacity(glm_workspace_bytes::<T>(m, n));
        Self::try_new_with_workspace(x, &mut arena)
    }

    /// Number of subject rows in the factored design matrix.
    #[verified_engine::verified]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of predictor columns in the factored design matrix.
    #[verified_engine::verified]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Estimates ordinary least-squares parameters $\hat{\beta}$ into `out_beta` using `workspace` (`0` heap allocations).
    #[verified_engine::verified]
    pub fn estimate_beta_with_workspace(
        &self,
        y: &DVector<T>,
        out_beta: &mut DVector<T>,
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Result<(), EngineeringError> {
        if y.len() != self.nrows {
            return Err(EngineeringError::DimensionMismatch {
                context: "Observation vector y length must match design matrix row count",
                expected: self.nrows,
                actual: y.len(),
            });
        }
        if out_beta.len() != self.ncols {
            return Err(EngineeringError::DimensionMismatch {
                context: "Output vector out_beta length must match design matrix predictor count",
                expected: self.ncols,
                actual: out_beta.len(),
            });
        }
        let mn = self.nrows * self.ncols;
        let qr = &self.arena.region_slice(self.qr)[..mn];
        let tau = &self.arena.region_slice(self.tau)[..self.ncols];
        let col_perm = &self.arena.region_slice(self.col_perm)[..self.ncols];
        let mut cur = workspace.cursor();
        let qt_y = cur
            .alloc_slice::<T>(self.nrows)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qt_y.copy_from_slice(y.as_slice());
        let z_work = cur
            .alloc_slice::<T>(self.ncols)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qr_apply_qt(qr, self.nrows, self.ncols, tau, qt_y)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qr_solve_r_pivoted(
            qr,
            (self.nrows, self.ncols),
            self.rank,
            col_perm,
            qt_y,
            out_beta.as_mut_slice(),
            z_work,
        )
        .map_err(|_| EngineeringError::SingularDesignMatrix)
    }

    /// Estimates ordinary least-squares parameters $\hat{\beta}$ for vertex observation vector `y` using the cached QR factor.
    #[verified_engine::verified]
    pub fn estimate_beta(&self, y: &DVector<T>) -> Result<DVector<T>, EngineeringError> {
        let mut out_beta = DVector::from_element(self.ncols, <T as Scalar>::ZERO);
        let mut arena =
            OwnedByteArena::with_capacity(glm_workspace_bytes::<T>(self.nrows, self.ncols));
        self.estimate_beta_with_workspace(y, &mut out_beta, &mut arena)?;
        Ok(out_beta)
    }

    /// Computes unscaled contrast variance $c^T (X^T X)^{-1} c = \|R^{-T} P^T c\|_2^2$ using `workspace` (`0` heap allocations).
    #[verified_engine::verified]
    pub fn contrast_variance_factor_with_workspace(
        &self,
        c: &DVector<T>,
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Result<T, EngineeringError> {
        if c.len() != self.ncols {
            return Err(EngineeringError::DimensionMismatch {
                context: "Contrast vector c length must match design matrix predictor count",
                expected: self.ncols,
                actual: c.len(),
            });
        }
        let mn = self.nrows * self.ncols;
        let qr = &self.arena.region_slice(self.qr)[..mn];
        let col_perm = &self.arena.region_slice(self.col_perm)[..self.ncols];
        let mut frame = workspace.reborrow();
        let w_work = frame
            .alloc_slice::<T>(self.ncols)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qr_solve_rt_permuted(
            qr,
            self.nrows,
            self.ncols,
            self.rank,
            col_perm,
            c.as_slice(),
            w_work,
        )
        .map_err(|_| EngineeringError::SingularDesignMatrix)
    }

    /// Computes unscaled contrast variance $c^T (X^T X)^{-1} c = \|R^{-T} P^T c\|_2^2$ directly from $R$ without forming $X^T X$.
    #[verified_engine::verified]
    pub fn contrast_variance_factor(&self, c: &DVector<T>) -> Result<T, EngineeringError> {
        let mut arena =
            OwnedByteArena::with_capacity(glm_workspace_bytes::<T>(self.nrows, self.ncols));
        self.contrast_variance_factor_with_workspace(c, &mut arena)
    }

    /// Evaluates $t = c^T \hat{\beta} / \sqrt{\sigma^2 c^T (X^T X)^{-1} c}$ using `workspace` (`0` heap allocations).
    #[verified_engine::verified]
    pub fn t_statistic_with_workspace(
        &self,
        c: &DVector<T>,
        beta: &DVector<T>,
        residual_variance: T,
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Result<T, EngineeringError> {
        if c.len() != self.ncols {
            return Err(EngineeringError::DimensionMismatch {
                context: "Contrast vector c length must match design matrix predictor count",
                expected: self.ncols,
                actual: c.len(),
            });
        }
        if beta.len() != self.ncols {
            return Err(EngineeringError::DimensionMismatch {
                context: "Parameter vector beta length must match design matrix predictor count",
                expected: self.ncols,
                actual: beta.len(),
            });
        }
        let variance_term = self.contrast_variance_factor_with_workspace(c, workspace)?;
        eval_t_ratio(c, beta, residual_variance, variance_term)
    }

    /// Evaluates $t = c^T \hat{\beta} / \sqrt{\sigma^2 c^T (X^T X)^{-1} c}$ using the pre-factored QR decomposition.
    #[verified_engine::verified]
    pub fn t_statistic(
        &self,
        c: &DVector<T>,
        beta: &DVector<T>,
        residual_variance: T,
    ) -> Result<T, EngineeringError> {
        let mut arena =
            OwnedByteArena::with_capacity(glm_workspace_bytes::<T>(self.nrows, self.ncols));
        self.t_statistic_with_workspace(c, beta, residual_variance, &mut arena)
    }
}

fn eval_t_ratio<T: RealField + Scalar>(
    c: &DVector<T>,
    beta: &DVector<T>,
    residual_variance: T,
    variance_term: T,
) -> Result<T, EngineeringError> {
    let scaled_variance = residual_variance * variance_term;
    if scaled_variance <= <T as Scalar>::ZERO || !<T as Scalar>::is_finite(scaled_variance) {
        return Err(EngineeringError::ZeroContrastVariance);
    }
    let denominator = <T as Scalar>::sqrt(scaled_variance);
    if denominator == <T as Scalar>::ZERO {
        return Err(EngineeringError::ZeroContrastVariance);
    }
    Ok(c.dot(beta) / denominator)
}

/// Estimates GLM $\hat{\beta}$ into `out_beta` via pivoted Householder QR using `workspace` (`0` heap allocations).
#[verified_engine::verified]
pub fn estimate_beta_with_workspace<T: RealField + Scalar>(
    x: &DMatrix<T>,
    y: &DVector<T>,
    out_beta: &mut DVector<T>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), EngineeringError> {
    let (m, n) = x.shape();
    if m != y.len() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Observation vector y length must match design matrix row count",
            expected: m,
            actual: y.len(),
        });
    }
    if out_beta.len() != n {
        return Err(EngineeringError::DimensionMismatch {
            context: "Output vector out_beta length must match design matrix predictor count",
            expected: n,
            actual: out_beta.len(),
        });
    }
    if m == 0 || n == 0 || m < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    let mn = m
        .checked_mul(n)
        .ok_or(EngineeringError::SingularDesignMatrix)?;
    let mut cur = workspace.cursor();
    let qr = alloc_glm::<T>(&mut cur, mn)?;
    copy_dmatrix_row_major(x, qr);
    let tau = alloc_glm::<T>(&mut cur, n)?;
    let col_perm = alloc_glm::<usize>(&mut cur, n)?;
    let qt_y = alloc_glm::<T>(&mut cur, m)?;
    qt_y.copy_from_slice(y.as_slice());
    let z_work = alloc_glm::<T>(&mut cur, n)?;
    let mut sub = cur.into_arena();
    let rank = qr_factor_pivoted(qr, m, n, tau, col_perm, &mut sub)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    if rank < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    qr_apply_qt(qr, m, n, tau, qt_y).map_err(|_| EngineeringError::SingularDesignMatrix)?;
    qr_solve_r_pivoted(
        qr,
        (m, n),
        rank,
        col_perm,
        qt_y,
        out_beta.as_mut_slice(),
        z_work,
    )
    .map_err(|_| EngineeringError::SingularDesignMatrix)
}

/// Estimates the beta parameters of the GLM using pivoted Householder QR ordinary least squares.
#[verified_engine::verified]
pub fn estimate_beta<T: RealField + Scalar>(
    x: &DMatrix<T>,
    y: &DVector<T>,
) -> Result<DVector<T>, EngineeringError> {
    let mut out_beta = DVector::from_element(x.ncols(), <T as Scalar>::ZERO);
    let mut arena = OwnedByteArena::with_capacity(glm_workspace_bytes::<T>(x.nrows(), x.ncols()));
    estimate_beta_with_workspace(x, y, &mut out_beta, &mut arena)?;
    Ok(out_beta)
}

/// Calculates contrast $t = c^T \hat{\beta} / \sqrt{\sigma^2 \|R^{-T} P^T c\|_2^2}$ via pivoted QR using `workspace` (`0` heap allocations).
#[verified_engine::verified]
pub fn t_statistic_with_workspace<T: RealField + Scalar>(
    c: &DVector<T>,
    beta: &DVector<T>,
    x: &DMatrix<T>,
    residual_variance: T,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<T, EngineeringError> {
    let (m, n) = x.shape();
    if c.len() != n {
        return Err(EngineeringError::DimensionMismatch {
            context: "Contrast vector c length must match design matrix predictor count",
            expected: n,
            actual: c.len(),
        });
    }
    if beta.len() != n {
        return Err(EngineeringError::DimensionMismatch {
            context: "Parameter vector beta length must match design matrix predictor count",
            expected: n,
            actual: beta.len(),
        });
    }
    if m == 0 || n == 0 || m < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    let mn = m
        .checked_mul(n)
        .ok_or(EngineeringError::SingularDesignMatrix)?;
    let mut cur = workspace.cursor();
    let qr = alloc_glm::<T>(&mut cur, mn)?;
    copy_dmatrix_row_major(x, qr);
    let tau = alloc_glm::<T>(&mut cur, n)?;
    let col_perm = alloc_glm::<usize>(&mut cur, n)?;
    let w_work = alloc_glm::<T>(&mut cur, n)?;
    let mut sub = cur.into_arena();
    let rank = qr_factor_pivoted(qr, m, n, tau, col_perm, &mut sub)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    if rank < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    let variance_term = qr_solve_rt_permuted(qr, m, n, rank, col_perm, c.as_slice(), w_work)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    eval_t_ratio(c, beta, residual_variance, variance_term)
}

/// Calculates the contrast t-statistic $t = c^T \hat{\beta} / \sqrt{\sigma^2 \|R^{-T} P^T c\|_2^2}$ via pivoted QR.
#[verified_engine::verified]
pub fn t_statistic<T: RealField + Scalar>(
    c: &DVector<T>,
    beta: &DVector<T>,
    x: &DMatrix<T>,
    residual_variance: T,
) -> Result<T, EngineeringError> {
    let mut arena = OwnedByteArena::with_capacity(glm_workspace_bytes::<T>(x.nrows(), x.ncols()));
    t_statistic_with_workspace(c, beta, x, residual_variance, &mut arena)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_glm_qr_recovers_near_collinear_beta_and_t_stat_at_eps_1e8() {
        let eps = 1e-8_f64;
        let (e2, a, b, c_val) = (2.0 * eps, 1.0 - eps, 1.0 + eps, 1.0 - 2.0 * eps);
        let rows = [
            1.0, 0.0, eps, 1.0, 0.0, -eps, 1.0, 0.0, e2, 1.0, 1.0, a, 1.0, 1.0, b, 1.0, 1.0, c_val,
        ];
        let x = DMatrix::from_row_slice(6, 3, &rows);
        let beta_true = DVector::from_column_slice(&[2.4, -0.5, 0.4]);
        let y = &x * &beta_true;

        let beta_est = estimate_beta(&x, &y).expect("QR GLM beta at eps=1e-8");
        let rel_l2 = (&beta_est - &beta_true).norm() / beta_true.norm();
        assert!(rel_l2 <= 1.7e-8, "GLM rel_l2 {rel_l2:e} > 1.7e-8");
        assert!((beta_est[2] - 0.4).abs() <= 2e-8);

        let c = DVector::from_column_slice(&[0.0, 0.0, 1.0]);
        let sigma2 = 1e-6_f64;
        let qr = GlmDesignQr::try_new(&x).expect("pre-factored QR");
        let t_free = t_statistic(&c, &beta_est, &x, sigma2).expect("finite t_statistic");
        let t_cached = qr
            .t_statistic(&c, &beta_est, sigma2)
            .expect("cached QR t_statistic");
        assert!((t_free - t_cached).abs() < 1e-13);
        assert!(t_free > 0.0 && t_free.is_finite());
    }
}
