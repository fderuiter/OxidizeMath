//! Pre-factored Householder QR General Linear Model (GLM) estimators and contrast `t_statistic` kernels.

use crate::error::EngineeringError;
use engine_linalg::{
    ByteArena, Scalar, SolverWorkspace, qr_apply_qt, qr_factor_pivoted, qr_factor_workspace_len,
    qr_solve_r_pivoted, qr_solve_rt_permuted,
};
use math_memory::padded_slice_bytes;
use nalgebra::{DMatrix, DVector, RealField};

/// Conservative byte budget for [`estimate_beta_with_workspace`] and [`t_statistic_with_workspace`]
/// on an `nrows x ncols` design matrix.
#[must_use]
pub fn glm_workspace_bytes<T: Scalar>(nrows: usize, ncols: usize) -> usize {
    let mn = nrows.saturating_mul(ncols);
    let qr_work = qr_factor_workspace_len(ncols);
    padded_slice_bytes::<T>(mn)
        .unwrap_or(usize::MAX)
        .saturating_add(padded_slice_bytes::<T>(ncols).unwrap_or(usize::MAX))
        .saturating_add(padded_slice_bytes::<usize>(ncols).unwrap_or(usize::MAX))
        .saturating_add(padded_slice_bytes::<T>(qr_work).unwrap_or(usize::MAX))
        .saturating_add(padded_slice_bytes::<T>(nrows).unwrap_or(usize::MAX))
        .saturating_add(padded_slice_bytes::<T>(ncols).unwrap_or(usize::MAX))
}

/// Pre-factored column-pivoted Householder QR decomposition ($X P = Q R$) of a GLM design matrix $X$.
///
/// Factoring $X$ once allows solving ordinary least-squares $\hat{\beta}(y)$ and evaluating contrast
/// `t_statistic` values across thousands of cortical vertices in $O(mp + p^2)$ per vertex without
/// ever forming or inverting $X^T X$.
#[derive(Debug, Clone)]
pub struct GlmDesignQr<T: RealField + Scalar> {
    qr: Vec<T>,
    tau: Vec<T>,
    col_perm: Vec<usize>,
    nrows: usize,
    ncols: usize,
    rank: usize,
}

impl<T: RealField + Scalar> GlmDesignQr<T> {
    /// Factors design matrix `x` (`subjects x predictors`) via Householder QR with column pivoting.
    ///
    /// Returns [`EngineeringError::SingularDesignMatrix`] if `x` is empty, underdetermined (`nrows < ncols`),
    /// non-finite, or numerically rank-deficient (`rank < ncols`).
    #[verified_engine::verified]
    pub fn try_new(x: &DMatrix<T>) -> Result<Self, EngineeringError> {
        let (m, n) = x.shape();
        if m == 0 || n == 0 || m < n {
            return Err(EngineeringError::SingularDesignMatrix);
        }
        let mut qr = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                qr.push(x[(r, c)]);
            }
        }
        let mut tau = vec![<T as Scalar>::ZERO; n];
        let mut col_perm = vec![0_usize; n];
        let mut scratch = vec![<T as Scalar>::ZERO; qr_factor_workspace_len(n)];
        let rank = qr_factor_pivoted(
            &mut qr,
            m,
            n,
            &mut tau,
            &mut col_perm,
            &mut ByteArena::from_slice(&mut scratch),
        )
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        if rank < n {
            return Err(EngineeringError::SingularDesignMatrix);
        }
        Ok(Self {
            qr,
            tau,
            col_perm,
            nrows: m,
            ncols: n,
            rank,
        })
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

    /// Zero-allocation OLS estimator writing $\hat{\beta}$ into `out_beta` using `ws`.
    #[verified_engine::verified]
    pub fn estimate_beta_with_workspace<W: SolverWorkspace>(
        &self,
        y: &[T],
        out_beta: &mut [T],
        ws: &mut W,
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
                context: "Output beta slice length must match design matrix predictor count",
                expected: self.ncols,
                actual: out_beta.len(),
            });
        }
        let mut scope = ws.reborrow();
        let qt_y = scope
            .alloc_slice_copy(y)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        let z_work = scope
            .alloc_slice::<T>(self.ncols)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qr_apply_qt(&self.qr, self.nrows, self.ncols, &self.tau, qt_y)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qr_solve_r_pivoted(
            &self.qr,
            (self.nrows, self.ncols),
            self.rank,
            &self.col_perm,
            qt_y,
            out_beta,
            z_work,
        )
        .map_err(|_| EngineeringError::SingularDesignMatrix)
    }

    /// Estimates ordinary least-squares parameters $\hat{\beta}$ for vertex observation vector `y` using the cached QR factor.
    #[verified_engine::verified]
    pub fn estimate_beta(&self, y: &DVector<T>) -> Result<DVector<T>, EngineeringError> {
        let mut beta = vec![<T as Scalar>::ZERO; self.ncols];
        let mut scratch = vec![<T as Scalar>::ZERO; self.nrows + self.ncols + 2];
        self.estimate_beta_with_workspace(
            y.as_slice(),
            &mut beta,
            &mut ByteArena::from_slice(&mut scratch),
        )?;
        Ok(DVector::from_vec(beta))
    }

    /// Zero-allocation unscaled contrast variance $c^T (X^T X)^{-1} c = \|R^{-T} P^T c\|_2^2$ using `ws`.
    #[verified_engine::verified]
    pub fn contrast_variance_factor_with_workspace<W: SolverWorkspace>(
        &self,
        c: &[T],
        ws: &mut W,
    ) -> Result<T, EngineeringError> {
        if c.len() != self.ncols {
            return Err(EngineeringError::DimensionMismatch {
                context: "Contrast vector c length must match design matrix predictor count",
                expected: self.ncols,
                actual: c.len(),
            });
        }
        let mut scope = ws.reborrow();
        let w_work = scope
            .alloc_slice::<T>(self.ncols)
            .map_err(|_| EngineeringError::SingularDesignMatrix)?;
        qr_solve_rt_permuted(
            &self.qr,
            self.nrows,
            self.ncols,
            self.rank,
            &self.col_perm,
            c,
            w_work,
        )
        .map_err(|_| EngineeringError::SingularDesignMatrix)
    }

    /// Computes unscaled contrast variance $c^T (X^T X)^{-1} c = \|R^{-T} P^T c\|_2^2$ directly from $R$ without forming $X^T X$.
    #[verified_engine::verified]
    pub fn contrast_variance_factor(&self, c: &DVector<T>) -> Result<T, EngineeringError> {
        let mut scratch = vec![<T as Scalar>::ZERO; self.ncols + 2];
        self.contrast_variance_factor_with_workspace(
            c.as_slice(),
            &mut ByteArena::from_slice(&mut scratch),
        )
    }

    /// Zero-allocation contrast $t$-statistic using `ws`.
    #[verified_engine::verified]
    pub fn t_statistic_with_workspace<W: SolverWorkspace>(
        &self,
        c: &[T],
        beta: &[T],
        residual_variance: T,
        ws: &mut W,
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
        let variance_term = self.contrast_variance_factor_with_workspace(c, ws)?;
        finish_t_statistic(c, beta, residual_variance, variance_term)
    }

    /// Evaluates $t = c^T \hat{\beta} / \sqrt{\sigma^2 c^T (X^T X)^{-1} c}$ using the pre-factored QR decomposition.
    #[verified_engine::verified]
    pub fn t_statistic(
        &self,
        c: &DVector<T>,
        beta: &DVector<T>,
        residual_variance: T,
    ) -> Result<T, EngineeringError> {
        let mut scratch = vec![<T as Scalar>::ZERO; self.ncols + 2];
        self.t_statistic_with_workspace(
            c.as_slice(),
            beta.as_slice(),
            residual_variance,
            &mut ByteArena::from_slice(&mut scratch),
        )
    }
}

fn finish_t_statistic<T: RealField + Scalar>(
    c: &[T],
    beta: &[T],
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
    let mut dot = <T as Scalar>::ZERO;
    for (&ci, &bi) in c.iter().zip(beta.iter()) {
        dot += ci * bi;
    }
    Ok(dot / denominator)
}

/// Zero-allocation GLM OLS parameter estimation writing $\hat{\beta}$ into `out_beta` using `ws`.
#[verified_engine::verified]
pub fn estimate_beta_with_workspace<T: RealField + Scalar, W: SolverWorkspace>(
    x: &DMatrix<T>,
    y: &DVector<T>,
    out_beta: &mut [T],
    ws: &mut W,
) -> Result<(), EngineeringError> {
    let (m, n) = x.shape();
    if y.len() != m {
        return Err(EngineeringError::DimensionMismatch {
            context: "Observation vector y length must match design matrix row count",
            expected: m,
            actual: y.len(),
        });
    }
    if out_beta.len() != n {
        return Err(EngineeringError::DimensionMismatch {
            context: "Output beta slice length must match design matrix predictor count",
            expected: n,
            actual: out_beta.len(),
        });
    }
    if m == 0 || n == 0 || m < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    let mut scope = ws.reborrow();
    let qr = scope
        .retain_slice::<T>(m * n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    for r in 0..m {
        for c in 0..n {
            qr[math_commons::math_kernel::types::flatten_2d_index(c, r, n)] = x[(r, c)];
        }
    }
    debug_assert_eq!(out_beta.len(), n);
    let tau = scope
        .alloc_slice::<T>(n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let col_perm = scope
        .alloc_slice::<usize>(n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let qt_y = scope
        .alloc_slice_copy(y.as_slice())
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let z_work = scope
        .alloc_slice::<T>(n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let mut qr_ws = scope.reborrow().into_byte_arena();
    let rank = qr_factor_pivoted(qr, m, n, tau, col_perm, &mut qr_ws)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    if rank < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    qr_apply_qt(qr, m, n, tau, qt_y).map_err(|_| EngineeringError::SingularDesignMatrix)?;
    qr_solve_r_pivoted(qr, (m, n), rank, col_perm, qt_y, out_beta, z_work)
        .map_err(|_| EngineeringError::SingularDesignMatrix)
}

/// Estimates the beta parameters of the GLM using pivoted Householder QR ordinary least squares.
///
/// For multi-vertex cohorts sharing one design matrix `x`, pre-factor `x` once via [`GlmDesignQr::try_new`].
#[verified_engine::verified]
pub fn estimate_beta<T: RealField + Scalar>(
    x: &DMatrix<T>,
    y: &DVector<T>,
) -> Result<DVector<T>, EngineeringError> {
    if x.nrows() != y.len() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Observation vector y length must match design matrix row count",
            expected: x.nrows(),
            actual: y.len(),
        });
    }
    GlmDesignQr::try_new(x)?.estimate_beta(y)
}

/// Zero-allocation GLM contrast $t$-statistic $t = c^T \hat{\beta} / \sqrt{\sigma^2 \|R^{-T} P^T c\|_2^2}$ using `ws`.
#[verified_engine::verified]
pub fn t_statistic_with_workspace<T: RealField + Scalar, W: SolverWorkspace>(
    c: &[T],
    beta: &[T],
    x: &DMatrix<T>,
    residual_variance: T,
    ws: &mut W,
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
    let mut scope = ws.reborrow();
    let qr = scope
        .retain_slice::<T>(m * n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    for r in 0..m {
        for col in 0..n {
            qr[math_commons::math_kernel::types::flatten_2d_index(col, r, n)] = x[(r, col)];
        }
    }
    debug_assert_eq!(c.len(), n);
    let tau = scope
        .alloc_slice::<T>(n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let col_perm = scope
        .alloc_slice::<usize>(n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let w_work = scope
        .alloc_slice::<T>(n)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    let mut qr_ws = scope.reborrow().into_byte_arena();
    let rank = qr_factor_pivoted(qr, m, n, tau, col_perm, &mut qr_ws)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    if rank < n {
        return Err(EngineeringError::SingularDesignMatrix);
    }
    let variance_term = qr_solve_rt_permuted(qr, m, n, rank, col_perm, c, w_work)
        .map_err(|_| EngineeringError::SingularDesignMatrix)?;
    finish_t_statistic(c, beta, residual_variance, variance_term)
}

/// Calculates the contrast t-statistic $t = c^T \hat{\beta} / \sqrt{\sigma^2 \|R^{-T} P^T c\|_2^2}$ via pivoted QR.
#[verified_engine::verified]
pub fn t_statistic<T: RealField + Scalar>(
    c: &DVector<T>,
    beta: &DVector<T>,
    x: &DMatrix<T>,
    residual_variance: T,
) -> Result<T, EngineeringError> {
    if c.len() != x.ncols() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Contrast vector c length must match design matrix predictor count",
            expected: x.ncols(),
            actual: c.len(),
        });
    }
    if beta.len() != x.ncols() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Parameter vector beta length must match design matrix predictor count",
            expected: x.ncols(),
            actual: beta.len(),
        });
    }
    GlmDesignQr::try_new(x)?.t_statistic(c, beta, residual_variance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_glm_qr_recovers_near_collinear_beta_and_t_stat_at_eps_1e8() {
        let eps = 1e-8_f64;
        let x = DMatrix::from_row_slice(
            6,
            3,
            &[
                1.0,
                0.0,
                eps,
                1.0,
                0.0,
                -eps,
                1.0,
                0.0,
                2.0 * eps,
                1.0,
                1.0,
                1.0 - eps,
                1.0,
                1.0,
                1.0 + eps,
                1.0,
                1.0,
                1.0 - 2.0 * eps,
            ],
        );
        let beta_true = DVector::from_column_slice(&[2.4, -0.5, 0.4]);
        let y = &x * &beta_true;

        let beta_est = estimate_beta(&x, &y).expect("QR GLM beta at eps=1e-8");
        let rel_l2 = (&beta_est - &beta_true).norm() / beta_true.norm();
        assert!(rel_l2 <= 1.7e-8, "GLM rel_l2 {rel_l2:e} > 1.7e-8");
        assert!(
            (beta_est[2] - 0.4).abs() <= 2e-8,
            "GLM beta_2={} drifted from 0.4",
            beta_est[2]
        );

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
