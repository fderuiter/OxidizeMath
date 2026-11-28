//! # Matrix Decomposition
//!
//! Factorizing matrices into products of simpler matrices is fundamental for computation and theoretical analysis.
//!
//! ## LU Decomposition (Gaussian Elimination)
//! *   Any square matrix $A$ can be factored into a lower triangular matrix $L$ and an upper triangular matrix $U$:
//!     $$A = LU$$
//! *   **L (Lower):** Contains the multipliers used during Gaussian elimination, with $1$s on the diagonal.
//! *   **U (Upper):** The echelon form resulting from elimination; its diagonal entries are the **pivots**.
//! *   **Application:** This allows solving systems $Ax = b$ efficiently by splitting the problem into two triangular systems: $Lc = b$ (forward substitution) and $Ux = c$ (back substitution).
//!
//! ## Singular Value Decomposition (SVD)
//! *   Any $m \times n$ matrix $A$ can be factored into:
//!     $$A = U \Sigma V^T$$
//! *   **Rank:** The number of non-zero singular values equals the **rank** of the matrix.
//! *   **Application:** SVD is used for least squares estimation, low-rank Eckart-Young approximation, and determining the effective rank of a matrix.

use super::numerical::{LinearAlgebraError, workspace::copy_dmatrix_to_row_major_slice};
use engine_linalg::{
    SolverWorkspace, svd_effective_rank, svd_low_rank_approx, svd_ops_workspace_len,
};
use math_memory::{ArenaLayout, OwnedByteArena};
use nalgebra::DMatrix;

/// Returns conservative byte capacity required by [`effective_rank_with_workspace`] and [`try_low_rank_approximation_with_workspace`].
#[must_use]
#[verified_engine::verified]
pub fn svd_decomposition_workspace_bytes(nrows: usize, ncols: usize) -> usize {
    let mn = nrows.saturating_mul(ncols);
    let mut layout = ArenaLayout::new();
    let _ = layout.reserve::<f64>(
        mn.saturating_mul(2)
            .saturating_add(svd_ops_workspace_len(nrows, ncols)),
    );
    layout.total_bytes().saturating_add(64)
}

/// Computes the effective rank of `matrix` using caller-supplied `workspace` (`0` heap allocations).
#[verified_engine::verified]
pub fn effective_rank_with_workspace(
    matrix: &DMatrix<f64>,
    tolerance: Option<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<usize, LinearAlgebraError> {
    let tol = tolerance.unwrap_or(math_commons::registry::TOLERANCE_HIGH);
    let (m, n) = matrix.shape();
    let mut cur = workspace.cursor();
    let a_row = cur
        .alloc_slice::<f64>(m * n)
        .map_err(engine_linalg::LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(matrix, a_row);
    let mut sub = cur.into_arena();
    Ok(svd_effective_rank(a_row, m, n, tol, &mut sub)?)
}

/// Computes the effective rank of a matrix using [`engine_linalg`] Singular Value Decomposition (SVD).
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::linear_algebra::decomposition::effective_rank;
/// use math_explorer::nalgebra::DMatrix;
///
/// let diag = DMatrix::from_diagonal(&math_explorer::nalgebra::DVector::from_vec(vec![1.0, 0.00000000001, 3.0]));
/// assert_eq!(effective_rank(&diag, Some(1e-5)), 2);
/// ```
#[verified_engine::verified]
pub fn effective_rank(matrix: &DMatrix<f64>, tolerance: Option<f64>) -> usize {
    let (m, n) = matrix.shape();
    let mut ws = OwnedByteArena::with_capacity(svd_decomposition_workspace_bytes(m, n));
    effective_rank_with_workspace(matrix, tolerance, &mut ws).unwrap_or(0)
}

/// Computes rank-`k` Eckart-Young SVD approximation into `out_approx` (`nrows x ncols`) using `workspace` (`0` heap allocations).
#[verified_engine::verified]
pub fn try_low_rank_approximation_with_workspace(
    matrix: &DMatrix<f64>,
    k: usize,
    out_approx: &mut DMatrix<f64>,
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), LinearAlgebraError> {
    let (m, n) = matrix.shape();
    if out_approx.shape() != (m, n) {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let mut cur = workspace.cursor();
    let a_row = cur
        .alloc_slice::<f64>(m * n)
        .map_err(engine_linalg::LinalgError::from)?;
    copy_dmatrix_to_row_major_slice(matrix, a_row);
    let approx_row = cur
        .alloc_slice::<f64>(m * n)
        .map_err(engine_linalg::LinalgError::from)?;
    let mut sub = cur.into_arena();
    svd_low_rank_approx(a_row, m, n, k, approx_row, &mut sub)?;
    for r in 0..m {
        for c in 0..n {
            out_approx[(r, c)] =
                approx_row[math_commons::math_kernel::types::flatten_2d_index(c, r, n)];
        }
    }
    Ok(())
}

/// Checked Eckart-Young rank-`k` SVD matrix approximation returning [`LinearAlgebraError`] on `NaN` or non-convergence.
#[verified_engine::verified]
pub fn try_low_rank_approximation(
    matrix: &DMatrix<f64>,
    k: usize,
) -> Result<DMatrix<f64>, LinearAlgebraError> {
    let (m, n) = matrix.shape();
    let mut out_approx = DMatrix::zeros(m, n);
    let mut ws = OwnedByteArena::with_capacity(svd_decomposition_workspace_bytes(m, n));
    try_low_rank_approximation_with_workspace(matrix, k, &mut out_approx, &mut ws)?;
    Ok(out_approx)
}

/// Computes a rank-`k` SVD approximation of `matrix` without panicking; prefer [`try_low_rank_approximation`] for checked error handling.
#[verified_engine::verified]
pub fn low_rank_approximation(matrix: &DMatrix<f64>, k: usize) -> DMatrix<f64> {
    try_low_rank_approximation(matrix, k)
        .unwrap_or_else(|_| DMatrix::zeros(matrix.nrows(), matrix.ncols()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    #[verified_engine::verified]
    fn test_effective_rank() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(effective_rank(&m, None), 1);

        let m2 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(effective_rank(&m2, None), 2);
    }

    #[test]
    #[verified_engine::verified]
    fn test_low_rank_approx_and_checked_nan_rejection() {
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let approx = try_low_rank_approximation(&m, 1).unwrap();
        let diff = &m - &approx;
        assert!((diff.norm() - 1.0).abs() < 1e-5);
        assert_eq!(low_rank_approximation(&m, 1), approx);

        let bad = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 2.0]);
        assert!(try_low_rank_approximation(&bad, 1).is_err());

        let mut out_approx = DMatrix::zeros(2, 2);
        let mut ws = OwnedByteArena::with_capacity(svd_decomposition_workspace_bytes(2, 2));
        let (rank, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let r = effective_rank_with_workspace(&m, Some(1e-8), &mut ws).unwrap();
            try_low_rank_approximation_with_workspace(&m, 1, &mut out_approx, &mut ws).unwrap();
            r
        });
        assert_eq!(rank, 2);
        assert_eq!(out_approx, approx);
    }
}
