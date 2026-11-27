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
// [cite:engine_linalg]
// [cite:advanced_linear_algebra]

use alloc::vec;

use super::numerical::{LinearAlgebraError, dmatrix_to_row_major};
use crate::{SolverWorkspace, svd_effective_rank, svd_low_rank_approx, svd_ops_workspace_len};
use nalgebra::DMatrix;

const DEFAULT_EFFECTIVE_RANK_TOL: f64 = 1e-12;

/// Computes the effective rank of a matrix using [`crate`] Singular Value Decomposition (SVD).
///
/// # Examples
///
/// ```
/// use engine_linalg::nalgebra_ops::decomposition::effective_rank;
/// use nalgebra::{DMatrix, DVector};
///
/// let diag = DMatrix::from_diagonal(&DVector::from_row_slice(&[1.0, 0.00000000001, 3.0]));
/// assert_eq!(effective_rank(&diag, Some(1e-5)), 2);
/// ```
pub fn effective_rank(matrix: &DMatrix<f64>, tolerance: Option<f64>) -> usize {
    let tol = tolerance.unwrap_or(DEFAULT_EFFECTIVE_RANK_TOL);
    let (m, n) = matrix.shape();
    let a_row = dmatrix_to_row_major(matrix);
    let mut scratch = vec![0.0_f64; svd_ops_workspace_len(m, n)];
    svd_effective_rank(&a_row, m, n, tol, &mut SolverWorkspace::new(&mut scratch)).unwrap_or(0)
}

/// Checked Eckart-Young rank-`k` SVD matrix approximation returning [`LinearAlgebraError`] on `NaN` or non-convergence.
pub fn try_low_rank_approximation(
    matrix: &DMatrix<f64>,
    k: usize,
) -> Result<DMatrix<f64>, LinearAlgebraError> {
    let (m, n) = matrix.shape();
    let a_row = dmatrix_to_row_major(matrix);
    let mut approx = vec![0.0_f64; m * n];
    let mut scratch = vec![0.0_f64; svd_ops_workspace_len(m, n)];
    svd_low_rank_approx(
        &a_row,
        m,
        n,
        k,
        &mut approx,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DMatrix::from_row_slice(m, n, &approx))
}

/// Computes a rank-`k` SVD approximation of `matrix` without panicking; prefer [`try_low_rank_approximation`] for checked error handling.
pub fn low_rank_approximation(matrix: &DMatrix<f64>, k: usize) -> DMatrix<f64> {
    try_low_rank_approximation(matrix, k)
        .unwrap_or_else(|_| DMatrix::zeros(matrix.nrows(), matrix.ncols()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_effective_rank() {
        let m = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(effective_rank(&m, None), 1);

        let m2 = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 0.0, 1.0]);
        assert_eq!(effective_rank(&m2, None), 2);
    }

    #[test]
    fn test_low_rank_approx_and_checked_nan_rejection() {
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        let approx = try_low_rank_approximation(&m, 1).unwrap();
        let diff = &m - &approx;
        assert!((diff.norm() - 1.0).abs() < 1e-5);
        assert_eq!(low_rank_approximation(&m, 1), approx);

        let bad = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 2.0]);
        assert!(try_low_rank_approximation(&bad, 1).is_err());
    }
}
