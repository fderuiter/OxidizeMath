//! Numerical Linear Algebra Solvers.
//!
//! Provides `DMatrix` / `DVector` checked linear system, pivoted Householder QR least-squares
//! (with SVD fallback), Cholesky, Tikhonov, Krylov, and Moore-Penrose pseudoinverse solvers
//! backed by `#![no_std]` [`engine_linalg`].

use engine_linalg::{
    CsrMatrixView, LinalgError, SolverWorkspace, cg_solve, cg_workspace_len, cholesky_solve,
    cholesky_workspace_len, gmres_solve, gmres_workspace_len, lu_solve, lu_workspace_len,
    qr_or_svd_solve, qr_or_svd_workspace_len, svd_ops_workspace_len, svd_pseudoinverse,
    tikhonov_solve, tikhonov_workspace_len,
};
use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors related to Linear Algebra operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LinearAlgebraError {
    /// The matrix is singular and cannot be inverted or decomposed.
    #[error("Matrix is singular")]
    SingularMatrix,
    /// Dimensions of operands are incompatible.
    #[error("Matrix/Vector dimensions are incompatible")]
    DimensionMismatch,
    /// Solution could not be found.
    #[error("Solution could not be found")]
    SolutionNotFound,
    /// Input contains non-finite (`NaN`/`Infinity`) values or out-of-range parameters.
    #[error("Invalid linear algebra input: {0}")]
    InvalidInput(&'static str),
    /// Matrix is not symmetric positive-definite.
    #[error("Matrix is not symmetric positive-definite")]
    NotPositiveDefinite,
    /// Iterative decomposition or Krylov solver failed to converge within budget.
    #[error("Iterative linear algebra solver failed to converge")]
    NonConvergence,
    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
}

#[verified_engine::verified]
pub(crate) fn dmatrix_to_row_major(a: &DMatrix<f64>) -> Vec<f64> {
    let (m, n) = a.shape();
    let mut out = Vec::with_capacity(m * n);
    for r in 0..m {
        for c in 0..n {
            out.push(a[(r, c)]);
        }
    }
    out
}

/// Solves a square linear system $A x = b$ via partial-pivoted LU in [`engine_linalg`].
#[verified_engine::verified]
pub fn solve_linear_system(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    if n == 0 {
        return Ok(DVector::zeros(0));
    }
    let a_row = dmatrix_to_row_major(a);
    let mut x = vec![0.0_f64; n];
    let mut pivots = vec![0_usize; n];
    let mut scratch = vec![0.0_f64; lu_workspace_len(n)];
    lu_solve(
        &a_row,
        b.as_slice(),
        &mut x,
        n,
        &mut pivots,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DVector::from_vec(x))
}

/// Solves linear least squares $\min_c \|A c - b\|_2$ via Householder QR with column pivoting and SVD fallback.
///
/// Avoids squaring the condition number $\kappa(A)$ in explicit normal equations $A^T A c = A^T b$,
/// recovering near-collinear parameters accurately at $\varepsilon = 10^{-8}$.
#[verified_engine::verified]
pub fn solve_normal_equation(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    if m == 0 || n == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    let a_row = dmatrix_to_row_major(a);
    let mut x = vec![0.0_f64; n];
    let mut perm = vec![0_usize; n];
    let mut scratch = vec![0.0_f64; qr_or_svd_workspace_len(m, n)];
    let rank = qr_or_svd_solve(
        &a_row,
        m,
        n,
        b.as_slice(),
        &mut x,
        &mut perm,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    if rank == 0 {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    Ok(DVector::from_vec(x))
}

/// Checked Moore-Penrose pseudoinverse $A^+ = V \Sigma^+ U^T$ returning [`LinearAlgebraError`] on `NaN` or non-convergence.
#[verified_engine::verified]
pub fn try_moore_penrose_pseudoinverse(
    matrix: &DMatrix<f64>,
    epsilon: f64,
) -> Result<DMatrix<f64>, LinearAlgebraError> {
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(LinearAlgebraError::InvalidInput(
            "singular value tolerance epsilon must be finite and non-negative",
        ));
    }
    let (m, n) = matrix.shape();
    let a_row = dmatrix_to_row_major(matrix);
    let mut pinv = vec![0.0_f64; n * m];
    let mut scratch = vec![0.0_f64; svd_ops_workspace_len(m, n)];
    svd_pseudoinverse(
        &a_row,
        m,
        n,
        epsilon,
        &mut pinv,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DMatrix::from_row_slice(n, m, &pinv))
}

/// Calculates the Moore-Penrose Pseudoinverse ($J^{\dagger}$); prefer [`try_moore_penrose_pseudoinverse`] for checked error handling.
#[verified_engine::verified]
pub fn moore_penrose_pseudoinverse(matrix: &DMatrix<f64>, epsilon: f64) -> DMatrix<f64> {
    try_moore_penrose_pseudoinverse(matrix, epsilon)
        .unwrap_or_else(|_| DMatrix::zeros(matrix.ncols(), matrix.nrows()))
}

/// Solves a symmetric positive-definite linear system $A x = b$ via strict Cholesky factorization.
#[verified_engine::verified]
pub fn solve_cholesky_system(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let a_row = dmatrix_to_row_major(a);
    let mut x = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; cholesky_workspace_len(n)];
    cholesky_solve(
        &a_row,
        b.as_slice(),
        &mut x,
        n,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DVector::from_vec(x))
}

/// Solves Tikhonov ridge regression $\min_x \|A x - b\|_2^2 + \lambda^2 \|x\|_2^2$ via SVD filter factors.
#[verified_engine::verified]
pub fn solve_tikhonov_system(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lambda: f64,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let a_row = dmatrix_to_row_major(a);
    let mut x = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; tikhonov_workspace_len(m, n)];
    tikhonov_solve(
        &a_row,
        m,
        n,
        b.as_slice(),
        lambda,
        &mut x,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DVector::from_vec(x))
}

type CsrOwnedParts = (Vec<usize>, Vec<usize>, Vec<f64>);

fn dense_to_csr_parts(a: &DMatrix<f64>) -> Result<CsrOwnedParts, LinearAlgebraError> {
    let (m, n) = a.shape();
    let mut row_ptr = Vec::with_capacity(m + 1);
    let mut col_ind = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);
    for r in 0..m {
        for c in 0..n {
            let v = a[(r, c)];
            if !v.is_finite() {
                return Err(LinearAlgebraError::InvalidInput(
                    "matrix contains non-finite entry",
                ));
            }
            if v != 0.0 {
                col_ind.push(c);
                values.push(v);
            }
        }
        row_ptr.push(values.len());
    }
    Ok((row_ptr, col_ind, values))
}

/// Solves an SPD linear system $A x = b$ using [`engine_linalg`]'s preallocated Conjugate Gradient solver.
#[verified_engine::verified]
pub fn solve_conjugate_gradient(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    tol: f64,
    max_iter: usize,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let (row_ptr, col_ind, values) = dense_to_csr_parts(a)?;
    let csr = CsrMatrixView::new(n, n, &row_ptr, &col_ind, &values)?;
    let mut x = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; cg_workspace_len(n)];
    cg_solve(
        &csr,
        b.as_slice(),
        &mut x,
        tol,
        max_iter,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DVector::from_vec(x))
}

/// Solves a general square linear system $A x = b$ using [`engine_linalg`]'s restarted GMRES solver.
#[verified_engine::verified]
pub fn solve_gmres_restarted(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    restart: usize,
    tol: f64,
    max_restarts: usize,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (m, n) = a.shape();
    if m != n || n != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let (row_ptr, col_ind, values) = dense_to_csr_parts(a)?;
    let csr = CsrMatrixView::new(n, n, &row_ptr, &col_ind, &values)?;
    let mut x = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; gmres_workspace_len(n, restart)];
    gmres_solve(
        &csr,
        b.as_slice(),
        &mut x,
        restart,
        tol,
        max_restarts,
        &mut SolverWorkspace::new(&mut scratch),
    )?;
    Ok(DVector::from_vec(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_normal_equation_and_lauchli_conditioning() {
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_column_slice(&[1.0, 2.0, 3.0]);
        let c_vec = solve_normal_equation(&a, &b).unwrap();
        assert!(c_vec[0].abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((c_vec[1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);

        let eps = 1e-8_f64;
        let lauchli = DMatrix::from_row_slice(
            4,
            3,
            &[1.0, 1.0, 1.0, eps, 0.0, 0.0, 0.0, eps, 0.0, 0.0, 0.0, eps],
        );
        let beta_true = DVector::from_column_slice(&[1.2, -0.7, 0.4]);
        let y = &lauchli * &beta_true;
        let est = solve_normal_equation(&lauchli, &y).unwrap();
        let rel_l2 = (&est - &beta_true).norm() / beta_true.norm();
        assert!(rel_l2 <= 1.7e-8);
        assert!((est[2] - 0.4).abs() <= 2e-8);
    }

    #[test]
    #[verified_engine::verified]
    fn test_checked_pseudoinverse_and_krylov_wrappers() {
        let bad = DMatrix::from_row_slice(2, 2, &[1.0, f64::NAN, 0.0, 1.0]);
        assert!(try_moore_penrose_pseudoinverse(&bad, 1e-10).is_err());
        assert!(try_moore_penrose_pseudoinverse(&DMatrix::identity(2, 2), -1.0).is_err());

        let spd = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let rhs = DVector::from_column_slice(&[5.0, 4.0]);
        let x_chol = solve_cholesky_system(&spd, &rhs).unwrap();
        let x_cg = solve_conjugate_gradient(&spd, &rhs, 1e-12, 20).unwrap();
        let x_gm = solve_gmres_restarted(&spd, &rhs, 2, 1e-12, 10).unwrap();
        assert!((x_chol[0] - 1.0).abs() < 1e-11);
        assert!((x_cg[1] - 1.0).abs() < 1e-11);
        assert!((x_gm[0] - 1.0).abs() < 1e-11);
    }

    #[test]
    #[verified_engine::verified]
    fn test_linalg_error_source_retained() {
        use std::error::Error as StdError;

        let singular = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 4.0]);
        let rhs = DVector::from_column_slice(&[1.0, 2.0]);
        let err = solve_linear_system(&singular, &rhs).unwrap_err();
        assert_eq!(err, LinearAlgebraError::Linalg(LinalgError::SingularMatrix));
        assert!(err.source().is_some());

        let non_spd = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        let chol_err = solve_cholesky_system(&non_spd, &rhs).unwrap_err();
        assert_eq!(
            chol_err,
            LinearAlgebraError::Linalg(LinalgError::NotPositiveDefinite)
        );
        assert!(chol_err.source().is_some());
    }
}
