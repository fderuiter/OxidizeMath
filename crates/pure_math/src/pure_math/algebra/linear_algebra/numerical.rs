//! Numerical Linear Algebra Solvers.
//!
//! Provides `DMatrix` / `DVector` checked linear system, pivoted Householder QR least-squares
//! (with SVD fallback), Cholesky, Tikhonov / augmented-QR Ridge, Lawson-Hanson active-set NNLS,
//! Krylov, and Moore-Penrose pseudoinverse solvers backed by `#![no_std]` [`engine_linalg`].

use engine_linalg::{
    CsrMatrixView, LinalgError, SolverWorkspace, cg_solve, cg_workspace_len, cholesky_solve,
    cholesky_workspace_len, gmres_solve, gmres_workspace_len, lu_solve, lu_workspace_len,
    qr_or_svd_solve, qr_or_svd_workspace_len, svd_ops_workspace_len, svd_pseudoinverse,
    tikhonov_solve, tikhonov_workspace_len,
};
use nalgebra::{DMatrix, DVector, SVD};
use thiserror::Error;

#[cfg(test)]
#[path = "numerical_tests.rs"]
mod tests;

/// Errors related to Linear Algebra operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum LinearAlgebraError {
    /// The matrix is singular and cannot be inverted or decomposed.
    #[error("Matrix is singular")]
    SingularMatrix,
    /// Dimensions of operands are incompatible or empty.
    #[error("Matrix/Vector dimensions are incompatible")]
    DimensionMismatch,
    /// Solution could not be found.
    #[error("Solution could not be found")]
    SolutionNotFound,
    /// Input contains non-finite (`NaN`/`Infinity`) values or out-of-range parameters.
    #[error("Invalid linear algebra input: {0}")]
    InvalidInput(&'static str),
    /// Input contains `NaN` or infinite floating-point values.
    #[error("Input contains non-finite (NaN or infinite) values")]
    NonFiniteInput,
    /// Matrix or regularization parameter is not symmetric positive-definite.
    #[error("Matrix is not symmetric positive-definite")]
    NotPositiveDefinite,
    /// Iterative decomposition or Krylov solver failed to converge within budget.
    #[error("Iterative linear algebra solver failed to converge")]
    NonConvergence,
    /// Iterative decomposition or constrained solver failed to converge within its iteration cap.
    #[error("Iterative decomposition or solver failed to converge")]
    ConvergenceFailed,
}

#[verified_engine::verified]
pub(crate) fn map_linalg_error(err: LinalgError) -> LinearAlgebraError {
    match err {
        LinalgError::DimensionMismatch => LinearAlgebraError::DimensionMismatch,
        LinalgError::SingularMatrix => LinearAlgebraError::SingularMatrix,
        LinalgError::InvalidInput => {
            LinearAlgebraError::InvalidInput("non-finite matrix entry or invalid parameter")
        }
        LinalgError::NotPositiveDefinite => LinearAlgebraError::NotPositiveDefinite,
        LinalgError::NonConvergence => LinearAlgebraError::NonConvergence,
        LinalgError::WorkspaceTooSmall { .. } => LinearAlgebraError::SolutionNotFound,
    }
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

#[verified_engine::verified]
pub(crate) fn ensure_finite_matrix(matrix: &DMatrix<f64>) -> Result<(), LinearAlgebraError> {
    if !matrix.iter().all(|v| v.is_finite()) {
        return Err(LinearAlgebraError::NonFiniteInput);
    }
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    Ok(())
}

#[verified_engine::verified]
pub(crate) fn validate_system_inputs(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<(), LinearAlgebraError> {
    if !a.iter().all(|v| v.is_finite()) || !b.iter().all(|v| v.is_finite()) {
        return Err(LinearAlgebraError::NonFiniteInput);
    }
    if a.nrows() == 0 || a.ncols() == 0 || b.is_empty() || a.nrows() != b.len() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    Ok(())
}

#[verified_engine::verified]
pub(crate) fn bounded_svd(
    matrix: &DMatrix<f64>,
    compute_u: bool,
    compute_v: bool,
) -> Result<SVD<f64, nalgebra::Dyn, nalgebra::Dyn>, LinearAlgebraError> {
    ensure_finite_matrix(matrix)?;
    let max_iter = 100 * matrix.nrows().max(matrix.ncols()).max(1);
    SVD::try_new(matrix.clone(), compute_u, compute_v, f64::EPSILON, max_iter)
        .ok_or(LinearAlgebraError::ConvergenceFailed)
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
    )
    .map_err(|err| match err {
        LinalgError::SingularMatrix => LinearAlgebraError::SolutionNotFound,
        other => map_linalg_error(other),
    })?;
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
    )
    .map_err(map_linalg_error)?;
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
    )
    .map_err(map_linalg_error)?;
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
    )
    .map_err(map_linalg_error)?;
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
    )
    .map_err(map_linalg_error)?;
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
    let csr = CsrMatrixView::new(n, n, &row_ptr, &col_ind, &values).map_err(map_linalg_error)?;
    let mut x = vec![0.0_f64; n];
    let mut scratch = vec![0.0_f64; cg_workspace_len(n)];
    cg_solve(
        &csr,
        b.as_slice(),
        &mut x,
        tol,
        max_iter,
        &mut SolverWorkspace::new(&mut scratch),
    )
    .map_err(map_linalg_error)?;
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
    let csr = CsrMatrixView::new(n, n, &row_ptr, &col_ind, &values).map_err(map_linalg_error)?;
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
    )
    .map_err(map_linalg_error)?;
    Ok(DVector::from_vec(x))
}

/// Solves Tikhonov / Ridge-regularized least squares:
/// $$\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2, \qquad \lambda \ge 0$$
///
/// Solves the augmented $(m + n) \times n$ system $\begin{bmatrix} A \\ \sqrt{\lambda} I_n \end{bmatrix} x \approx \begin{bmatrix} b \\ 0 \end{bmatrix}$
/// via pivoted Householder QR (with SVD fallback) so $A^T A$ is never explicitly formed.
#[verified_engine::verified]
pub fn solve_ridge_regression(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    lambda: f64,
) -> Result<DVector<f64>, LinearAlgebraError> {
    if !lambda.is_finite() {
        return Err(LinearAlgebraError::NonFiniteInput);
    }
    validate_system_inputs(a, b)?;
    if lambda < 0.0 {
        return Err(LinearAlgebraError::NotPositiveDefinite);
    }
    if lambda == 0.0 {
        return solve_normal_equation(a, b);
    }

    let (m, n) = (a.nrows(), a.ncols());
    let sqrt_lambda = lambda.sqrt();
    let mut aug_a = DMatrix::zeros(m + n, n);
    aug_a.view_mut((0, 0), (m, n)).copy_from(a);
    for j in 0..n {
        aug_a[(m + j, j)] = sqrt_lambda;
    }
    let mut aug_b = DVector::zeros(m + n);
    aug_b.rows_mut(0, m).copy_from(b);
    solve_normal_equation(&aug_a, &aug_b)
}

fn solve_passive_subproblem(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    passive: &[bool],
) -> Result<DVector<f64>, LinearAlgebraError> {
    let indices: Vec<usize> = passive
        .iter()
        .enumerate()
        .filter_map(|(j, &is_p)| is_p.then_some(j))
        .collect();
    let mut s = DVector::zeros(a.ncols());
    if indices.is_empty() {
        return Ok(s);
    }
    let mut a_p = DMatrix::zeros(a.nrows(), indices.len());
    for (col_pos, &orig_col) in indices.iter().enumerate() {
        a_p.set_column(col_pos, &a.column(orig_col));
    }
    let z = solve_normal_equation(&a_p, b)?;
    for (col_pos, &orig_col) in indices.iter().enumerate() {
        s[orig_col] = z[col_pos];
    }
    Ok(s)
}

fn run_nnls_inner_loop(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
    x: &mut DVector<f64>,
    passive: &mut [bool],
    zero_tol: f64,
) -> Result<bool, LinearAlgebraError> {
    let n = a.ncols();
    let mut progressed = false;
    for _ in 0..=n {
        if !passive.iter().any(|&p| p) {
            return Ok(progressed);
        }
        let s = solve_passive_subproblem(a, b, passive)?;
        if passive
            .iter()
            .enumerate()
            .all(|(j, &is_p)| !is_p || s[j] > zero_tol)
        {
            *x = s;
            return Ok(true);
        }
        let alpha = passive
            .iter()
            .enumerate()
            .filter(|&(j, &is_p)| is_p && s[j] <= zero_tol)
            .map(|(j, _)| {
                let denom = x[j] - s[j];
                if denom > 0.0 {
                    (x[j] / denom).clamp(0.0, 1.0)
                } else {
                    0.0
                }
            })
            .fold(1.0_f64, f64::min);
        if alpha > 0.0 {
            progressed = true;
        }
        *x += alpha * (&s - &*x);
        for j in 0..n {
            if passive[j] && x[j] <= zero_tol {
                x[j] = 0.0;
                passive[j] = false;
            }
        }
    }
    Err(LinearAlgebraError::ConvergenceFailed)
}

/// Solves Non-Negative Least Squares (NNLS) via the active-set Lawson-Hanson algorithm:
/// $$\min_{x \ge 0} \|Ax - b\|_2^2$$
#[verified_engine::verified]
pub fn solve_nnls(a: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, LinearAlgebraError> {
    validate_system_inputs(a, b)?;
    let (m, n) = (a.nrows(), a.ncols());
    let mut x = DVector::zeros(n);
    let mut passive = vec![false; n];
    let mut rejected = vec![false; n];
    let grad_tol = f64::EPSILON * (m.max(n) as f64) * 100.0 * a.norm().max(1.0);
    let zero_tol = f64::EPSILON * 16.0;
    let max_outer = (10 * n * n).clamp(100, 5_000);

    for _ in 0..max_outer {
        let w = a.transpose() * (b - a * &x);
        let best = (0..n)
            .filter(|&j| !passive[j] && !rejected[j])
            .max_by(|&i, &j| w[i].total_cmp(&w[j]));
        let Some(t) = best.filter(|&j| w[j] > grad_tol) else {
            for val in x.iter_mut() {
                if *val < 0.0 {
                    *val = 0.0;
                }
            }
            return Ok(x);
        };
        passive[t] = true;
        if run_nnls_inner_loop(a, b, &mut x, &mut passive, zero_tol)? {
            rejected.fill(false);
        } else {
            rejected[t] = true;
        }
    }
    Err(LinearAlgebraError::ConvergenceFailed)
}

// [cite:advanced_linear_algebra]
