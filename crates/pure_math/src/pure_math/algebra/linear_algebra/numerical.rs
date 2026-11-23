//! Numerical Linear Algebra Solvers.
//!
//! Provides numerically stable implementations for solving square linear systems,
//! unconstrained least squares (Householder QR with thin SVD fallback), Moore-Penrose
//! pseudoinverses, Tikhonov / Ridge regression (augmented QR), and active-set
//! Lawson-Hanson non-negative least squares (NNLS).

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
    /// Input contains `NaN` or infinite floating-point values.
    #[error("Input contains non-finite (NaN or infinite) values")]
    NonFiniteInput,
    /// Iterative decomposition or constrained solver failed to converge within its iteration cap.
    #[error("Iterative decomposition or solver failed to converge")]
    ConvergenceFailed,
    /// Matrix or regularization parameter is not positive definite.
    #[error("Matrix or regularization penalty is not positive definite")]
    NotPositiveDefinite,
}

pub(crate) fn ensure_finite_matrix(matrix: &DMatrix<f64>) -> Result<(), LinearAlgebraError> {
    if !matrix.iter().all(|v| v.is_finite()) {
        return Err(LinearAlgebraError::NonFiniteInput);
    }
    if matrix.nrows() == 0 || matrix.ncols() == 0 {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    Ok(())
}

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

fn try_householder_qr_solve(a: &DMatrix<f64>, b: &DVector<f64>) -> Option<DVector<f64>> {
    let (m, n) = (a.nrows(), a.ncols());
    if m < n {
        return None;
    }
    let (q, r) = a.clone().qr().unpack();
    let max_diag = (0..n).map(|i| r[(i, i)].abs()).fold(0.0_f64, f64::max);
    let qr_tol = f64::EPSILON * (m.max(n) as f64) * 64.0 * max_diag;
    if max_diag <= f64::EPSILON || (0..n).any(|i| r[(i, i)].abs() <= qr_tol) {
        return None;
    }
    let qtb = q.transpose() * b;
    let sol = r.solve_upper_triangular(&qtb)?;
    sol.iter().all(|v| v.is_finite()).then_some(sol)
}

fn solve_thin_svd_fallback(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let (u, s, v_t) = super::decomposition::svd_decomposition(a)?;
    let max_sigma = s.iter().copied().fold(0.0_f64, f64::max);
    if max_sigma <= f64::EPSILON {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    let svd_tol = f64::EPSILON * (a.nrows().max(a.ncols()) as f64) * max_sigma;
    let mut ut_b = u.transpose() * b;
    for i in 0..s.len() {
        ut_b[i] = if s[i] > svd_tol { ut_b[i] / s[i] } else { 0.0 };
    }
    let sol = v_t.transpose() * ut_b;
    if !sol.iter().all(|v| v.is_finite()) {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    Ok(sol)
}

/// Solves a square linear system $Ax = b$ using LU decomposition with partial pivoting.
///
/// # Arguments
///
/// * `a` - The square coefficient matrix $A$.
/// * `b` - The right-hand side vector $b$.
///
/// # Returns
///
/// * `Result<DVector<f64>, LinearAlgebraError>` - The solution vector $x$.
#[verified_engine::verified]
pub fn solve_linear_system(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    validate_system_inputs(a, b)?;
    if a.nrows() != a.ncols() {
        return Err(LinearAlgebraError::DimensionMismatch);
    }
    let sol = a
        .clone()
        .lu()
        .solve(b)
        .ok_or(LinearAlgebraError::SolutionNotFound)?;
    if !sol.iter().all(|v| v.is_finite()) {
        return Err(LinearAlgebraError::SingularMatrix);
    }
    Ok(sol)
}

/// Solves the Linear Least-Squares problem $\min_{\mathbf{c}} \|A\mathbf{c} - \mathbf{b}\|_2^2$.
///
/// Uses thin Householder $QR$ factorization ($R\mathbf{c} = Q^T\mathbf{b}$) without forming
/// $A^T A$ (preventing $\kappa(A)^2$ condition-number squaring), and falls back to truncated
/// thin SVD when $A$ is underdetermined or $R$ exhibits near-zero pivots.
///
/// # Arguments
///
/// * `a` - Design matrix $A \in \mathbb{R}^{m \times n}$.
/// * `b` - Observation vector $b \in \mathbb{R}^m$.
///
/// # Returns
///
/// * `Result<DVector<f64>, LinearAlgebraError>` - The least-squares coefficient vector $\mathbf{c}$.
#[verified_engine::verified]
pub fn solve_normal_equation(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    validate_system_inputs(a, b)?;
    if let Some(sol) = try_householder_qr_solve(a, b) {
        return Ok(sol);
    }
    solve_thin_svd_fallback(a, b)
}

/// Calculates the Moore-Penrose Pseudoinverse ($A^{\dagger}$) via thin SVD.
///
/// Singular values at or below `epsilon` are treated as zero.
///
/// # Arguments
///
/// * `matrix` - Input matrix $A$.
/// * `epsilon` - Non-negative singular-value cutoff threshold.
///
/// # Returns
///
/// * `Result<DMatrix<f64>, LinearAlgebraError>` - The pseudoinverse matrix $A^{\dagger}$.
#[verified_engine::verified]
pub fn moore_penrose_pseudoinverse(
    matrix: &DMatrix<f64>,
    epsilon: f64,
) -> Result<DMatrix<f64>, LinearAlgebraError> {
    if !epsilon.is_finite() {
        return Err(LinearAlgebraError::NonFiniteInput);
    }
    ensure_finite_matrix(matrix)?;
    if epsilon < 0.0 {
        return Err(LinearAlgebraError::NotPositiveDefinite);
    }
    let (u, s, v_t) = super::decomposition::svd_decomposition(matrix)?;
    let mut v_scaled = v_t.transpose();
    for i in 0..s.len() {
        let inv = if s[i] > epsilon { 1.0 / s[i] } else { 0.0 };
        v_scaled.column_mut(i).scale_mut(inv);
    }
    Ok(v_scaled * u.transpose())
}

/// Solves Tikhonov / Ridge-regularized least squares:
/// $$\min_x \|Ax - b\|_2^2 + \lambda \|x\|_2^2, \qquad \lambda \ge 0$$
///
/// Solves the augmented $(m + n) \times n$ system $\begin{bmatrix} A \\ \sqrt{\lambda} I_n \end{bmatrix} x \approx \begin{bmatrix} b \\ 0 \end{bmatrix}$
/// via Householder QR (with thin SVD fallback) so $A^T A$ is never explicitly formed.
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
