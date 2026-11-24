//! Numerical Linear Algebra Solvers.
//!
//! Provides implementations for solving linear systems, least squares, and matrix operations
//! commonly used in engineering and physics problems.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors related to Linear Algebra operations.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
}

/// Solves a linear system $Ax = b$ using LU decomposition (or best available method).
///
/// # Arguments
///
/// * `a` - The matrix $A$.
/// * `b` - The vector $b$.
///
/// # Returns
///
/// * `Result<DVector<f64>, LinearAlgebraError>` - The solution vector $x$.
#[verified_engine::verified]
pub fn solve_linear_system(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    a.clone()
        .lu()
        .solve(b)
        .ok_or(LinearAlgebraError::SolutionNotFound)
}

/// Solves the Normal Equation for Least Squares.
///
/// $$ \mathbf{c} = (A^T A)^{-1} A^T \mathbf{b} $$
///
/// Used for finding the optimal coefficients that minimize the error squared.
///
/// # Arguments
///
/// * `a` - Design matrix $A$.
/// * `b` - Observation vector $b$.
///
/// # Returns
///
/// * `Result<DVector<f64>, LinearAlgebraError>` - The coefficient vector $\mathbf{c}$.
#[verified_engine::verified]
pub fn solve_normal_equation(
    a: &DMatrix<f64>,
    b: &DVector<f64>,
) -> Result<DVector<f64>, LinearAlgebraError> {
    let a_t = a.transpose();
    let ata = &a_t * a;
    let atb = &a_t * b;

    // Invert (A^T A)
    // Note: Cholesky decomposition is more efficient for symmetric positive definite matrices (like A^T A)
    match ata.clone().cholesky() {
        Some(cholesky) => Ok(cholesky.solve(&atb)),
        None => ata
            .try_inverse()
            .map(|inv| &inv * &atb)
            .ok_or(LinearAlgebraError::SingularMatrix),
    }
}

/// Calculates the Moore-Penrose Pseudoinverse ($J^{\dagger}$).
///
/// $$ \dot{\gamma} = J^{\dagger} \tau_{cmd} $$
///
/// # Arguments
///
/// * `matrix` - The matrix $J$.
/// * `epsilon` - Tolerance for singular values (to handle near-singularities).
///
/// # Returns
///
/// * `DMatrix<f64>` - The pseudoinverse matrix.
#[verified_engine::verified]
pub fn moore_penrose_pseudoinverse(matrix: &DMatrix<f64>, epsilon: f64) -> DMatrix<f64> {
    match matrix.clone().svd(true, true).pseudo_inverse(epsilon) {
        Ok(pinv) => pinv,
        Err(_) => DMatrix::zeros(matrix.ncols(), matrix.nrows()), // Should generally not happen with SVD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_normal_equation() {
        // Fit y = mx + c
        // Points: (1, 1), (2, 2), (3, 3)
        // A = [[1, 1], [1, 2], [1, 3]] (cols: c, m)
        // b = [1, 2, 3]
        let a = DMatrix::from_row_slice(3, 2, &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0]);
        let b = DVector::from_column_slice(&[1.0, 2.0, 3.0]);

        let c_vec = solve_normal_equation(&a, &b).unwrap();
        // Expect c=0, m=1
        assert!(c_vec[0].abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((c_vec[1] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }
}
