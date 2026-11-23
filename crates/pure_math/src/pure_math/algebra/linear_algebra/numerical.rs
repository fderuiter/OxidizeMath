//! Numerical Linear Algebra Solvers.
//!
//! Provides implementations for solving linear systems, least squares, and matrix operations
//! commonly used in engineering and physics problems.

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
}

/// Solves the square dense linear system $A\mathbf{x} = \mathbf{b}$ via column-pivoted LU factorization $PA = LU$.
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Partial pivoting bounds element growth in practice with backward error $O(\varepsilon_{\text{mach}})$ and forward error $O(\kappa_2(A)\varepsilon_{\text{mach}})$.
///
/// # Domain of Validity
/// Requires square non-singular matrix $A \in \mathbb{R}^{n \times n}$ and right-hand side $\mathbf{b} \in \mathbb{R}^n$ of matching dimension.
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

/// Solves the linear least-squares problem $\min_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|_2^2$ via the normal equations $A^\top A \mathbf{x} = A^\top \mathbf{b}$.
///
/// Reference: Golub & Van Loan (2013), *Matrix Computations*, DOI: 10.56021/9781421407944; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Forming $A^\top A$ squares the spectral condition number $\kappa_2(A^\top A) = \kappa_2(A)^2$; ill-conditioned designs ($\kappa_2(A) > 10^7$) lose half-precision accuracy.
///
/// # Domain of Validity
/// Requires full-column-rank design matrix $A \in \mathbb{R}^{m \times n}$ ($m \ge n$) and compatible target vector $\mathbf{b} \in \mathbb{R}^m$.
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
