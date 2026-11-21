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
//! *   For symmetric matrices, this becomes $A = LDL^T$.
//!
//! ## Singular Value Decomposition (SVD)
//! *   Any $m \times n$ matrix $A$ can be factored into:
//!     $$A = U \Sigma V^T$$
//! *   **U (Left Singular Vectors):** An $m \times m$ orthogonal matrix containing the eigenvectors of $AA^T$.
//! *   **V (Right Singular Vectors):** An $n \times n$ orthogonal matrix containing the eigenvectors of $A^T A$.
//! *   **$\Sigma$ (Singular Values):** An $m \times n$ diagonal matrix containing the **singular values** $\sigma_i \ge 0$.
//! *   **Formula:** The singular values are the square roots of the non-zero eigenvalues of $A^T A$ (or $AA^T$):
//!     $$\sigma_i = \sqrt{\lambda_i(A^T A)}$$
//! *   **Rank:** The number of non-zero singular values equals the **rank** of the matrix.
//! *   **Application:** SVD is used for least squares estimation, image compression (by keeping only the largest $\sigma_i$), and determining the effective rank of a matrix.

use nalgebra::DMatrix;

/// Computes the effective rank of a matrix using Singular Value Decomposition (SVD).
///
/// The effective rank is determined by counting singular values greater than a given tolerance.
///
/// # Arguments
///
/// * `matrix` - The input matrix.
/// * `tolerance` - The threshold below which singular values are considered zero.
///   If `None`, a default tolerance of math_commons::registry::TOLERANCE_HIGH is used.
///
/// # Returns
///
/// The number of singular values greater than the tolerance.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::linear_algebra::decomposition::effective_rank;
/// use math_explorer::nalgebra::DMatrix;
///
/// let diag = DMatrix::from_diagonal(&math_explorer::nalgebra::DVector::from_vec(vec![1.0, 0.00000000001, 3.0]));
/// // With high tolerance, rank is 2.
/// assert_eq!(effective_rank(&diag, Some(1e-5)), 2);
/// ```
#[verified_engine::verified]
pub fn effective_rank(matrix: &DMatrix<f64>, tolerance: Option<f64>) -> usize {
    let svd = matrix.clone().svd(false, false);
    let tol = tolerance.unwrap_or(math_commons::registry::TOLERANCE_HIGH);
    svd.singular_values
        .iter()
        .filter(|&sigma| *sigma > tol)
        .count()
}

/// Computes a low-rank approximation of a matrix using SVD.
///
/// This reconstructs the matrix using only the top `k` singular values.
/// This is widely used in image compression and noise reduction.
///
/// # Arguments
/// * `matrix` - The input matrix.
/// * `k` - The rank of the approximation.
///
/// # Returns
/// The approximated matrix $A_k$.
///
/// # Panics
/// Panics if SVD computation fails (e.g., non-convergence), which is rare.
#[verified_engine::verified]
pub fn low_rank_approximation(matrix: &DMatrix<f64>, k: usize) -> DMatrix<f64> {
    let svd = matrix.clone().svd(true, true);
    let u = svd.u.as_ref().expect("SVD failed to compute U");
    let v_t = svd.v_t.as_ref().expect("SVD failed to compute V^T");
    let s = &svd.singular_values;

    let mut s_k = DMatrix::zeros(u.ncols(), v_t.nrows());
    for i in 0..k.min(s.len()) {
        s_k[(i, i)] = s[i];
    }

    u * s_k * v_t
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
    fn test_low_rank_approx() {
        // Rank 2 matrix
        let m = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 1.0]);
        // Approx rank 1 should keep 2.0 (largest singular value)
        let approx = low_rank_approximation(&m, 1);

        // Approx should be [2, 0; 0, 0] since V^T and U align with axes for diagonal matrix
        // But nalgebra might return different U/V but A_k should be unique for distinct singular values.

        // Let's check Frobenius norm of difference
        // Difference should be mostly the removed singular value (1.0)
        let diff = &m - &approx;
        let norm = diff.norm(); // Frobenius norm

        assert!((norm - 1.0).abs() < 1e-5);
    }
}
