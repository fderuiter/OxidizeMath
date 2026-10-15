// [cite:advanced_linear_algebra]
//! # Canonical Forms
//!
//! Canonical forms categorize matrices that cannot be fully diagonalized (defective matrices) or provide a standard representation for similarity classes.
//!
//! ## Jordan Canonical Form
//! *   If a matrix $A$ has $s$ linearly independent eigenvectors, it is similar to a block diagonal matrix $J$ (i.e., $J = M^{-1}AM$), where:
//!     $$J = \begin{bmatrix} J_1 & & \\ & \ddots & \\ & & J_s \end{bmatrix}$$
//! *   Each **Jordan block** $J_i$ is a triangular matrix with a single eigenvalue $\lambda_i$ on the diagonal and $1$s on the super-diagonal (just above the main diagonal):
//!     $$J_i = \begin{bmatrix} \lambda_i & 1 & & \\ & \lambda_i & \ddots & \\ & & \ddots & 1 \\ & & & \lambda_i \end{bmatrix}$$
//! *   This form is essential for solving differential equations involving defective matrices (where geometric multiplicity < algebraic multiplicity), introducing terms like $t e^{\lambda t}$ into the solution.
//!
//! ## Rational Canonical Form
//! *   This form exists even when the characteristic polynomial does not factor into linear terms (e.g., over the rational numbers).
//! *   A linear operator $T$ has a unique block diagonal representation $M = \text{diag}(C_1, \dots, C_r)$, where each $C_i$ is a **companion matrix** associated with specific polynomials (elementary divisors or invariant factors).
//! *   A companion matrix for a polynomial $f(t) = t^n + a_{n-1}t^{n-1} + \dots + a_0$ has $1$s on the sub-diagonal and the negative coefficients of $f(t)$ in the last column.

use nalgebra::{Complex, DMatrix};

/// Represents a Jordan Block.
pub struct JordanBlock {
    #[allow(missing_docs)]
    pub eigenvalue: Complex<f64>,
    #[allow(missing_docs)]
    pub size: usize,
}

impl JordanBlock {
    /// Creates a new Jordan Block.
    #[verified_engine::verified]
    pub fn new(eigenvalue: Complex<f64>, size: usize) -> Self {
        Self { eigenvalue, size }
    }

    /// Converts the Jordan Block to a dense matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_math::pure_math::algebra::linear_algebra::canonical::JordanBlock;
    /// use nalgebra::Complex;
    ///
    /// let block = JordanBlock::new(Complex::new(2.0, 0.0), 3);
    /// let mat = block.to_matrix();
    ///
    /// assert_eq!(mat[(0, 0)], Complex::new(2.0, 0.0));
    /// assert_eq!(mat[(0, 1)], Complex::new(1.0, 0.0)); // Super-diagonal
    /// assert_eq!(mat[(1, 0)], Complex::new(0.0, 0.0));
    /// ```
    #[verified_engine::verified]
    pub fn to_matrix(&self) -> DMatrix<Complex<f64>> {
        let mut matrix = DMatrix::from_element(self.size, self.size, Complex::new(0.0, 0.0));
        for i in 0..self.size {
            matrix[(i, i)] = self.eigenvalue;
            if i + 1 < self.size {
                matrix[(i, i + 1)] = Complex::new(1.0, 0.0);
            }
        }
        matrix
    }
}

/// Represents a Companion Matrix for a polynomial $f(t) = t^n + a_{n-1}t^{n-1} + \dots + a_0$.
///
/// The coefficients are stored in order $[a_0, a_1, \dots, a_{n-1}]$.
pub struct CompanionMatrix {
    #[allow(missing_docs)]
    pub coeffs: Vec<f64>,
}

impl CompanionMatrix {
    /// Creates a new Companion Matrix.
    ///
    /// # Arguments
    /// * `coeffs` - The coefficients of the polynomial $[a_0, a_1, \dots, a_{n-1}]$.
    #[verified_engine::verified]
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    /// Converts the Companion Matrix to a dense matrix.
    ///
    /// The matrix has 1s on the sub-diagonal and negative coefficients in the last column.
    ///
    /// # Examples
    ///
    /// ```
    /// use pure_math::pure_math::algebra::linear_algebra::canonical::CompanionMatrix;
    ///
    /// // Polynomial t^2 + 3t + 2 => coeffs = [2.0, 3.0]
    /// let companion = CompanionMatrix::new(vec![2.0, 3.0]);
    /// let mat = companion.to_matrix();
    ///
    /// // Expected:
    /// // [ 0, -2 ]
    /// // [ 1, -3 ]
    /// assert_eq!(mat[(0, 1)], -2.0);
    /// assert_eq!(mat[(1, 1)], -3.0);
    /// assert_eq!(mat[(1, 0)], 1.0);
    /// ```
    #[verified_engine::verified]
    pub fn to_matrix(&self) -> DMatrix<f64> {
        let n = self.coeffs.len();
        let mut matrix = DMatrix::zeros(n, n);

        for i in 0..n {
            // Last column gets -a_i
            matrix[(i, n - 1)] = -self.coeffs[i];

            // Sub-diagonal gets 1
            if i + 1 < n {
                matrix[(i + 1, i)] = 1.0;
            }
        }
        matrix
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Complex;

    #[test]
    #[verified_engine::verified]
    fn test_jordan_block() {
        let block = JordanBlock::new(Complex::new(3.0, 0.0), 2);
        let m = block.to_matrix();
        assert_eq!(m[(0, 0)], Complex::new(3.0, 0.0));
        assert_eq!(m[(0, 1)], Complex::new(1.0, 0.0));
        assert_eq!(m[(1, 1)], Complex::new(3.0, 0.0));
        assert_eq!(m[(1, 0)], Complex::new(0.0, 0.0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_companion_matrix() {
        // t^2 + 5t + 6 => coeffs [6, 5]
        let c = CompanionMatrix::new(vec![6.0, 5.0]);
        let m = c.to_matrix();
        // 0  -6
        // 1  -5
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 2);
        assert_eq!(m[(0, 0)], 0.0);
        assert_eq!(m[(0, 1)], -6.0);
        assert_eq!(m[(1, 0)], 1.0);
        assert_eq!(m[(1, 1)], -5.0);
    }
}
