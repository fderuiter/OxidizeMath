//! # Eigenvalues and Eigenvectors
//!
//! The core problem of linear algebra is to find those special vectors $x$ that are not changed in direction by a transformation $A$, but only scaled.
//!
//! ## Definition
//! For a square matrix $A$, a scalar $\lambda$ is an **eigenvalue** and a non-zero vector $x$ is the corresponding **eigenvector** if they satisfy the equation:
//! $$Ax = \lambda x \quad \text{or} \quad (A - \lambda I)x = 0$$
//! where $I$ is the identity matrix.
//!
//! ## The Characteristic Equation
//! To find the eigenvalues, one solves for the roots of the characteristic polynomial:
//! $$\det(A - \lambda I) = 0$$
//! This is a polynomial of degree $N$ for an $N \times N$ matrix. The sum of the eigenvalues equals the **trace** ($\text{Tr} A = \sum \lambda_i$), and their product equals the **determinant** ($\det A = \prod \lambda_i$).
//!
//! ## Symmetric and Hermitian Matrices
//! *   A real symmetric matrix ($A^T = A$) or a complex Hermitian matrix ($A^\dagger = A$) guarantees that all eigenvalues are **real numbers**.
//! *   Eigenvectors corresponding to distinct eigenvalues are **orthogonal** (or can be chosen to be orthonormal).
//! *   **Spectral Theorem:** A real symmetric matrix can be factored as $A = Q\Lambda Q^T$, where $Q$ is an orthogonal matrix of eigenvectors and $\Lambda$ is a diagonal matrix of eigenvalues.
//!
//! ## Physical Interpretations
//!
//! ### Stability Analysis
//! In systems of differential equations $u' = Au$, the solution involves terms like $e^{\lambda t}$.
//! *   The system is **stable** (solutions approach zero) if all eigenvalues have negative real parts ($\text{Re}(\lambda) < 0$).
//! *   It is **unstable** if any $\text{Re}(\lambda) > 0$. Purely imaginary eigenvalues correspond to neutral stability (oscillations).
//!
//! ### Vibration and Normal Modes
//! In oscillating systems (like masses connected by springs), the equations of motion often take the form $\ddot{x} = Ax$.
//! *   We assume a solution $x(t) = v e^{i\omega t}$, leading to the eigenvalue problem $Av = -\omega^2 v$, where $\lambda = -\omega^2$.
//! *   The square roots of the eigenvalues give the **natural frequencies** ($\omega$) of the system, and the eigenvectors represent the **normal modes** (the patterns of vibration where all parts move sinusoidally with the same frequency).

use nalgebra::DMatrix;

/// Checks if a linear dynamical system represented by matrix $A$ is stable.
///
/// Stability requires all eigenvalues to have negative real parts.
///
/// # Arguments
///
/// * `matrix` - The system matrix $A$.
///
/// # Returns
///
/// `true` if stable, `false` otherwise.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::linear_algebra::eigen::is_stable;
/// use math_explorer::nalgebra::DMatrix;
///
/// let stable_matrix = DMatrix::from_row_slice(2, 2, &[-2.0, 0.0, 0.0, -3.0]);
/// assert!(is_stable(&stable_matrix));
///
/// let unstable_matrix = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, -3.0]);
/// assert!(!is_stable(&unstable_matrix));
/// ```
#[verified_engine::verified]
pub fn is_stable(matrix: &DMatrix<f64>) -> bool {
    let eigens = matrix.complex_eigenvalues();
    eigens.iter().all(|lambda| lambda.re < 0.0)
}

/// Calculates the natural frequencies of an oscillating system $\ddot{x} = Ax$.
///
/// In this context, eigenvalues $\lambda$ are related to frequencies $\omega$ by $\lambda = -\omega^2$.
/// Thus, $\omega = \sqrt{-\lambda}$.
///
/// # Arguments
///
/// * `matrix` - The system matrix $A$.
///
/// # Returns
///
/// A vector of natural frequencies (magnitude). If $-\lambda$ is negative (unstable mode), returns NaN for that mode.
///
/// # Examples
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::algebra::linear_algebra::eigen::natural_frequencies;
/// use math_explorer::nalgebra::DMatrix;
///
/// // System with eigenvalues -4 and -9
/// let matrix = DMatrix::from_row_slice(2, 2, &[-4.0, 0.0, 0.0, -9.0]);
/// let freqs = natural_frequencies(&matrix);
/// // Frequencies should be 2.0 and 3.0
/// assert!((freqs[0] - 2.0).abs() < 1e-5 || (freqs[0] - 3.0).abs() < 1e-5);
/// ```
#[verified_engine::verified]
pub fn natural_frequencies(matrix: &DMatrix<f64>) -> Vec<f64> {
    let eigens = matrix.complex_eigenvalues();
    eigens
        .iter()
        .map(|lambda| {
            // We expect lambda to be real and negative for stable oscillation.
            // lambda = -omega^2
            // -lambda = omega^2
            let neg_lambda = -lambda;

            // Check if effectively real
            if neg_lambda.im.abs() < math_commons::registry::TOLERANCE_HIGH {
                if neg_lambda.re >= 0.0 {
                    neg_lambda.re.sqrt()
                } else {
                    f64::NAN
                }
            } else {
                // Complex eigenvalue case? For now treat magnitude or NaN.
                // Given the context of "Natural Frequencies", usually implies real.
                f64::NAN
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    #[verified_engine::verified]
    fn test_stability() {
        // Stable: -1, -2
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![-1.0, -2.0]));
        assert!(is_stable(&m));

        // Unstable: 1, -2
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, -2.0]));
        assert!(!is_stable(&m));
    }

    #[test]
    #[verified_engine::verified]
    fn test_natural_frequencies() {
        // Eigenvalues -4, -9 => Frequencies 2, 3
        let m = DMatrix::from_diagonal(&DVector::from_vec(vec![-4.0, -9.0]));
        let freqs = natural_frequencies(&m);
        assert!(freqs.iter().any(|&f| (f - 2.0).abs() < 1e-5));
        assert!(freqs.iter().any(|&f| (f - 3.0).abs() < 1e-5));
    }
}
