//! Core types for copula operations with Cholesky positive-definiteness verification.

use crate::error::CopulaError;
use nalgebra::{Cholesky, DMatrix};

/// A probability value in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Creates a validated probability in `[0, 1]`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::Probability;
    ///
    /// let p = Probability::new(0.75).unwrap();
    /// assert_eq!(p.value(), 0.75);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, CopulaError> {
        if !(0.0..=1.0).contains(&value) || !value.is_finite() {
            return Err(CopulaError::InvalidProbability { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw probability value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// A correlation coefficient in `[-1, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correlation(f64);

impl Correlation {
    /// Creates a validated correlation coefficient in `[-1, 1]`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::Correlation;
    ///
    /// let rho = Correlation::new(-0.3).unwrap();
    /// assert_eq!(rho.value(), -0.3);
    /// ```
    #[verified_engine::verified]
    pub fn new(value: f64) -> Result<Self, CopulaError> {
        if !(-1.0..=1.0).contains(&value) || !value.is_finite() {
            return Err(CopulaError::InvalidCorrelation { value });
        }
        Ok(Self(value))
    }

    /// Returns the raw correlation value.
    #[verified_engine::verified]
    pub fn value(&self) -> f64 {
        self.0
    }
}

/// A symmetric, strictly positive-definite correlation matrix with unit diagonal.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationMatrix {
    matrix: DMatrix<f64>,
    cholesky: DMatrix<f64>,
}

impl CorrelationMatrix {
    /// Validates symmetry, unit diagonal, correlation bounds, and strict Cholesky positive-definiteness.
    ///
    /// # Errors
    /// - [`CopulaError::NotSymmetric`] if `|R_ij - R_ji|` exceeds tolerance.
    /// - [`CopulaError::InvalidCorrelation`] if any entry is non-finite or outside `[-1, 1]`.
    /// - [`CopulaError::NotPositiveDefinite`] if Cholesky factorization $R = L L^T$ fails.
    #[verified_engine::verified]
    pub fn new(matrix: DMatrix<f64>) -> Result<Self, CopulaError> {
        validate_square_and_entries(&matrix)?;
        let sym = symmetrize_unit_diagonal(&matrix);
        let chol = Cholesky::new(sym).ok_or(CopulaError::NotPositiveDefinite)?;
        let lower = chol.unpack();
        for i in 0..lower.nrows() {
            if !lower[(i, i)].is_finite() || lower[(i, i)] <= math_commons::registry::TOLERANCE_HIGH
            {
                return Err(CopulaError::NotPositiveDefinite);
            }
        }
        Ok(Self {
            matrix,
            cholesky: lower,
        })
    }

    /// Creates a bivariate correlation matrix `[[1, rho], [rho, 1]]` for `|rho| < 1`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::{CorrelationMatrix, Correlation};
    ///
    /// let rho = Correlation::new(-0.3).unwrap();
    /// let matrix = CorrelationMatrix::bivariate(rho).unwrap();
    /// assert_eq!(matrix.dimension(), 2);
    /// ```
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let r = rho.value();
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, r, r, 1.0]);
        Self::new(matrix)
    }

    /// Creates a $d \times d$ equicorrelated matrix with off-diagonal correlation `rho`.
    #[verified_engine::verified]
    pub fn equicorrelated(dimension: usize, rho: Correlation) -> Result<Self, CopulaError> {
        if dimension == 0 {
            return Err(CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(1),
                    actual: math_commons::math_kernel::types::Dimension(0),
                },
            ));
        }
        let r = rho.value();
        let mat = DMatrix::from_fn(dimension, dimension, |i, j| if i == j { 1.0 } else { r });
        Self::new(mat)
    }

    /// Returns the underlying correlation matrix $R$.
    #[verified_engine::verified]
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Returns the lower-triangular Cholesky factor $L$ satisfying $R = L L^T$.
    #[verified_engine::verified]
    pub fn cholesky_factor(&self) -> &DMatrix<f64> {
        &self.cholesky
    }

    /// Returns the dimension of the matrix.
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the correlation between variables `i` and `j`.
    #[verified_engine::verified]
    pub fn get_correlation(&self, i: usize, j: usize) -> Option<f64> {
        if i < self.dimension() && j < self.dimension() {
            Some(self.matrix[(i, j)])
        } else {
            None
        }
    }
}

fn validate_square_and_entries(matrix: &DMatrix<f64>) -> Result<(), CopulaError> {
    if matrix.nrows() == 0 || matrix.nrows() != matrix.ncols() {
        return Err(CopulaError::Math(
            math_commons::error::MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(matrix.nrows().max(1)),
                actual: math_commons::math_kernel::types::Dimension(matrix.ncols()),
            },
        ));
    }
    let tol = math_commons::registry::TOLERANCE_HIGH;
    for i in 0..matrix.nrows() {
        for j in 0..matrix.ncols() {
            let val = matrix[(i, j)];
            if !val.is_finite() || val.abs() > 1.0 + tol {
                return Err(CopulaError::InvalidCorrelation { value: val });
            }
        }
        if (matrix[(i, i)] - 1.0).abs() > tol {
            return Err(CopulaError::Math(
                math_commons::error::MathError::NumericalError {
                    reason: format!("Diagonal element {} is not 1.0", i),
                },
            ));
        }
        for j in (i + 1)..matrix.ncols() {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > tol {
                return Err(CopulaError::NotSymmetric);
            }
        }
    }
    Ok(())
}

fn symmetrize_unit_diagonal(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let n = matrix.nrows();
    DMatrix::from_fn(n, n, |i, j| {
        if i == j {
            1.0
        } else {
            0.5 * (matrix[(i, j)] + matrix[(j, i)])
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_probability_valid_and_invalid() {
        assert_eq!(Probability::new(0.5).unwrap().value(), 0.5);
        assert_eq!(Probability::new(0.0).unwrap().value(), 0.0);
        assert_eq!(Probability::new(1.0).unwrap().value(), 1.0);
        assert!(Probability::new(-0.1).is_err());
        assert!(Probability::new(1.1).is_err());
        assert!(Probability::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_valid_and_invalid() {
        assert_eq!(Correlation::new(0.5).unwrap().value(), 0.5);
        assert_eq!(Correlation::new(-0.8).unwrap().value(), -0.8);
        assert!(Correlation::new(-1.1).is_err());
        assert!(Correlation::new(1.1).is_err());
        assert!(Correlation::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_bivariate_and_symmetry() {
        let rho = Correlation::new(0.6).unwrap();
        let matrix = CorrelationMatrix::bivariate(rho).unwrap();
        assert_eq!(matrix.dimension(), 2);
        assert_eq!(matrix.get_correlation(0, 1), Some(0.6));
        assert_eq!(matrix.get_correlation(1, 0), Some(0.6));
        assert_eq!(matrix.get_correlation(0, 0), Some(1.0));

        let bad_diag = DMatrix::from_row_slice(2, 2, &[0.5, 0.3, 0.3, 1.0]);
        assert!(CorrelationMatrix::new(bad_diag).is_err());

        let asym = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.5, 1.0]);
        assert_eq!(CorrelationMatrix::new(asym), Err(CopulaError::NotSymmetric));
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_rejects_indefinite_and_singular_matrices() {
        // Symmetric 3x3 with unit diagonal and entries in [-1, 1], but negative eigenvalue (-0.2)
        let indefinite =
            DMatrix::from_row_slice(3, 3, &[1.0, -0.6, -0.6, -0.6, 1.0, -0.6, -0.6, -0.6, 1.0]);
        assert_eq!(
            CorrelationMatrix::new(indefinite),
            Err(CopulaError::NotPositiveDefinite)
        );

        // Singular rank-1 correlation matrix (rho = 1.0)
        let singular = DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            CorrelationMatrix::new(singular),
            Err(CopulaError::NotPositiveDefinite)
        );
    }
}
