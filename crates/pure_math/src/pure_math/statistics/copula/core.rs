//! Core types for copula operations.

use crate::error::CopulaError;
use nalgebra::DMatrix;

/// A probability value in [0, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Creates a new probability value.
    ///
    /// # Arguments
    ///
    /// * `value` - The probability (must be in [0, 1])
    ///
    /// # Returns
    ///
    /// * `Result<Probability, CopulaError>` - The validated probability or an error
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

/// A correlation coefficient in [-1, 1].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Correlation(f64);

impl Correlation {
    /// Creates a new correlation coefficient.
    ///
    /// # Arguments
    ///
    /// * `value` - The correlation (must be in [-1, 1])
    ///
    /// # Returns
    ///
    /// * `Result<Correlation, CopulaError>` - The validated correlation or an error
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

/// A correlation matrix for multivariate copulas.
///
/// Must be symmetric and positive definite with 1s on the diagonal.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationMatrix {
    matrix: DMatrix<f64>,
}

impl CorrelationMatrix {
    /// Creates a new correlation matrix with validation.
    ///
    /// # Arguments
    ///
    /// * `matrix` - The correlation matrix (must be symmetric, positive definite, with 1s on diagonal)
    ///
    /// # Returns
    ///
    /// * `Result<CorrelationMatrix, CopulaError>` - The validated matrix or an error
    #[verified_engine::verified]
    pub fn new(matrix: DMatrix<f64>) -> Result<Self, CopulaError> {
        // Check square
        if matrix.nrows() != matrix.ncols() {
            return Err(crate::error::CopulaError::Math(
                math_commons::error::MathError::DimensionMismatch {
                    expected: math_commons::math_kernel::types::Dimension(matrix.nrows()),
                    actual: math_commons::math_kernel::types::Dimension(matrix.ncols()),
                },
            ));
        }

        // Check symmetric
        for i in 0..matrix.nrows() {
            for j in i + 1..matrix.ncols() {
                if (matrix[(i, j)] - matrix[(j, i)]).abs() > math_commons::registry::TOLERANCE_HIGH
                {
                    return Err(CopulaError::NotSymmetric);
                }
            }
        }

        // Check diagonal is 1
        for i in 0..matrix.nrows() {
            if (matrix[(i, i)] - 1.0).abs() > math_commons::registry::TOLERANCE_HIGH {
                return Err(crate::error::CopulaError::Math(
                    math_commons::error::MathError::NumericalError {
                        reason: format!("Diagonal element {} is not 1.0", i),
                    },
                ));
            }
        }

        // Check all elements in [-1, 1]
        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if matrix[(i, j)].abs() > 1.0 + math_commons::registry::TOLERANCE_HIGH {
                    return Err(CopulaError::InvalidCorrelation {
                        value: matrix[(i, j)],
                    });
                }
            }
        }

        Ok(Self { matrix })
    }

    /// Creates a bivariate correlation matrix from a single correlation coefficient.
    ///
    /// # Arguments
    ///
    /// * `rho` - The correlation between the two variables
    ///
    /// # Returns
    ///
    /// A 2x2 correlation matrix [[1, ρ], [ρ, 1]]
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::{CorrelationMatrix, Correlation};
    ///
    /// let rho = Correlation::new(-0.3).unwrap();
    /// let matrix = CorrelationMatrix::bivariate(rho).unwrap();
    /// ```
    #[verified_engine::verified]
    pub fn bivariate(rho: Correlation) -> Result<Self, CopulaError> {
        let r = rho.value();
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, r, r, 1.0]);
        Self::new(matrix)
    }

    /// Returns the underlying matrix.
    #[verified_engine::verified]
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Returns the dimension of the matrix.
    #[verified_engine::verified]
    pub fn dimension(&self) -> usize {
        self.matrix.nrows()
    }

    /// Returns the correlation between variables i and j.
    #[verified_engine::verified]
    pub fn get_correlation(&self, i: usize, j: usize) -> Option<f64> {
        if i < self.dimension() && j < self.dimension() {
            Some(self.matrix[(i, j)])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_probability_valid() {
        let p = Probability::new(0.5).unwrap();
        assert_eq!(p.value(), 0.5);

        let p_zero = Probability::new(0.0).unwrap();
        assert_eq!(p_zero.value(), 0.0);

        let p_one = Probability::new(1.0).unwrap();
        assert_eq!(p_one.value(), 1.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_probability_invalid() {
        assert!(Probability::new(-0.1).is_err());
        assert!(Probability::new(1.1).is_err());
        assert!(Probability::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_valid() {
        let rho = Correlation::new(0.5).unwrap();
        assert_eq!(rho.value(), 0.5);

        let rho_neg = Correlation::new(-0.8).unwrap();
        assert_eq!(rho_neg.value(), -0.8);
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_invalid() {
        assert!(Correlation::new(-1.1).is_err());
        assert!(Correlation::new(1.1).is_err());
        assert!(Correlation::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_bivariate() {
        let rho = Correlation::new(0.6).unwrap();
        let matrix = CorrelationMatrix::bivariate(rho).unwrap();

        assert_eq!(matrix.dimension(), 2);
        assert_eq!(matrix.get_correlation(0, 1), Some(0.6));
        assert_eq!(matrix.get_correlation(1, 0), Some(0.6));
        assert_eq!(matrix.get_correlation(0, 0), Some(1.0));
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_invalid_diagonal() {
        let matrix = DMatrix::from_row_slice(2, 2, &[0.5, 0.3, 0.3, 1.0]);
        assert!(CorrelationMatrix::new(matrix).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_not_symmetric() {
        let matrix = DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.5, 1.0]);
        assert!(CorrelationMatrix::new(matrix).is_err());
    }
}
