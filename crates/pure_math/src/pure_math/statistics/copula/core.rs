//! Core types for copula operations.

use crate::error::CopulaError;
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use math_commons::registry::TOLERANCE_HIGH;
use nalgebra::{DMatrix, SymmetricEigen};

/// Pivot threshold below which a positive-semidefinite Schur complement diagonal is snapped to zero.
const PSD_PIVOT_EPS: f64 = 1e-12;

/// Minimum diagonal threshold for accepting unpivoted strict Cholesky without spectral verification.
const STRICT_CHOLESKY_MIN_DIAG: f64 = 1e-7;

/// A probability value in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Probability(f64);

impl Probability {
    /// Creates a new probability value in `[0, 1]`.
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
    /// Creates a new correlation coefficient in `[-1, 1]`.
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

/// A validated symmetric positive-semidefinite correlation matrix with cached lower-triangular factor.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrelationMatrix {
    matrix: DMatrix<f64>,
    cholesky: DMatrix<f64>,
}

#[verified_engine::verified]
fn validate_entries_and_symmetry(matrix: &DMatrix<f64>) -> Result<DMatrix<f64>, CopulaError> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(n),
            actual: Dimension(matrix.ncols()),
        }));
    }
    if n == 0 {
        return Err(CopulaError::Math(MathError::DimensionMismatch {
            expected: Dimension(1),
            actual: Dimension(0),
        }));
    }

    for i in 0..n {
        for j in 0..n {
            let v = matrix[(i, j)];
            if !v.is_finite() || v.abs() > 1.0 + TOLERANCE_HIGH {
                return Err(CopulaError::InvalidCorrelation { value: v });
            }
        }
        if (matrix[(i, i)] - 1.0).abs() > TOLERANCE_HIGH {
            return Err(CopulaError::InvalidCorrelation {
                value: matrix[(i, i)],
            });
        }
    }

    let mut sym = DMatrix::zeros(n, n);
    for i in 0..n {
        sym[(i, i)] = 1.0;
        for j in (i + 1)..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > TOLERANCE_HIGH {
                return Err(CopulaError::NotSymmetric);
            }
            let avg = (0.5 * (matrix[(i, j)] + matrix[(j, i)])).clamp(-1.0, 1.0);
            sym[(i, j)] = avg;
            sym[(j, i)] = avg;
        }
    }
    Ok(sym)
}

fn semidefinite_cholesky(sym: &DMatrix<f64>) -> DMatrix<f64> {
    let n = sym.nrows();
    let mut lower = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..i {
            let l_jj = lower[(j, j)];
            if l_jj > 0.0 {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += lower[(i, k)] * lower[(j, k)];
                }
                lower[(i, j)] = (sym[(i, j)] - sum) / l_jj;
            }
        }
        let mut diag_sum = 0.0;
        for j in 0..i {
            let v = lower[(i, j)];
            diag_sum += v * v;
        }
        let rem = sym[(i, i)] - diag_sum;
        if rem > PSD_PIVOT_EPS {
            lower[(i, i)] = rem.sqrt();
        }
    }
    lower
}

fn compute_psd_factor(sym: DMatrix<f64>) -> Result<DMatrix<f64>, CopulaError> {
    if let Some(chol) = sym.clone().cholesky() {
        let lower = chol.unpack();
        let min_diag = (0..lower.nrows())
            .map(|i| lower[(i, i)])
            .fold(f64::INFINITY, f64::min);
        if min_diag > STRICT_CHOLESKY_MIN_DIAG {
            return Ok(lower);
        }
    }

    let eig = SymmetricEigen::new(sym.clone());
    for &ev in eig.eigenvalues.iter() {
        if !ev.is_finite() || ev < -TOLERANCE_HIGH {
            return Err(CopulaError::NotPositiveDefinite);
        }
    }
    Ok(semidefinite_cholesky(&sym))
}

impl CorrelationMatrix {
    /// Creates a validated symmetric positive-semidefinite correlation matrix and caches its
    /// lower-triangular Cholesky factor.
    #[verified_engine::verified]
    pub fn new(matrix: DMatrix<f64>) -> Result<Self, CopulaError> {
        let sym = validate_entries_and_symmetry(&matrix)?;
        let cholesky = compute_psd_factor(sym)?;
        Ok(Self { matrix, cholesky })
    }

    /// Creates a bivariate correlation matrix `[[1, rho], [rho, 1]]` from a single correlation.
    ///
    /// # Example
    ///
    /// ```
    /// # use pure_math::__doctest_facade as math_explorer;
    /// use math_explorer::pure_math::statistics::copula::{Correlation, CorrelationMatrix};
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

    /// Returns the underlying correlation matrix.
    #[verified_engine::verified]
    pub fn matrix(&self) -> &DMatrix<f64> {
        &self.matrix
    }

    /// Returns the cached lower-triangular Cholesky factor `L` satisfying `L * L^T = Sigma`.
    #[verified_engine::verified]
    pub fn cholesky_factor(&self) -> &DMatrix<f64> {
        &self.cholesky
    }

    /// Returns the dimension of the correlation matrix.
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_probability_and_correlation_bounds() {
        assert_eq!(Probability::new(0.0).unwrap().value(), 0.0);
        assert_eq!(Probability::new(0.5).unwrap().value(), 0.5);
        assert_eq!(Probability::new(1.0).unwrap().value(), 1.0);
        assert!(Probability::new(-0.1).is_err());
        assert!(Probability::new(1.1).is_err());
        assert!(Probability::new(f64::NAN).is_err());

        assert_eq!(Correlation::new(-1.0).unwrap().value(), -1.0);
        assert_eq!(Correlation::new(1.0).unwrap().value(), 1.0);
        assert!(Correlation::new(-1.1).is_err());
        assert!(Correlation::new(1.1).is_err());
        assert!(Correlation::new(f64::NAN).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_bivariate_and_boundaries() {
        let rho = Correlation::new(0.6).unwrap();
        let matrix = CorrelationMatrix::bivariate(rho).unwrap();
        assert_eq!(matrix.dimension(), 2);
        assert_eq!(matrix.get_correlation(0, 1), Some(0.6));
        assert_eq!(matrix.get_correlation(1, 0), Some(0.6));

        let pos_one = CorrelationMatrix::bivariate(Correlation::new(1.0).unwrap()).unwrap();
        let reconstructed_pos = pos_one.cholesky_factor() * pos_one.cholesky_factor().transpose();
        assert_relative_eq!(reconstructed_pos[(0, 1)], 1.0, epsilon = TOLERANCE_HIGH);

        let neg_one = CorrelationMatrix::bivariate(Correlation::new(-1.0).unwrap()).unwrap();
        let reconstructed_neg = neg_one.cholesky_factor() * neg_one.cholesky_factor().transpose();
        assert_relative_eq!(reconstructed_neg[(0, 1)], -1.0, epsilon = TOLERANCE_HIGH);
    }

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_rejects_indefinite_and_malformed() {
        let indefinite =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.9, 0.9, 0.9, 1.0, -0.9, 0.9, -0.9, 1.0]);
        assert_eq!(
            CorrelationMatrix::new(indefinite),
            Err(CopulaError::NotPositiveDefinite)
        );

        let near_indefinite =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.8, 0.8, 0.8, 1.0, -0.3, 0.8, -0.3, 1.0]);
        assert_eq!(
            CorrelationMatrix::new(near_indefinite),
            Err(CopulaError::NotPositiveDefinite)
        );

        assert!(CorrelationMatrix::new(DMatrix::zeros(0, 0)).is_err());
        assert!(
            CorrelationMatrix::new(DMatrix::from_row_slice(2, 2, &[0.5, 0.3, 0.3, 1.0])).is_err()
        );
        assert_eq!(
            CorrelationMatrix::new(DMatrix::from_row_slice(2, 2, &[1.0, 0.3, 0.5, 1.0])),
            Err(CopulaError::NotSymmetric)
        );
        assert!(
            CorrelationMatrix::new(DMatrix::from_row_slice(
                2,
                2,
                &[1.0, f64::NAN, f64::NAN, 1.0]
            ))
            .is_err()
        );
    }
}
