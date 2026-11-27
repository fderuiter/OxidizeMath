//! Core types for copula operations.

use crate::error::CopulaError;
use engine_linalg::{SolverWorkspace, cholesky_semidefinite, cholesky_semidefinite_workspace_len};
use math_commons::error::MathError;
use math_commons::math_kernel::types::Dimension;
use math_commons::registry::TOLERANCE_HIGH;
use math_memory::{ArenaLayout, OwnedByteArena};
use nalgebra::DMatrix;

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
fn validate_and_fill_sym_row(
    matrix: &DMatrix<f64>,
    sym_row: &mut [f64],
) -> Result<(), CopulaError> {
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

    sym_row.fill(0.0);
    for i in 0..n {
        sym_row[math_commons::math_kernel::types::flatten_2d_index(i, i, n)] = 1.0;
        for j in (i + 1)..n {
            if (matrix[(i, j)] - matrix[(j, i)]).abs() > TOLERANCE_HIGH {
                return Err(CopulaError::NotSymmetric);
            }
            let avg = (0.5 * (matrix[(i, j)] + matrix[(j, i)])).clamp(-1.0, 1.0);
            sym_row[math_commons::math_kernel::types::flatten_2d_index(j, i, n)] = avg;
            sym_row[math_commons::math_kernel::types::flatten_2d_index(i, j, n)] = avg;
        }
    }
    Ok(())
}

impl CorrelationMatrix {
    /// Returns conservative byte scratch capacity required by [`Self::new_with_workspace`] for an `n x n` matrix.
    #[must_use]
    #[verified_engine::verified]
    pub fn workspace_bytes(n: usize) -> usize {
        let nn = n.saturating_mul(n);
        let mut layout = ArenaLayout::new();
        let _ = layout.reserve::<f64>(
            nn.saturating_mul(2)
                .saturating_add(cholesky_semidefinite_workspace_len(n)),
        );
        layout.total_bytes().saturating_add(64)
    }

    /// Validates `matrix` and factors its positive-semidefinite Cholesky root into `cholesky_buf` using `workspace` (`0` heap allocations).
    #[verified_engine::verified]
    pub fn new_with_workspace(
        matrix: DMatrix<f64>,
        mut cholesky_buf: DMatrix<f64>,
        workspace: &mut (impl SolverWorkspace + ?Sized),
    ) -> Result<Self, CopulaError> {
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
        if cholesky_buf.shape() != (n, n) {
            return Err(CopulaError::Math(MathError::DimensionMismatch {
                expected: Dimension(n),
                actual: Dimension(cholesky_buf.nrows()),
            }));
        }
        let mut cur = workspace.cursor();
        let sym_row = cur
            .alloc_slice::<f64>(n.saturating_mul(n))
            .map_err(|_| CopulaError::NotPositiveDefinite)?;
        validate_and_fill_sym_row(&matrix, sym_row)?;
        let lower = cur
            .alloc_slice::<f64>(n.saturating_mul(n))
            .map_err(|_| CopulaError::NotPositiveDefinite)?;
        let mut sub = cur.into_arena();
        cholesky_semidefinite(sym_row, lower, n, TOLERANCE_HIGH, TOLERANCE_HIGH, &mut sub)
            .map_err(|_| CopulaError::NotPositiveDefinite)?;
        for r in 0..n {
            for c in 0..n {
                cholesky_buf[(r, c)] =
                    lower[math_commons::math_kernel::types::flatten_2d_index(c, r, n)];
            }
        }
        Ok(Self {
            matrix,
            cholesky: cholesky_buf,
        })
    }

    /// Creates a validated symmetric positive-semidefinite correlation matrix and caches its
    /// lower-triangular Cholesky factor.
    #[verified_engine::verified]
    pub fn new(matrix: DMatrix<f64>) -> Result<Self, CopulaError> {
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
        let cholesky_buf = DMatrix::zeros(n, n);
        let mut ws = OwnedByteArena::with_capacity(Self::workspace_bytes(n));
        Self::new_with_workspace(matrix, cholesky_buf, &mut ws)
    }

    /// Creates an equicorrelated `dimension x dimension` correlation matrix with off-diagonal `rho`.
    #[verified_engine::verified]
    pub fn equicorrelated(dimension: usize, rho: Correlation) -> Result<Self, CopulaError> {
        if dimension == 0 {
            return Err(CopulaError::Math(MathError::DimensionMismatch {
                expected: Dimension(1),
                actual: Dimension(0),
            }));
        }
        let r = rho.value();
        let matrix = DMatrix::from_fn(dimension, dimension, |i, j| if i == j { 1.0 } else { r });
        Self::new(matrix)
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
        assert_eq!(matrix.get_correlation(0, 0), Some(1.0));
        assert!((matrix.cholesky_factor()[(1, 1)] - 0.8).abs() < 1e-12);

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

    #[test]
    #[verified_engine::verified]
    fn test_correlation_matrix_rejects_indefinite_3x3_min_eigenvalue_minus_0_8() {
        let indefinite_frustrated =
            DMatrix::from_row_slice(3, 3, &[1.0, 0.9, -0.9, 0.9, 1.0, 0.9, -0.9, 0.9, 1.0]);
        assert_eq!(
            CorrelationMatrix::new(indefinite_frustrated),
            Err(CopulaError::NotPositiveDefinite)
        );

        let indefinite_equicorr =
            DMatrix::from_row_slice(3, 3, &[1.0, -0.9, -0.9, -0.9, 1.0, -0.9, -0.9, -0.9, 1.0]);
        assert_eq!(
            CorrelationMatrix::new(indefinite_equicorr),
            Err(CopulaError::NotPositiveDefinite)
        );

        let valid3 = DMatrix::from_row_slice(3, 3, &[1.0, 0.4, 0.2, 0.4, 1.0, 0.3, 0.2, 0.3, 1.0]);
        let chol_buf = DMatrix::zeros(3, 3);
        let mut ws = OwnedByteArena::with_capacity(CorrelationMatrix::workspace_bytes(3));
        let (cm, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            CorrelationMatrix::new_with_workspace(valid3, chol_buf, &mut ws).unwrap()
        });
        assert_eq!(cm.dimension(), 3);
    }
}
