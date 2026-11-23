//! Zero-Inflated Poisson regression implementation.

use super::core::{Count, ZipParams};
use super::distribution::ZipDistribution;
use super::link_functions::{LogLink, LogitLink};
use crate::error::ZipError;
use nalgebra::{DMatrix, DVector};

/// ZIP regression model.
///
/// This model uses two sets of predictors:
/// - X predictors for the Poisson rate: λ = exp(X'β)
/// - Z predictors for the zero-inflation: ρ = expit(Z'α)
///
/// # Mathematical Framework
///
/// The model specifies:
/// - log(λᵢ) = β₀ + β₁x₁ᵢ + ... + βₚxₚᵢ
/// - logit(ρᵢ) = α₀ + α₁z₁ᵢ + ... + αᵧzᵧᵢ
///
/// where:
/// - β is the coefficient vector for the count model
/// - α is the coefficient vector for the zero-inflation model
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::zip_regression::{ZipRegression, Count};
/// use math_explorer::nalgebra::{DMatrix, DVector};
///
/// // Simple example with intercept-only models
/// let counts = vec![Count::new(0), Count::new(1), Count::new(0), Count::new(2)];
///
/// // Intercept-only design matrices
/// let x = DMatrix::from_element(4, 1, 1.0);  // Count model
/// let z = DMatrix::from_element(4, 1, 1.0);  // Zero-inflation model
///
/// let model = ZipRegression::new(counts, x, z).unwrap();
/// ```
pub struct ZipRegression {
    /// The observed counts.
    pub(super) counts: Vec<Count>,
    /// Design matrix for the count model (X).
    pub(super) x_matrix: DMatrix<f64>,
    /// Design matrix for the zero-inflation model (Z).
    pub(super) z_matrix: DMatrix<f64>,
}

impl ZipRegression {
    /// Creates a new ZIP regression model.
    ///
    /// # Arguments
    ///
    /// * `counts` - Vector of observed counts
    /// * `x_matrix` - Design matrix for the count model (n_samples × p)
    /// * `z_matrix` - Design matrix for the zero-inflation model (n_samples × q)
    ///
    /// # Returns
    ///
    /// * `Result<ZipRegression, ZipError>` - The model or an error
    ///
    /// # Errors
    ///
    /// Returns `ZipError::InsufficientData` if there are fewer than 2 counts.
    /// Returns `ZipError::InvalidDimensions` if the number of rows in `x_matrix` or `z_matrix` does not match the number of counts.
    #[verified_engine::verified]
    pub fn new(
        counts: Vec<Count>,
        x_matrix: DMatrix<f64>,
        z_matrix: DMatrix<f64>,
    ) -> Result<Self, ZipError> {
        let n = counts.len();

        if n < 2 {
            return Err(ZipError::InsufficientData {
                required: 2,
                actual: n,
            });
        }

        if x_matrix.nrows() != n {
            return Err(ZipError::InvalidDimensions {
                expected: format!("X: {} rows", n),
                actual: format!("X: {} rows", x_matrix.nrows()),
            });
        }

        if z_matrix.nrows() != n {
            return Err(ZipError::InvalidDimensions {
                expected: format!("Z: {} rows", n),
                actual: format!("Z: {} rows", z_matrix.nrows()),
            });
        }

        if x_matrix.ncols() == 0 {
            return Err(ZipError::InvalidDimensions {
                expected: "X: >= 1 columns".to_string(),
                actual: "X: 0 columns".to_string(),
            });
        }

        if z_matrix.ncols() == 0 {
            return Err(ZipError::InvalidDimensions {
                expected: "Z: >= 1 columns".to_string(),
                actual: "Z: 0 columns".to_string(),
            });
        }

        Ok(Self {
            counts,
            x_matrix,
            z_matrix,
        })
    }

    /// Predicts ZIP parameters for given predictor values.
    ///
    /// # Arguments
    ///
    /// * `x_row` - Predictor values for the count model
    /// * `z_row` - Predictor values for the zero-inflation model
    /// * `beta` - Coefficients for the count model
    /// * `alpha` - Coefficients for the zero-inflation model
    ///
    /// # Returns
    ///
    /// * `Result<ZipParams, ZipError>` - The predicted parameters
    ///
    /// # Errors
    ///
    /// Returns a `ZipError` if the calculated parameters (after link functions) are invalid.
    #[verified_engine::verified]
    pub fn predict(
        x_row: &DVector<f64>,
        z_row: &DVector<f64>,
        beta: &DVector<f64>,
        alpha: &DVector<f64>,
    ) -> Result<ZipParams, ZipError> {
        // Compute linear predictors
        let eta = x_row.dot(beta);
        let gamma = z_row.dot(alpha);

        // Apply link functions
        let lambda = LogLink::link(eta);
        let rho = LogitLink::link(gamma);

        ZipParams::from_values(rho, lambda)
    }

    /// Computes the log-likelihood of the model given parameters.
    ///
    /// # Arguments
    ///
    /// * `beta` - Coefficients for the count model
    /// * `alpha` - Coefficients for the zero-inflation model
    ///
    /// # Returns
    ///
    /// The log-likelihood value
    #[verified_engine::verified]
    pub fn log_likelihood(&self, beta: &DVector<f64>, alpha: &DVector<f64>) -> f64 {
        let mut ll = 0.0;
        let beta_t = beta.transpose();
        let alpha_t = alpha.transpose();

        for i in 0..self.counts.len() {
            let eta = self.x_matrix.row(i).dot(&beta_t);
            let gamma = self.z_matrix.row(i).dot(&alpha_t);

            let lambda = LogLink::link(eta);
            let rho = LogitLink::link(gamma);

            let params = match ZipParams::from_values(rho, lambda) {
                Ok(p) => p,
                Err(_) => return f64::NEG_INFINITY,
            };

            let dist = ZipDistribution::new(params);
            ll += dist.ln_pmf(self.counts[i]);
        }

        ll
    }

    /// Returns the number of observations.
    #[verified_engine::verified]
    pub fn n_obs(&self) -> usize {
        self.counts.len()
    }

    /// Returns the observed counts.
    #[verified_engine::verified]
    pub fn counts(&self) -> &[Count] {
        &self.counts
    }
}

/// Simple ZIP regression with method of moments initialization.
///
/// This function provides a simplified interface for ZIP regression using
/// method of moments to initialize parameters.
///
/// # Arguments
///
/// * `counts` - Vector of observed counts
///
/// # Returns
///
/// * `Result<ZipParams, ZipError>` - Estimated parameters using method of moments
///
/// # Errors
///
/// Returns `ZipError::InsufficientData` if the `counts` slice is empty.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::zip_regression::{simple_zip_fit, Count};
///
/// let counts = vec![Count::new(0), Count::new(0), Count::new(1), Count::new(2), Count::new(0)];
/// let params = simple_zip_fit(&counts).unwrap();
/// ```
#[verified_engine::verified]
pub fn simple_zip_fit(counts: &[Count]) -> Result<ZipParams, ZipError> {
    if counts.is_empty() {
        return Err(ZipError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    // Count zeros
    let n = counts.len() as f64;
    let n_zeros = counts.iter().filter(|&&c| c.value() == 0).count() as f64;
    let prop_zeros = n_zeros / n;

    // Compute sample mean and variance
    let mean: f64 = counts.iter().map(|c| c.as_f64()).sum::<f64>() / n;
    let variance: f64 = counts
        .iter()
        .map(|c| {
            let diff = c.as_f64() - mean;
            diff * diff
        })
        .sum::<f64>()
        / n;

    // Method of moments estimates
    // For ZIP: E[Y] = (1-ρ)λ, Var[Y] = (1-ρ)λ(1 + ρλ)
    // If we assume mean > 0, we can solve:

    if mean <= 0.0 || variance <= 0.0 {
        // Degenerate case: all zeros
        return ZipParams::from_values(0.99, 0.01);
    }

    // Exact Lambert (1992) method-of-moments when overdispersed (variance > mean):
    // E[Y] = (1-ρ)λ, Var[Y] = (1-ρ)λ(1 + ρλ) => λ_MoM = mean + variance/mean - 1, ρ_MoM = 1 - mean/λ_MoM
    let (rho_est, lambda_adj) = if variance > mean {
        let lambda_mom = (mean + variance / mean - 1.0).max(mean);
        let rho_mom = (1.0 - mean / lambda_mom).clamp(0.0, 1.0 - 1.0e-12);
        let p0 = (-lambda_mom).exp();
        let rho_zero = if prop_zeros > p0 {
            ((prop_zeros - p0) / (1.0 - p0)).clamp(0.0, 1.0 - 1.0e-12)
        } else {
            rho_mom
        };
        let rho_combined = rho_mom.max(rho_zero);
        let lambda_combined = (mean / (1.0 - rho_combined)).max(0.01);
        (rho_combined, lambda_combined)
    } else {
        let poisson_zero_prob = (-mean).exp();
        let rho_zero = if prop_zeros > poisson_zero_prob {
            ((prop_zeros - poisson_zero_prob) / (1.0 - poisson_zero_prob)).clamp(0.0, 1.0 - 1.0e-12)
        } else {
            0.0
        };
        let lambda_zero = (mean / (1.0 - rho_zero)).max(0.01);
        (rho_zero, lambda_zero)
    };

    // Ensure valid parameter ranges without artificial upper clamping at 0.95
    let rho_final = rho_est.clamp(0.0, 1.0 - 1.0e-12);
    let lambda_final = lambda_adj.max(0.01);

    ZipParams::from_values(rho_final, lambda_final)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_zip_regression_creation() {
        let counts = vec![Count::new(0), Count::new(1), Count::new(2)];
        let x = DMatrix::from_element(3, 1, 1.0);
        let z = DMatrix::from_element(3, 1, 1.0);

        let model = ZipRegression::new(counts, x, z);
        assert!(model.is_ok());
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_regression_dimension_mismatch() {
        let counts = vec![Count::new(0), Count::new(1), Count::new(2)];
        let x = DMatrix::from_element(2, 1, 1.0); // Wrong size
        let z = DMatrix::from_element(3, 1, 1.0);

        let model = ZipRegression::new(counts, x, z);
        assert!(model.is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_zip_predict() {
        let x_row = DVector::from_vec(vec![1.0, 2.0]);
        let z_row = DVector::from_vec(vec![1.0, 0.5]);
        let beta = DVector::from_vec(vec![0.5, 0.3]);
        let alpha = DVector::from_vec(vec![-1.0, 0.2]);

        let params = ZipRegression::predict(&x_row, &z_row, &beta, &alpha).unwrap();

        // eta = 1.0*0.5 + 2.0*0.3 = 1.1
        // lambda = exp(1.1)
        let expected_lambda = 1.1_f64.exp();
        assert_relative_eq!(
            params.lambda.value(),
            expected_lambda,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );

        // gamma = 1.0*(-1.0) + 0.5*0.2 = -0.9
        // rho = expit(-0.9)
        let expected_rho = LogitLink::link(-0.9);
        assert_relative_eq!(
            params.rho.value(),
            expected_rho,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_simple_zip_fit() {
        // Simulate data with known parameters
        let counts = vec![
            Count::new(0),
            Count::new(0),
            Count::new(0),
            Count::new(1),
            Count::new(2),
            Count::new(1),
        ];

        let params = simple_zip_fit(&counts).unwrap();

        // Should estimate reasonable parameters
        assert!(params.rho.value() >= 0.0 && params.rho.value() <= 1.0);
        assert!(params.lambda.value() > 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_simple_zip_fit_all_zeros() {
        let counts = vec![Count::new(0), Count::new(0), Count::new(0)];
        let params = simple_zip_fit(&counts).unwrap();

        // Should estimate high zero-inflation
        assert!(params.rho.value() > 0.5);
    }

    #[test]
    #[verified_engine::verified]
    fn test_log_likelihood() {
        let counts = vec![Count::new(0), Count::new(1), Count::new(2)];
        let x = DMatrix::from_element(3, 1, 1.0);
        let z = DMatrix::from_element(3, 1, 1.0);

        let model = ZipRegression::new(counts, x, z).unwrap();

        let beta = DVector::from_vec(vec![0.5]);
        let alpha = DVector::from_vec(vec![-1.0]);

        let ll = model.log_likelihood(&beta, &alpha);

        // Log-likelihood should be finite and negative
        assert!(ll.is_finite());
        assert!(ll < 0.0);
    }
}
