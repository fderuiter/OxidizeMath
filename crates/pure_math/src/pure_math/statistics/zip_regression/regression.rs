//! Zero-Inflated Poisson regression implementation with EM covariate estimation.

use super::core::{Count, ZipParams};
use super::distribution::ZipDistribution;
use super::em_solver::{ZipFitResult, fit_zip_em};
use super::link_functions::{LogLink, LogitLink};
use crate::error::ZipError;
use nalgebra::{DMatrix, DVector};

/// Zero-Inflated Poisson regression model $\log \lambda_i = X_i \beta$, $\mathrm{logit}\,\rho_i = Z_i \alpha$.
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::statistics::zip_regression::{ZipRegression, Count};
/// use math_explorer::nalgebra::DMatrix;
///
/// let counts = vec![
///     Count::new(0), Count::new(0), Count::new(2),
///     Count::new(0), Count::new(3), Count::new(1),
/// ];
/// let x = DMatrix::from_element(6, 1, 1.0);
/// let z = DMatrix::from_element(6, 1, 1.0);
///
/// let model = ZipRegression::new(counts, x, z).unwrap();
/// let fitted = model.fit_default().unwrap();
/// assert!(fitted.log_likelihood.is_finite());
/// ```
pub struct ZipRegression {
    counts: Vec<Count>,
    x_matrix: DMatrix<f64>,
    z_matrix: DMatrix<f64>,
}

impl ZipRegression {
    /// Creates a new ZIP regression model validating sample count and design matrix row dimensions.
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
        Ok(Self {
            counts,
            x_matrix,
            z_matrix,
        })
    }

    /// Predicts [`ZipParams`] $(\rho, \lambda)$ for predictor rows `x_row` and `z_row`.
    #[verified_engine::verified]
    pub fn predict(
        x_row: &DVector<f64>,
        z_row: &DVector<f64>,
        beta: &DVector<f64>,
        alpha: &DVector<f64>,
    ) -> Result<ZipParams, ZipError> {
        let eta = x_row.dot(beta);
        let gamma = z_row.dot(alpha);
        let lambda = LogLink::link(eta);
        let rho = LogitLink::link(gamma);
        ZipParams::from_values(rho, lambda)
    }

    /// Computes the observed-data log-likelihood $\sum_{i=1}^n \ln P(Y_i = y_i \mid X_i \beta, Z_i \alpha)$.
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
            let Ok(params) = ZipParams::from_values(rho, lambda) else {
                return f64::NEG_INFINITY;
            };
            let dist = ZipDistribution::new(params);
            ll += dist.ln_pmf(self.counts[i]);
            if !ll.is_finite() {
                return f64::NEG_INFINITY;
            }
        }
        ll
    }

    /// Fits $(\beta, \alpha)$ via Expectation-Maximization with damped IRLS M-steps.
    ///
    /// # Errors
    /// Returns [`ZipError::ConvergenceFailed`] if EM fails to converge within `max_iter` steps.
    #[verified_engine::verified]
    pub fn fit(&self, max_iter: usize, tol: f64) -> Result<ZipFitResult, ZipError> {
        fit_zip_em(
            self,
            &self.counts,
            &self.x_matrix,
            &self.z_matrix,
            max_iter,
            tol,
        )
    }

    /// Fits $(\beta, \alpha)$ with default budget (`max_iter = 200`, `tol = 1e-7`).
    #[verified_engine::verified]
    pub fn fit_default(&self) -> Result<ZipFitResult, ZipError> {
        self.fit(200, 1e-7)
    }

    /// Returns the number of observations.
    #[verified_engine::verified]
    pub fn n_obs(&self) -> usize {
        self.counts.len()
    }

    /// Returns the observed counts slice.
    #[verified_engine::verified]
    pub fn counts(&self) -> &[Count] {
        &self.counts
    }

    /// Returns the count-model design matrix $X$.
    #[verified_engine::verified]
    pub fn x_matrix(&self) -> &DMatrix<f64> {
        &self.x_matrix
    }

    /// Returns the zero-inflation design matrix $Z$.
    #[verified_engine::verified]
    pub fn z_matrix(&self) -> &DMatrix<f64> {
        &self.z_matrix
    }
}

/// Intercept-only Zero-Inflated Poisson estimator combining method-of-moments initialization
/// with scalar EM refinement across the full `[0, 1)` zero-inflation range (without artificial `0.95` clamping).
#[verified_engine::verified]
pub fn simple_zip_fit(counts: &[Count]) -> Result<ZipParams, ZipError> {
    if counts.is_empty() {
        return Err(ZipError::InsufficientData {
            required: 1,
            actual: 0,
        });
    }

    let n = counts.len() as f64;
    let n_zeros = counts.iter().filter(|&&c| c.value() == 0).count() as f64;
    let prop_zeros = n_zeros / n;
    let mean: f64 = counts.iter().map(|c| c.as_f64()).sum::<f64>() / n;
    let variance: f64 = counts
        .iter()
        .map(|c| {
            let d = c.as_f64() - mean;
            d * d
        })
        .sum::<f64>()
        / n;

    if mean <= 0.0 || variance <= 0.0 {
        return ZipParams::from_values(0.99, 0.01);
    }

    let mut lambda = if variance > mean {
        (variance / mean - 1.0 + mean).max(mean)
    } else {
        mean
    };
    let p0_init = (-lambda).exp();
    let mut rho = if prop_zeros > p0_init && p0_init < 1.0 {
        ((prop_zeros - p0_init) / (1.0 - p0_init)).clamp(0.0, 0.999_999)
    } else {
        0.0
    };

    for _ in 0..48 {
        let exp_neg_lam = (-lambda).exp();
        let denom = rho + (1.0 - rho) * exp_neg_lam;
        let w0 = if denom > 0.0 { rho / denom } else { 0.0 };
        let next_rho = (prop_zeros * w0).clamp(0.0, 0.999_999);
        let active_frac = (1.0 - prop_zeros * w0).max(1e-9);
        let next_lam = (mean / active_frac).max(1e-6);
        if (next_rho - rho).abs() < 1e-11 && (next_lam - lambda).abs() < 1e-11 {
            rho = next_rho;
            lambda = next_lam;
            break;
        }
        rho = next_rho;
        lambda = next_lam;
    }

    ZipParams::from_values(rho, lambda)
}

#[cfg(test)]
mod tests {
    #![cfg_attr(any(), verified_engine::verified(opt_out = "tests"))]

    use super::*;
    use approx::assert_relative_eq;
    use oxidize_core::rng::OxidizeRng;

    #[test]
    fn test_zip_regression_creation_and_predict() {
        let counts = vec![Count::new(0), Count::new(1), Count::new(2)];
        let x = DMatrix::from_element(3, 1, 1.0);
        let z = DMatrix::from_element(3, 1, 1.0);
        let model = ZipRegression::new(counts, x, z).unwrap();
        let ll = model.log_likelihood(
            &DVector::from_vec(vec![0.5]),
            &DVector::from_vec(vec![-1.0]),
        );
        assert!(ll.is_finite() && ll < 0.0);

        let x_bad = DMatrix::from_element(2, 1, 1.0);
        let z_ok = DMatrix::from_element(3, 1, 1.0);
        assert!(
            ZipRegression::new(
                vec![Count::new(0), Count::new(1), Count::new(2)],
                x_bad,
                z_ok
            )
            .is_err()
        );

        let params = ZipRegression::predict(
            &DVector::from_vec(vec![1.0, 2.0]),
            &DVector::from_vec(vec![1.0, 0.5]),
            &DVector::from_vec(vec![0.5, 0.3]),
            &DVector::from_vec(vec![-1.0, 0.2]),
        )
        .unwrap();
        assert_relative_eq!(params.lambda.value(), 1.1_f64.exp(), epsilon = 1e-10);
        assert_relative_eq!(params.rho.value(), LogitLink::link(-0.9), epsilon = 1e-10);
    }

    #[test]
    fn test_simple_zip_fit_high_zero_inflation_unclamped() {
        // 97 zeros + three active counts of 5 -> true rho ≈ 0.97 (> legacy 0.95 clamp!)
        let mut counts = vec![Count::new(0); 97];
        counts.extend([Count::new(5), Count::new(5), Count::new(5)]);
        let params = simple_zip_fit(&counts).unwrap();
        assert!(
            params.rho.value() > 0.96,
            "expected rho > 0.96 without 0.95 clamp, got {}",
            params.rho.value()
        );
        assert!((params.lambda.value() - 5.0).abs() < 0.2);

        let all_zeros = vec![Count::new(0), Count::new(0), Count::new(0)];
        assert!(simple_zip_fit(&all_zeros).unwrap().rho.value() > 0.5);
    }

    #[test]
    fn test_zip_em_covariate_recovery_and_non_convergence_error() {
        let mut rng = OxidizeRng::from_seed(202604);
        let n = 300;
        let true_beta = DVector::from_vec(vec![0.8, 0.6]);
        let true_alpha = DVector::from_vec(vec![-0.5, 1.1]);

        let mut x_data = Vec::with_capacity(n * 2);
        let mut z_data = Vec::with_capacity(n * 2);
        let mut counts = Vec::with_capacity(n);
        for i in 0..n {
            let xi = -1.0 + 2.0 * (i as f64) / ((n - 1) as f64);
            let zi = if i % 2 == 0 { -0.8 } else { 0.8 };
            x_data.extend_from_slice(&[1.0, xi]);
            z_data.extend_from_slice(&[1.0, zi]);
            let p = ZipRegression::predict(
                &DVector::from_vec(vec![1.0, xi]),
                &DVector::from_vec(vec![1.0, zi]),
                &true_beta,
                &true_alpha,
            )
            .unwrap();
            counts.push(ZipDistribution::new(p).sample(&mut rng));
        }
        let x_mat = DMatrix::from_row_slice(n, 2, &x_data);
        let z_mat = DMatrix::from_row_slice(n, 2, &z_data);
        let model = ZipRegression::new(counts, x_mat, z_mat).unwrap();

        let fit = model.fit_default().unwrap();
        assert!((fit.beta[0] - 0.8).abs() < 0.25);
        assert!((fit.beta[1] - 0.6).abs() < 0.25);
        assert!((fit.alpha[1] - 1.1).abs() < 0.45);

        // Insufficient iteration budget (1 step with strict tol) returns ConvergenceFailed
        let non_conv = model.fit(1, 1e-15);
        assert!(matches!(non_conv, Err(ZipError::ConvergenceFailed { .. })));
    }
}
