//! Statistical Regression Models.
//!
//! Implementations of regression analysis techniques.

use nalgebra::{DMatrix, DVector};

/// Multivariate Linear Regression.
///
/// Models the relationship between dependent variable $Y$ and independent variables $\tilde{X}$.
///
/// $$ Y = \tilde{X}B + E $$
///
/// # Arguments
///
/// * `features` ($\tilde{X}$) - Matrix of input features (n_samples x n_features). Should include bias column if needed.
/// * `targets` ($Y$) - Vector of target values (n_samples x 1).
///
/// # Returns
///
/// * `Option<DVector<f64>>` - The coefficient vector $B$ (n_features x 1).
#[verified_engine::verified]
pub fn multivariate_linear_regression(
    features: &DMatrix<f64>,
    targets: &DVector<f64>,
) -> Option<DVector<f64>> {
    let m = features.nrows();
    let n = features.ncols();
    if m == 0
        || n == 0
        || m != targets.len()
        || features.iter().any(|v| !v.is_finite())
        || targets.iter().any(|v| !v.is_finite())
    {
        return None;
    }
    if m >= n {
        let qr = features.clone().qr();
        let r = qr.r();
        let mut max_diag = 0.0_f64;
        for i in 0..n {
            max_diag = max_diag.max(r[(i, i)].abs());
        }
        let r_tol = max_diag * f64::EPSILON * (m.max(n) as f64);
        if max_diag > 0.0 && (0..n).all(|i| r[(i, i)].abs() > r_tol) {
            let qtb = qr.q().tr_mul(targets);
            if let Some(sol) = r.solve_upper_triangular(&qtb) {
                return Some(sol);
            }
        }
    }
    let svd = features.clone().svd(true, true);
    let max_sv = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    if max_sv <= 0.0 {
        return None;
    }
    let tol = max_sv * f64::EPSILON * (m.max(n) as f64);
    svd.solve(targets, tol).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_multivariate_regression() {
        // y = 2x1 + 3x2 + 1
        // x1, x2 -> y
        // 1, 1 -> 6
        // 1, 2 -> 9
        // 2, 1 -> 8
        // Design matrix X (add column of 1s for bias at end or beginning)
        // Let's put bias first: [1, x1, x2]
        let x_data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0];
        let x = DMatrix::from_row_slice(3, 3, &x_data);
        let y = DVector::from_column_slice(&[6.0, 9.0, 8.0]);

        let b = multivariate_linear_regression(&x, &y).unwrap();
        // Expect [1, 2, 3]
        assert!((b[0] - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((b[1] - 2.0).abs() < math_commons::registry::TOLERANCE_FAST);
        assert!((b[2] - 3.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_multivariate_regression_near_collinear_eps_1e8() {
        let eps = 1e-8_f64;
        let x = DMatrix::from_row_slice(
            6,
            3,
            &[
                1.0,
                -1.0,
                -1.0 + eps,
                1.0,
                -0.5,
                -0.5 - eps,
                1.0,
                0.0,
                2.0 * eps,
                1.0,
                0.4,
                0.4 - eps,
                1.0,
                0.8,
                0.8 + eps,
                1.0,
                1.2,
                1.2 - 2.0 * eps,
            ],
        );
        let beta_true = DVector::from_column_slice(&[1.5, -0.8, 0.4]);
        let y = &x * &beta_true;
        let est = multivariate_linear_regression(&x, &y).expect("QR least-squares solution");
        assert!(
            (est[2] - 0.4).abs() <= 2e-8,
            "Expected beta_2 within 2e-8 of 0.4, got {}",
            est[2]
        );
    }
}
