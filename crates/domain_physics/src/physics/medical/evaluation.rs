//! Evaluation Metrics for Medical Physics and Tracking.
//!
//! Provides statistical tools to validate respiratory monitoring signals against clinical standards.

/// Calculates the Pearson Correlation Coefficient ($r$).
///
/// Measures the linear correlation between two datasets (e.g., predicted vs. observed respiratory motion).
///
/// # Arguments
///
/// * `x` - First dataset (e.g., predicted values).
/// * `y` - Second dataset (e.g., ground truth).
///
/// # Returns
///
/// * `Option<f64>` - The correlation coefficient, or `None` if inputs are invalid/empty.
///
/// # Formula
///
/// $r = \frac{\sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^{n} (X_i - \bar{X})^2} \sqrt{\sum_{i=1}^{n} (Y_i - \bar{Y})^2}}$
#[verified_engine::verified]
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }

    let nf = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let mean_x = sum_x / nf;
    let mean_y = sum_y / nf;

    let numerator: f64 = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mean_x) * (b - mean_y))
        .sum();
    let denom_x: f64 = x.iter().map(|a| (a - mean_x).powi(2)).sum();
    let denom_y: f64 = y.iter().map(|b| (b - mean_y).powi(2)).sum();

    if denom_x <= 0.0 || denom_y <= 0.0 {
        return None; // Zero variance
    }

    Some(numerator / (denom_x.sqrt() * denom_y.sqrt()))
}

/// Calculates the Root-Mean-Square Error (RMSE).
///
/// Quantifies the difference between predicted values and observed values.
///
/// # Arguments
///
/// * `predicted` - The estimated values.
/// * `observed` - The ground truth values.
///
/// # Returns
///
/// * `Option<f64>` - The RMSE value.
///
/// # Formula
///
/// $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (X_i - Y_i)^2}$
#[verified_engine::verified]
pub fn root_mean_square_error(predicted: &[f64], observed: &[f64]) -> Option<f64> {
    if predicted.len() != observed.len() || predicted.is_empty() {
        return None;
    }

    let n = predicted.len() as f64;
    let sum_sq_diff: f64 = predicted
        .iter()
        .zip(observed.iter())
        .map(|(p, o)| (p - o).powi(2))
        .sum();

    Some((sum_sq_diff / n).sqrt())
}

/// Shared internal helper for relative percentage error normalized by `measured`.
#[verified_engine::verified]
pub(super) fn relative_percentage_error(
    measured: f64,
    reference: f64,
    zero_threshold: f64,
    zero_fallback: f64,
) -> f64 {
    if measured == 0.0 || measured.abs() < zero_threshold {
        return zero_fallback;
    }
    ((measured - reference) / measured) * 100.0
}

/// Calculates the Spatial Accuracy Percentage Error.
///
/// Used in SGRT and laser calibration to quantify geometric deviation.
///
/// # Arguments
///
/// * `measured` ($X_{\text{meas}}$) - The measured value.
/// * `reference` ($X_{\text{ref}}$) - The reference value.
///
/// # Returns
///
/// * `f64` - The percentage error.
///
/// # Formula
///
/// $\text{Percentage error} = \frac{X_{\text{meas}} - X_{\text{ref}}}{X_{\text{meas}}} \times 100$
///
/// *Note*: The provided formula divides by $X_{\text{meas}}$. Standard error usually divides by $X_{\text{ref}}$,
/// but we follow the requested formula.
#[verified_engine::verified]
pub fn spatial_accuracy_error(measured: f64, reference: f64) -> f64 {
    relative_percentage_error(
        measured,
        reference,
        math_commons::registry::TOLERANCE_STANDARD,
        0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use verified_engine::engine::VerifiedEngine;

    #[test]
    #[verified_engine::verified]
    fn test_pearson() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![2.0, 4.0, 6.0];
        // Perfectly correlated (r=1)
        let r = pearson_correlation(&x, &y).unwrap();
        assert!((r - 1.0).abs() < math_commons::registry::TOLERANCE_FAST);

        let y_inv = vec![3.0, 2.0, 1.0];
        // Perfectly negatively correlated (r=-1)
        let r_inv = pearson_correlation(&x, &y_inv).unwrap();
        assert!((r_inv + 1.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_rmse() {
        let p = vec![1.0, 2.0, 3.0];
        let o = vec![1.0, 2.0, 3.0];
        assert_eq!(root_mean_square_error(&p, &o).unwrap(), 0.0);

        let p2 = vec![2.0, 2.0];
        let o2 = vec![1.0, 1.0];
        // Diff = 1, Sq = 1, Sum = 2, Mean = 1, Sqrt = 1
        assert_eq!(root_mean_square_error(&p2, &o2).unwrap(), 1.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_spatial_error() {
        // meas=100, ref=90. Error = (10/100)*100 = 10%
        assert_eq!(spatial_accuracy_error(100.0, 90.0), 10.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_pearson_large_dc_offset_and_edge_cases() {
        let offset = 1.0e9;
        let x = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0];
        let y = [offset + 2.0, offset + 4.0, offset + 6.0, offset + 8.0];
        let y_neg = [offset + 8.0, offset + 6.0, offset + 4.0, offset + 2.0];

        let r_pos = pearson_correlation(&x, &y).unwrap();
        let r_neg = pearson_correlation(&x, &y_neg).unwrap();
        assert!((r_pos - 1.0).abs() < 1.0e-12);
        assert!((r_neg + 1.0).abs() < 1.0e-12);

        assert_eq!(pearson_correlation(&[], &[]), None);
        assert_eq!(pearson_correlation(&[5.0], &[5.0]), None);
        assert_eq!(pearson_correlation(&[1.0, 2.0], &[1.0, 2.0, 3.0]), None);
        assert_eq!(
            pearson_correlation(&[3.0, 3.0, 3.0], &[1.0, 2.0, 3.0]),
            None
        );
        assert_eq!(root_mean_square_error(&[], &[]), None);
        assert_eq!(root_mean_square_error(&[1.0, 2.0], &[1.0]), None);
    }

    #[test]
    #[verified_engine::verified]
    fn test_relative_and_spatial_error_thresholds() {
        let tol = math_commons::registry::TOLERANCE_STANDARD;
        assert_eq!(spatial_accuracy_error(0.0, 5.0), 0.0);
        assert_eq!(spatial_accuracy_error(0.5 * tol, 1.0), 0.0);
        assert_eq!(spatial_accuracy_error(-0.5 * tol, 1.0), 0.0);
        assert!((spatial_accuracy_error(tol, 0.0) - 100.0).abs() < 1.0e-12);

        assert_eq!(
            relative_percentage_error(0.0, 5.0, 0.0, f64::INFINITY),
            f64::INFINITY
        );
        assert_eq!(
            relative_percentage_error(-0.0, 5.0, 0.0, f64::INFINITY),
            f64::INFINITY
        );
        let sub_tol = relative_percentage_error(0.5 * tol, 0.0, 0.0, f64::INFINITY);
        assert!((sub_tol - 100.0).abs() < 1.0e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_evaluation_kernels_zero_allocation() {
        let x = [1.0e8 + 1.0, 1.0e8 + 2.0, 1.0e8 + 3.0, 1.0e8 + 4.0];
        let y = [1.0e8 + 2.0, 1.0e8 + 3.0, 1.0e8 + 4.0, 1.0e8 + 5.0];

        let ((r, rmse, sp_err), metrics) = VerifiedEngine::run_verified(|| {
            let r = pearson_correlation(&x, &y);
            let rmse = root_mean_square_error(&x, &y);
            let sp_err = spatial_accuracy_error(50.0, 45.0);
            (r, rmse, sp_err)
        });

        assert!(metrics.function_calls >= 4);
        assert!((r.unwrap() - 1.0).abs() < 1.0e-12);
        assert!((rmse.unwrap() - 1.0).abs() < 1.0e-12);
        assert!((sp_err - 10.0).abs() < 1.0e-12);
    }
}
