//! Validation Metrics
//!
//! Standard metrics used to assess the accuracy, correlation, and reliability of respiratory signals.

use crate::physics::medical::evaluation;

/// Calculates the Percentage Error between a measured value and a reference value.
///
/// # Arguments
///
/// * `measured` - The experimental value ($X_{measured}$).
/// * `reference` - The ground truth value ($X_{reference}$).
///
/// # Returns
///
/// * `f64` - The percentage error.
///
/// # Formula
///
/// $Percentage Error = \frac{X_{measured} - X_{reference}}{X_{measured}} \times 100\%$
///
/// Note: This implementation follows the specific formula provided in the user request,
/// which uses `measured` in the denominator.
#[verified_engine::verified]
pub fn percentage_error(measured: f64, reference: f64) -> f64 {
    evaluation::relative_percentage_error(measured, reference, 0.0, f64::INFINITY)
}

/// Calculates the Root-Mean-Square Error (RMSE) between two datasets.
///
/// # Arguments
///
/// * `measured` - Vector of measured values ($X$).
/// * `reference` - Vector of reference values ($Y$).
///
/// # Returns
///
/// * `f64` - The RMSE value.
///
/// # Formula
///
/// $RMSE = \sqrt{\frac{1}{n} \sum (X_i - Y_i)^2}$
#[verified_engine::verified]
pub fn root_mean_square_error(measured: &[f64], reference: &[f64]) -> f64 {
    let n = measured.len().min(reference.len());
    evaluation::root_mean_square_error(&measured[..n], &reference[..n]).unwrap_or(0.0)
}

/// Calculates the Pearson Correlation Coefficient (r).
///
/// # Arguments
///
/// * `x` - First dataset.
/// * `y` - Second dataset.
///
/// # Returns
///
/// * `f64` - The correlation coefficient r (-1.0 to 1.0).
#[verified_engine::verified]
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    evaluation::pearson_correlation(&x[..n], &y[..n]).unwrap_or(0.0)
}

/// Calculates the DICE Similarity Coefficient.
///
/// # Arguments
///
/// * `set_a` - Boolean mask or set of indices for volume A.
/// * `set_b` - Boolean mask or set of indices for volume B.
///
/// # Returns
///
/// * `f64` - Dice coefficient (0.0 to 1.0).
///
/// # Formula
///
/// $DSC = \frac{2 |A \cap B|}{|A| + |B|}$
#[verified_engine::verified]
pub fn dice_similarity_coefficient(set_a: &[bool], set_b: &[bool]) -> f64 {
    let n = set_a.len().min(set_b.len());
    let mut intersection = 0;
    let mut size_a = 0;
    let mut size_b = 0;

    for i in 0..n {
        if set_a[i] {
            size_a += 1;
        }
        if set_b[i] {
            size_b += 1;
        }
        if set_a[i] && set_b[i] {
            intersection += 1;
        }
    }

    if size_a + size_b == 0 {
        return 1.0; // Both empty implies perfect overlap (vacuously)
    }

    (2.0 * intersection as f64) / (size_a as f64 + size_b as f64)
}

/// Calculates the Time Shift Error (Sum of Squared Differences) for a given shift.
///
/// # Arguments
///
/// * `measured` - The measured signal Y_measured.
/// * `reference` - The reference signal Y_reference.
///
/// # Returns
///
/// * `f64` - The sum of squared errors.
///
/// # Formula
///
/// $Error = \sum (Y_{measured}\[i\] - Y_{reference}\[i\])^2$
#[verified_engine::verified]
pub fn time_shift_error(measured: &[f64], reference: &[f64]) -> f64 {
    let n = measured.len().min(reference.len());
    measured
        .iter()
        .zip(reference.iter())
        .take(n)
        .map(|(m, r)| (m - r).powi(2))
        .sum()
}
