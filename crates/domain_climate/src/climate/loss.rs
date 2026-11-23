//! This module implements the loss functions for the CERA framework.

use crate::error::ClimateError;
use nalgebra::{Dyn, Matrix, Storage};

/// Fallible Mean Squared Error (MSE) between two matrices of identical shape.
///
/// Returns [`ClimateError::ShapeMismatch`] if `y_true` and `y_pred` differ in dimensions.
#[verified_engine::verified]
pub fn try_mse_loss<S1: Storage<f32, Dyn, Dyn>, S2: Storage<f32, Dyn, Dyn>>(
    y_true: &Matrix<f32, Dyn, Dyn, S1>,
    y_pred: &Matrix<f32, Dyn, Dyn, S2>,
) -> Result<f32, ClimateError> {
    if y_true.shape() != y_pred.shape() {
        return Err(ClimateError::ShapeMismatch {
            context: "Matrices must have the same shape for MSE.",
            expected: y_true.shape(),
            actual: y_pred.shape(),
        });
    }
    let count = y_true.nrows() * y_true.ncols();
    if count == 0 {
        return Ok(0.0);
    }
    let diff = y_true - y_pred;
    Ok(diff.norm_squared() / count as f32)
}

/// Computes the Mean Squared Error (MSE) between two matrices.
///
/// # Arguments
///
/// * `y_true` - The ground truth values.
/// * `y_pred` - The predicted values.
///
/// # Returns
///
/// The mean squared error.
#[verified_engine::verified]
pub fn mse_loss<S1: Storage<f32, Dyn, Dyn>, S2: Storage<f32, Dyn, Dyn>>(
    y_true: &Matrix<f32, Dyn, Dyn, S1>,
    y_pred: &Matrix<f32, Dyn, Dyn, S2>,
) -> f32 {
    try_mse_loss(y_true, y_pred).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible Earth Mover's Distance (EMD) across paired 1D column distributions.
///
/// Returns [`ClimateError::ShapeMismatch`] if `z1` and `z2` differ in dimensions.
#[verified_engine::verified]
pub fn try_earth_movers_distance<S1: Storage<f32, Dyn, Dyn>, S2: Storage<f32, Dyn, Dyn>>(
    z1: &Matrix<f32, Dyn, Dyn, S1>,
    z2: &Matrix<f32, Dyn, Dyn, S2>,
) -> Result<f32, ClimateError> {
    if z1.shape() != z2.shape() {
        return Err(ClimateError::ShapeMismatch {
            context: "Matrices must have the same shape for EMD.",
            expected: z1.shape(),
            actual: z2.shape(),
        });
    }
    if z1.ncols() == 0 || z1.nrows() == 0 {
        return Ok(0.0);
    }

    let mut total_emd = 0.0;
    let mut col1 = Vec::with_capacity(z1.nrows());
    let mut col2 = Vec::with_capacity(z2.nrows());

    for i in 0..z1.ncols() {
        col1.clear();
        col1.extend(z1.column(i).iter().copied());

        col2.clear();
        col2.extend(z2.column(i).iter().copied());

        col1.sort_unstable_by(|a, b| a.total_cmp(b));
        col2.sort_unstable_by(|a, b| a.total_cmp(b));

        let emd_i: f32 = col1
            .iter()
            .zip(col2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        total_emd += emd_i / col1.len() as f32;
    }

    Ok(total_emd / z1.ncols() as f32)
}

/// Computes the Earth Mover's Distance (EMD) for multiple 1D distributions.
///
/// This function takes two matrices where each column represents a 1D distribution.
/// It computes the EMD for each pair of columns and returns the average.
///
/// # Arguments
///
/// * `z1` - The first set of distributions, with shape (n_samples, n_distributions).
/// * `z2` - The second set of distributions, with shape (n_samples, n_distributions).
///
/// # Returns
///
/// The average Earth Mover's Distance.
///
/// # Examples
///
/// ```rust
/// # use domain_climate::__doctest_facade as math_explorer;
/// use math_explorer::climate::loss::{cera_loss, earth_movers_distance};
/// use math_explorer::nalgebra::DMatrix;
///
/// let control = DMatrix::from_row_slice(4, 1, &[1.0_f32, 2.0, 3.0, 4.0]);
/// let warm = DMatrix::from_row_slice(4, 1, &[3.0_f32, 4.0, 5.0, 6.0]);
/// let emd = earth_movers_distance(&control, &warm);
/// assert!((emd - 2.0).abs() < 1e-6);
///
/// let total = cera_loss(0.5, 0.25, emd, 0.2, 0.1);
/// assert!((total - (0.7 * 0.5 + 0.2 * 0.25 + 0.1 * 2.0)).abs() < 1e-6);
/// ```
#[verified_engine::verified]
pub fn earth_movers_distance<S1: Storage<f32, Dyn, Dyn>, S2: Storage<f32, Dyn, Dyn>>(
    z1: &Matrix<f32, Dyn, Dyn, S1>,
    z2: &Matrix<f32, Dyn, Dyn, S2>,
) -> f32 {
    try_earth_movers_distance(z1, z2).unwrap_or_else(|e| panic!("{e}"))
}

/// Computes the combined loss for the CERA framework.
/// The total loss is a weighted sum of the reconstruction loss, the prediction
/// loss, and the EMD loss.
///
/// # Arguments
///
/// * `reconstruction_loss` - The MSE of the autoencoder reconstruction.
/// * `prediction_loss` - The MSE of the predictor output.
/// * `emd_loss` - The Earth Mover's Distance between latent representations.
/// * `lambda_pred` - The weight for the prediction loss.
/// * `lambda_emd` - The weight for the EMD loss.
///
/// # Returns
///
/// The total weighted loss.
#[verified_engine::verified]
pub fn cera_loss(
    reconstruction_loss: f32,
    prediction_loss: f32,
    emd_loss: f32,
    lambda_pred: f32,
    lambda_emd: f32,
) -> f32 {
    (1.0 - lambda_pred - lambda_emd).max(0.0) * reconstruction_loss
        + lambda_pred * prediction_loss
        + lambda_emd * emd_loss
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    #[verified_engine::verified]
    fn test_mse_loss() {
        let y_true = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        let y_pred = DMatrix::from_row_slice(2, 2, &[1.0, 3.0, 2.0, 4.0]);
        // Differences are 0, -1, 1, 0. Squared differences are 0, 1, 1, 0.
        // Sum of squared differences is 2.
        // MSE = 2 / 4 = 0.5.
        let loss = mse_loss(&y_true, &y_pred);
        assert!((loss - 0.5).abs() < math_commons::registry::TOLERANCE_FAST_F32);
    }

    #[test]
    #[verified_engine::verified]
    fn test_earth_movers_distance() {
        // Two distributions for the first channel, two for the second
        let z1 = DMatrix::from_row_slice(4, 2, &[1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0]);
        let z2 = DMatrix::from_row_slice(4, 2, &[5.0, 4.0, 6.0, 3.0, 7.0, 2.0, 8.0, 1.0]);

        // Channel 1: z1_sorted = [1,2,3,4], z2_sorted = [5,6,7,8]
        // Diff = [4,4,4,4]. Sum of abs diff = 16. EMD_1 = 16 / 4 = 4.

        // Channel 2: z1_sorted = [5,6,7,8], z2_sorted = [1,2,3,4]
        // Diff = [4,4,4,4]. Sum of abs diff = 16. EMD_2 = 16 / 4 = 4.

        // Average EMD = (4 + 4) / 2 = 4.
        let emd = earth_movers_distance(&z1, &z2);
        assert!((emd - 4.0).abs() < math_commons::registry::TOLERANCE_FAST_F32);
    }
}
