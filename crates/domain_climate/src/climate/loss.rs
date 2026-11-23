//! This module implements the loss functions for the CERA framework, delegating optimal-transport
//! metrics (exact 1D Wasserstein $W_1$ / EMD and entropic log-domain Sinkhorn) to
//! [`engine_prob::transport`].

use crate::error::ClimateError;
use engine_prob::{SinkhornConfig, sinkhorn_log_domain, wasserstein_1d_into};
use math_commons::error::MathError;
use nalgebra::{Dyn, Matrix, Storage};

#[inline]
const fn idx2d(row: usize, col: usize, cols: usize) -> usize {
    row * cols + col
}

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
#[verified_engine::verified]
pub fn mse_loss<S1: Storage<f32, Dyn, Dyn>, S2: Storage<f32, Dyn, Dyn>>(
    y_true: &Matrix<f32, Dyn, Dyn, S1>,
    y_pred: &Matrix<f32, Dyn, Dyn, S2>,
) -> f32 {
    try_mse_loss(y_true, y_pred).unwrap_or_else(|e| panic!("{e}"))
}

/// Fallible Earth Mover's Distance ($W_1$) across paired 1D column distributions via
/// [`engine_prob::wasserstein_1d_into`].
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

    let mut total_emd = 0.0_f64;
    let mut col1 = Vec::with_capacity(z1.nrows());
    let mut col2 = Vec::with_capacity(z2.nrows());

    for i in 0..z1.ncols() {
        col1.clear();
        col1.extend(z1.column(i).iter().map(|&v| f64::from(v)));
        col2.clear();
        col2.extend(z2.column(i).iter().map(|&v| f64::from(v)));
        let w1 = wasserstein_1d_into(&mut col1, &mut col2).map_err(|e| {
            ClimateError::Math(MathError::NumericalError {
                reason: e.to_string(),
            })
        })?;
        total_emd += w1;
    }

    Ok((total_emd / (z1.ncols() as f64)) as f32)
}

/// Computes the Earth Mover's Distance (EMD) for multiple 1D distributions.
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

/// Computes entropic log-domain Sinkhorn optimal-transport cost averaged across paired columns of
/// `z1` and `z2` under uniform empirical weights and $|x - y|$ ground cost via
/// [`engine_prob::sinkhorn_log_domain`].
#[verified_engine::verified]
pub fn try_sinkhorn_transport_loss<S1: Storage<f32, Dyn, Dyn>, S2: Storage<f32, Dyn, Dyn>>(
    z1: &Matrix<f32, Dyn, Dyn, S1>,
    z2: &Matrix<f32, Dyn, Dyn, S2>,
    epsilon: f64,
) -> Result<f32, ClimateError> {
    if z1.shape() != z2.shape() {
        return Err(ClimateError::ShapeMismatch {
            context: "Matrices must have the same shape for Sinkhorn OT.",
            expected: z1.shape(),
            actual: z2.shape(),
        });
    }
    let n = z1.nrows();
    let cols = z1.ncols();
    if n == 0 || cols == 0 {
        return Ok(0.0);
    }
    let uniform = vec![1.0 / (n as f64); n];
    let mut cost = vec![0.0_f64; n * n];
    let cfg = SinkhornConfig {
        epsilon,
        max_iters: 500,
        tolerance: 1.0e-9,
    };
    let mut total_cost = 0.0_f64;
    for col in 0..cols {
        for r in 0..n {
            let xr = f64::from(z1[(r, col)]);
            for c in 0..n {
                let yc = f64::from(z2[(c, col)]);
                cost[idx2d(r, c, n)] = (xr - yc).abs();
            }
        }
        let (report, _plan) =
            sinkhorn_log_domain(&uniform, &uniform, &cost, &cfg).map_err(|e| {
                ClimateError::Math(MathError::NumericalError {
                    reason: e.to_string(),
                })
            })?;
        total_cost += report.transport_cost;
    }
    Ok((total_cost / (cols as f64)) as f32)
}

/// Computes the combined loss for the CERA framework.
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
        let loss = mse_loss(&y_true, &y_pred);
        assert!((loss - 0.5).abs() < math_commons::registry::TOLERANCE_FAST_F32);
    }

    #[test]
    #[verified_engine::verified]
    fn test_earth_movers_distance_and_sinkhorn_consistency() {
        let z1 = DMatrix::from_row_slice(4, 2, &[1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0]);
        let z2 = DMatrix::from_row_slice(4, 2, &[5.0, 4.0, 6.0, 3.0, 7.0, 2.0, 8.0, 1.0]);

        let emd = earth_movers_distance(&z1, &z2);
        assert!((emd - 4.0).abs() < math_commons::registry::TOLERANCE_FAST_F32);

        let sinkhorn = try_sinkhorn_transport_loss(&z1, &z2, 1.0e-3).unwrap();
        assert!((sinkhorn - emd).abs() < 1.0e-3);
    }
}
