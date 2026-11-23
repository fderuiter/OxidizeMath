use crate::error::EngineeringError;
use nalgebra::{DMatrix, DVector, RealField};

/// Estimates the beta parameters of the GLM using ordinary least squares.
/// beta = (X^T * X)^-1 * X^T * Y
///
/// # Arguments
/// * `x` - The design matrix (subjects x predictors).
/// * `y` - The data vector (e.g., thickness at a vertex for all subjects).
///
/// # Returns
/// The estimated beta parameters, or an [`EngineeringError`] if dimensions mismatch or `X^T * X` is singular.
#[verified_engine::verified]
pub fn estimate_beta<T: RealField>(
    x: &DMatrix<T>,
    y: &DVector<T>,
) -> Result<DVector<T>, EngineeringError> {
    if x.nrows() != y.len() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Observation vector y length must match design matrix row count",
            expected: x.nrows(),
            actual: y.len(),
        });
    }
    let xt = x.transpose();
    let xtx = &xt * x;
    let xtx_inv = xtx
        .try_inverse()
        .ok_or(EngineeringError::SingularDesignMatrix)?;
    let xty = xt * y;
    Ok(xtx_inv * xty)
}

/// Calculates the t-statistic for a given contrast.
/// t = c^T * beta / sqrt(sigma^2 * c^T * (X^T * X)^-1 * c)
///
/// # Arguments
/// * `c` - The contrast vector.
/// * `beta` - The estimated beta parameters.
/// * `x` - The design matrix.
/// * `residual_variance` - sigma^2, the variance of the residuals (Y - X*beta).
///
/// # Returns
/// The t-statistic, or an [`EngineeringError`] if matrices are non-conformable or non-invertible.
#[verified_engine::verified]
pub fn t_statistic<T: RealField + Copy>(
    c: &DVector<T>,
    beta: &DVector<T>,
    x: &DMatrix<T>,
    residual_variance: T,
) -> Result<T, EngineeringError> {
    if c.len() != x.ncols() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Contrast vector c length must match design matrix predictor count",
            expected: x.ncols(),
            actual: c.len(),
        });
    }
    if beta.len() != x.ncols() {
        return Err(EngineeringError::DimensionMismatch {
            context: "Parameter vector beta length must match design matrix predictor count",
            expected: x.ncols(),
            actual: beta.len(),
        });
    }
    let xtx = x.transpose() * x;
    let xtx_inv = xtx
        .try_inverse()
        .ok_or(EngineeringError::SingularDesignMatrix)?;

    let numerator = c.dot(beta);

    // Using dot product avoids allocating a 1x1 matrix and avoids computing c^T first
    let variance_term = c.dot(&(xtx_inv * c));
    let scaled_variance = residual_variance * variance_term;
    if scaled_variance <= T::zero() {
        return Err(EngineeringError::ZeroContrastVariance);
    }
    let denominator = scaled_variance.sqrt();

    if denominator == T::zero() {
        return Err(EngineeringError::ZeroContrastVariance);
    }

    Ok(numerator / denominator)
}
