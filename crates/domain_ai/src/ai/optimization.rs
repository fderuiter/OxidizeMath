use nalgebra::{DVector, RealField};

// Re-export the core trait and implementations from pure_math
pub use pure_math::pure_math::analysis::optimization::{
    Adam as GenericAdam, ModelOptimizer as Optimizer, OptimizationError,
    Optimizer as GenericOptimizer, ParamType, SGD,
};

// Type alias to maintain backward compatibility for Adam
// The generic Adam now takes a Key, so we fix it to (usize, ParamType) for the AI module.
#[allow(missing_docs)]
pub type Adam<T> = GenericAdam<T, (usize, ParamType)>;

/// Mean Squared Error (MSE) Loss function.
/// Used primarily for Regression.
///
/// J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2
use crate::error::AIError;

#[allow(missing_docs)]
#[verified_engine::verified]
pub fn mean_squared_error<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<T, AIError> {
    let diff = y_pred - y_true;
    let n = T::from_usize(y_pred.len()).ok_or_else(|| {
        math_commons::error::MathError::ConversionError {
            reason: "Failed to convert usize to T".to_string(),
        }
    })?;
    Ok(diff.dot(&diff) / n)
}

/// Derivative of MSE with respect to y_pred.
/// \frac{\partial J}{\partial \hat{y}} = \frac{2}{n} (\hat{y} - y)
#[verified_engine::verified]
pub fn mse_prime<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<DVector<T>, AIError> {
    let n = T::from_usize(y_pred.len()).ok_or_else(|| {
        math_commons::error::MathError::ConversionError {
            reason: "Failed to convert usize to T".to_string(),
        }
    })?;
    let two = T::from_f64(2.0).ok_or_else(|| math_commons::error::MathError::ConversionError {
        reason: "Failed to convert 2.0 to T".to_string(),
    })?;
    Ok((y_pred - y_true) * (two / n))
}

/// Cross-Entropy Loss function.
/// Used primarily for Classification.
///
/// J(\theta) = - \sum_{i} y_i \log(\hat{y}_i)
///
/// Note: This implementation assumes y_true is a one-hot vector or probability distribution.
#[verified_engine::verified]
pub fn cross_entropy_loss<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<T, AIError> {
    // Add a small epsilon to avoid log(0)
    let epsilon =
        T::from_f64(1e-15).ok_or_else(|| math_commons::error::MathError::ConversionError {
            reason: "Failed to convert 1e-15 to T".to_string(),
        })?;
    let y_pred_safe = y_pred.map(|v| if v > epsilon { v } else { epsilon });

    let log_likelihood = y_pred_safe.map(|v| v.ln());
    Ok(-(y_true.dot(&log_likelihood)))
}

/// Derivative of Cross-Entropy Loss combined with Softmax.
///
/// If output layer is Softmax and Loss is Cross-Entropy, the gradient w.r.t the logits z is:
/// \frac{\partial L}{\partial z} = \hat{y} - y
#[verified_engine::verified]
pub fn cross_entropy_softmax_prime<T: RealField + Copy>(
    z_logits: &DVector<T>,
    y_true: &DVector<T>,
) -> DVector<T> {
    let y_pred = softmax(z_logits);
    y_pred - y_true
}

/// Helper Softmax function for generic types.
#[verified_engine::verified]
fn softmax<T: RealField + Copy>(z: &DVector<T>) -> DVector<T> {
    let max_z = z.max();
    let exps = z.map(|v| (v - max_z).exp());
    let sum_exps = exps.sum();
    exps / sum_exps
}
