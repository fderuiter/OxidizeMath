use nalgebra::{DVector, RealField};

// Re-export the core trait and implementations from pure_math
pub use pure_math::pure_math::analysis::optimization::{
    Adam as GenericAdam, ModelOptimizer as Optimizer, OptimizationError,
    Optimizer as GenericOptimizer, ParamType, SGD,
};

use crate::error::AIError;

/// Specialized [`GenericAdam`] optimizer keyed by `(layer_index, ParamType)` for neural layers.
pub type Adam<T> = GenericAdam<T, (usize, ParamType)>;

fn validate_mse_inputs<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<T, AIError> {
    if y_pred.len() != y_true.len() {
        return Err(math_commons::error::MathError::DimensionMismatch {
            expected: math_commons::math_kernel::types::Dimension(y_true.len()),
            actual: math_commons::math_kernel::types::Dimension(y_pred.len()),
        }
        .into());
    }
    if y_pred.is_empty() {
        return Err(math_commons::error::MathError::NumericalError {
            reason: "MSE loss requires non-empty vectors",
        }
        .into());
    }
    if !y_pred.iter().all(|v| v.is_finite()) || !y_true.iter().all(|v| v.is_finite()) {
        return Err(math_commons::error::MathError::NumericalError {
            reason: "MSE loss inputs must be finite",
        }
        .into());
    }
    T::from_usize(y_pred.len()).ok_or_else(|| {
        math_commons::error::MathError::ConversionError {
            reason: "Failed to convert usize to T",
        }
        .into()
    })
}

/// Mean Squared Error (MSE) Loss function.
/// Used primarily for Regression.
///
/// $$J(\theta) = \frac{1}{n} \sum_{i=1}^{n} \left(y^{(i)} - \hat{y}^{(i)}\right)^2$$
#[verified_engine::verified]
pub fn mean_squared_error<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<T, AIError> {
    let n = validate_mse_inputs(y_pred, y_true)?;
    let mut sum_sq = T::zero();
    for (&yp, &yt) in y_pred.iter().zip(y_true.iter()) {
        let diff = yp - yt;
        sum_sq += diff * diff;
    }
    let mse = sum_sq / n;
    if !mse.is_finite() {
        return Err(math_commons::error::MathError::NumericalError {
            reason: "MSE loss overflowed to non-finite value",
        }
        .into());
    }
    Ok(mse)
}

/// Derivative of MSE with respect to y_pred.
/// \frac{\partial J}{\partial \hat{y}} = \frac{2}{n} (\hat{y} - y)
#[verified_engine::verified]
pub fn mse_prime<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<DVector<T>, AIError> {
    let n = validate_mse_inputs(y_pred, y_true)?;
    let two = T::from_f64(2.0).ok_or(math_commons::error::MathError::ConversionError {
        reason: "Failed to convert 2.0 to T",
    })?;
    let grad = (y_pred - y_true) * (two / n);
    if !grad.iter().all(|v| v.is_finite()) {
        return Err(math_commons::error::MathError::NumericalError {
            reason: "MSE gradient overflowed to non-finite value",
        }
        .into());
    }
    Ok(grad)
}

fn validate_probability_vector<T: RealField + Copy>(values: &DVector<T>) -> Result<(), AIError> {
    let zero = T::zero();
    let one = T::one();
    for &v in values.iter() {
        if !v.is_finite() || !(v >= zero && v <= one) {
            return Err(math_commons::error::MathError::NumericalError {
                reason: "Probability vector contains non-finite or out-of-[0, 1] value",
            }
            .into());
        }
    }
    Ok(())
}

fn clamp_unit_prob<T: RealField + Copy>(value: T, lower: T, upper: T) -> T {
    if value < lower {
        lower
    } else if value > upper {
        upper
    } else {
        value
    }
}

/// Cross-Entropy Loss function for binary (`len == 1`) and categorical (`len > 1`) classification.
///
/// Validates finite `[0.0, 1.0]` inputs and matching non-empty dimensions, clips predicted
/// probabilities symmetrically to `[epsilon, 1.0 - epsilon]`, and includes the negative-class
/// complement `-(1 - y) * ln(1 - p_c)` for scalar binary predictions (`len == 1`).
#[verified_engine::verified]
pub fn cross_entropy_loss<T: RealField + Copy>(
    y_pred: &DVector<T>,
    y_true: &DVector<T>,
) -> Result<T, AIError> {
    if y_pred.len() != y_true.len() {
        return Err(math_commons::error::MathError::DimensionMismatch {
            expected: math_commons::math_kernel::types::Dimension(y_true.len()),
            actual: math_commons::math_kernel::types::Dimension(y_pred.len()),
        }
        .into());
    }
    if y_pred.is_empty() {
        return Err(math_commons::error::MathError::NumericalError {
            reason: "cross_entropy_loss requires non-empty vectors",
        }
        .into());
    }
    validate_probability_vector(y_pred)?;
    validate_probability_vector(y_true)?;

    let eps_base = T::from_f64(1e-15).ok_or(math_commons::error::MathError::ConversionError {
        reason: "Failed to convert 1e-15 to T",
    })?;
    let sixteen = T::from_f64(16.0).ok_or(math_commons::error::MathError::ConversionError {
        reason: "Failed to convert 16.0 to T",
    })?;
    let min_eps = T::default_epsilon() * sixteen;
    let epsilon = if eps_base > min_eps {
        eps_base
    } else {
        min_eps
    };
    let one = T::one();
    let upper = one - epsilon;

    if y_pred.len() == 1 {
        let p_c = clamp_unit_prob(y_pred[0], epsilon, upper);
        let q_c = clamp_unit_prob(one - y_pred[0], epsilon, upper);
        let y = y_true[0];
        let loss = -(y * p_c.ln() + (one - y) * q_c.ln());
        return Ok(if loss > T::zero() { loss } else { T::zero() });
    }

    let mut total = T::zero();
    for (&p, &y) in y_pred.iter().zip(y_true.iter()) {
        let p_c = clamp_unit_prob(p, epsilon, upper);
        total -= y * p_c.ln();
    }
    Ok(if total > T::zero() { total } else { T::zero() })
}

/// Derivative of Cross-Entropy Loss combined with Softmax.
///
/// For $L(\mathbf{z}) = -\sum_j y_j \ln \hat{y}_j$ with $\hat{\mathbf{y}} = \operatorname{softmax}(\mathbf{z})$,
/// the analytical gradient with respect to $\mathbf{z}$ is:
/// $$\nabla_{\mathbf{z}} L = \left(\sum_j y_j\right)\hat{\mathbf{y}} - \mathbf{y}$$
#[verified_engine::verified]
pub fn cross_entropy_softmax_prime<T: RealField + Copy>(
    z_logits: &DVector<T>,
    y_true: &DVector<T>,
) -> DVector<T> {
    if z_logits.is_empty()
        || z_logits.len() != y_true.len()
        || !y_true.iter().all(|v| v.is_finite())
    {
        return DVector::zeros(z_logits.len());
    }
    let y_pred = crate::ai::activation_kernels::softmax(z_logits);
    let target_sum = y_true.iter().copied().fold(T::zero(), |acc, v| acc + v);
    if !target_sum.is_finite() {
        return DVector::zeros(z_logits.len());
    }
    y_pred * target_sum - y_true
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    #[verified_engine::verified]
    fn test_mse_and_mse_prime_reject_empty_mismatch_and_nan() {
        let empty = DVector::<f64>::zeros(0);
        assert!(mean_squared_error(&empty, &empty).is_err());
        assert!(mse_prime(&empty, &empty).is_err());

        let pred = DVector::from_vec(vec![1.0_f64, 2.0_f64]);
        let short = DVector::from_vec(vec![1.0_f64]);
        assert!(mean_squared_error(&pred, &short).is_err());
        assert!(mse_prime(&pred, &short).is_err());

        let nan_vec = DVector::from_vec(vec![1.0_f64, f64::NAN]);
        assert!(mean_squared_error(&nan_vec, &pred).is_err());
        assert!(mean_squared_error(&pred, &nan_vec).is_err());
        assert!(mse_prime(&nan_vec, &pred).is_err());
        assert!(mse_prime(&pred, &nan_vec).is_err());

        let target = DVector::from_vec(vec![0.0_f64, 4.0_f64]);
        let mse = mean_squared_error(&pred, &target).expect("valid mse");
        assert_relative_eq!(mse, 2.5, epsilon = 1e-12);
        let grad = mse_prime(&pred, &target).expect("valid mse_prime");
        assert_relative_eq!(grad[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(grad[1], -2.0, epsilon = 1e-12);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cross_entropy_softmax_prime_matches_finite_differences_for_non_simplex_targets() {
        let logits = DVector::from_vec(vec![0.4_f64, -0.7_f64, 1.1_f64]);
        // Non-simplex multi-label / unnormalized target within [0, 1] per component (sum = 1.65 != 1.0)
        let y_non_simplex = DVector::from_vec(vec![0.9_f64, 0.25_f64, 0.5_f64]);
        let analytical = cross_entropy_softmax_prime(&logits, &y_non_simplex);

        let eps = 1e-7_f64;
        for i in 0..logits.len() {
            let mut z_plus = logits.clone();
            let mut z_minus = logits.clone();
            z_plus[i] += eps;
            z_minus[i] -= eps;
            let p_plus = crate::ai::activation_kernels::softmax(&z_plus);
            let p_minus = crate::ai::activation_kernels::softmax(&z_minus);
            let loss_plus = cross_entropy_loss(&p_plus, &y_non_simplex).expect("loss_plus");
            let loss_minus = cross_entropy_loss(&p_minus, &y_non_simplex).expect("loss_minus");
            let fd = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (analytical[i] - fd).abs() < 1e-6,
                "component {i} mismatch: analytical={}, fd={fd}",
                analytical[i]
            );
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_cross_entropy_softmax_prime_uses_guarded_softmax() {
        let logits = DVector::from_vec(vec![0.0_f64, 0.0_f64]);
        let target = DVector::from_vec(vec![1.0_f64, 0.0_f64]);
        let grad = cross_entropy_softmax_prime(&logits, &target);
        assert_relative_eq!(
            grad[0],
            -0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            grad[1],
            0.5,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let empty = DVector::<f64>::zeros(0);
        assert_eq!(cross_entropy_softmax_prime(&empty, &empty).len(), 0);

        let nan_logits = DVector::from_vec(vec![1.0_f64, f64::NAN]);
        let nan_grad = cross_entropy_softmax_prime(&nan_logits, &target);
        assert_eq!(nan_grad[0], -1.0);
        assert_eq!(nan_grad[1], 0.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cross_entropy_loss_rejects_nan_and_mismatch_and_clips_symmetrically() {
        let valid_pred = DVector::from_vec(vec![0.8_f64, 0.2_f64]);
        let valid_true = DVector::from_vec(vec![1.0_f64, 0.0_f64]);
        let loss = cross_entropy_loss(&valid_pred, &valid_true).expect("valid categorical loss");
        assert_relative_eq!(
            loss,
            -(0.8_f64.ln()),
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );

        // NaN and out-of-range inputs return Err.
        let nan_pred = DVector::from_vec(vec![f64::NAN, 0.5_f64]);
        assert!(cross_entropy_loss(&nan_pred, &valid_true).is_err());
        let nan_true = DVector::from_vec(vec![1.0_f64, f64::NAN]);
        assert!(cross_entropy_loss(&valid_pred, &nan_true).is_err());
        let over_one = DVector::from_vec(vec![1.2_f64, 0.0_f64]);
        assert!(cross_entropy_loss(&over_one, &valid_true).is_err());
        let neg_prob = DVector::from_vec(vec![-0.1_f64, 1.0_f64]);
        assert!(cross_entropy_loss(&neg_prob, &valid_true).is_err());

        // Dimension mismatch and empty vectors return Err.
        let short_true = DVector::from_vec(vec![1.0_f64]);
        assert!(cross_entropy_loss(&valid_pred, &short_true).is_err());
        let empty = DVector::<f64>::zeros(0);
        assert!(cross_entropy_loss(&empty, &empty).is_err());

        // Scalar binary cross-entropy includes negative-class complement and clips 0.0 / 1.0 symmetrically.
        let bin_pred_one = DVector::from_vec(vec![1.0_f64]);
        let bin_true_zero = DVector::from_vec(vec![0.0_f64]);
        let complement_loss =
            cross_entropy_loss(&bin_pred_one, &bin_true_zero).expect("clipped complement");
        assert!(complement_loss.is_finite() && complement_loss > 30.0);

        let bin_true_one = DVector::from_vec(vec![1.0_f64]);
        let zero_err_loss =
            cross_entropy_loss(&bin_pred_one, &bin_true_one).expect("clipped upper");
        assert!((0.0..1e-12).contains(&zero_err_loss));

        // Single-precision f32 upper clipping strictly avoids ln(1.0_f32) == 0 collapse on complement.
        let f32_pred = DVector::from_vec(vec![1.0_f32]);
        let f32_zero = DVector::from_vec(vec![0.0_f32]);
        let f32_loss = cross_entropy_loss(&f32_pred, &f32_zero).expect("f32 clipped complement");
        assert!(f32_loss.is_finite() && f32_loss > 10.0);
    }
}
