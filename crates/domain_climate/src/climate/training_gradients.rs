//! Analytical MSE reconstruction and prediction gradient kernels for CERA training.

use nalgebra::{DVector, Dyn, Matrix, Storage};

/// Writes analytical Mean Squared Error reconstruction gradients and averaged inputs into
/// pre-allocated vectors without heap allocation.
#[verified_engine::verified]
pub(super) fn compute_reconstruction_gradients_into<
    S1: Storage<f32, Dyn, Dyn>,
    S2: Storage<f32, Dyn, Dyn>,
    S3: Storage<f32, Dyn, Dyn>,
    S4: Storage<f32, Dyn, Dyn>,
>(
    control_input: &Matrix<f32, Dyn, Dyn, S1>,
    control_recon: &Matrix<f32, Dyn, Dyn, S2>,
    warm_input: &Matrix<f32, Dyn, Dyn, S3>,
    warm_recon: &Matrix<f32, Dyn, Dyn, S4>,
    ae_input: &mut DVector<f32>,
    ae_loss_grad: &mut DVector<f32>,
) {
    let input_rows = control_input.nrows();
    let in_channels = ae_input.len();
    ae_input.fill(0.0);
    ae_loss_grad.fill(0.0);
    if input_rows == 0 || in_channels == 0 {
        return;
    }

    let scale = 1.0 / (input_rows.saturating_mul(in_channels)) as f32;
    for r in 0..input_rows {
        for c in 0..in_channels {
            ae_input[c] += 0.5 * (control_input[(r, c)] + warm_input[(r, c)]);
            ae_loss_grad[c] += scale
                * ((control_recon[(r, c)] - control_input[(r, c)])
                    + (warm_recon[(r, c)] - warm_input[(r, c)]));
        }
    }
    *ae_input /= input_rows as f32;
}

/// Computes analytical Mean Squared Error gradients and average input for reconstruction batches.
#[verified_engine::verified(alloc_ok = "convenience helper allocates output gradient vectors")]
pub(super) fn compute_reconstruction_gradients<
    S1: Storage<f32, Dyn, Dyn>,
    S2: Storage<f32, Dyn, Dyn>,
    S3: Storage<f32, Dyn, Dyn>,
    S4: Storage<f32, Dyn, Dyn>,
>(
    control_input: &Matrix<f32, Dyn, Dyn, S1>,
    control_recon: &Matrix<f32, Dyn, Dyn, S2>,
    warm_input: &Matrix<f32, Dyn, Dyn, S3>,
    warm_recon: &Matrix<f32, Dyn, Dyn, S4>,
    in_channels: usize,
) -> (DVector<f32>, DVector<f32>) {
    let mut ae_input = DVector::zeros(in_channels);
    let mut ae_loss_grad = DVector::zeros(in_channels);
    compute_reconstruction_gradients_into(
        control_input,
        control_recon,
        warm_input,
        warm_recon,
        &mut ae_input,
        &mut ae_loss_grad,
    );
    (ae_input, ae_loss_grad)
}

/// Writes analytical Mean Squared Error prediction gradients and averaged inputs into
/// pre-allocated vectors without heap allocation.
#[verified_engine::verified]
pub(super) fn compute_prediction_gradients_into<
    S1: Storage<f32, Dyn, Dyn>,
    S2: Storage<f32, Dyn, Dyn>,
    S3: Storage<f32, Dyn, Dyn>,
>(
    predictor_input: &Matrix<f32, Dyn, Dyn, S1>,
    prediction: &Matrix<f32, Dyn, Dyn, S2>,
    targets: &Matrix<f32, Dyn, Dyn, S3>,
    lambda_pred: f32,
    pred_input_vec: &mut DVector<f32>,
    pred_loss_grad: &mut DVector<f32>,
) {
    let batch_size = prediction.nrows();
    let pred_in_dim = pred_input_vec.len();
    let output_size = pred_loss_grad.len();
    pred_input_vec.fill(0.0);
    pred_loss_grad.fill(0.0);
    if batch_size == 0 || pred_in_dim == 0 || output_size == 0 {
        return;
    }

    for r in 0..batch_size {
        for c in 0..pred_in_dim {
            pred_input_vec[c] += predictor_input[(r, c)];
        }
    }
    *pred_input_vec /= batch_size as f32;

    let pred_scale = if lambda_pred.is_finite() {
        lambda_pred * 2.0 / (batch_size.saturating_mul(output_size)) as f32
    } else {
        0.0
    };
    for r in 0..batch_size {
        for c in 0..output_size {
            pred_loss_grad[c] += pred_scale * (prediction[(r, c)] - targets[(r, c)]);
        }
    }
}

/// Computes analytical Mean Squared Error gradients and average input for prediction batches.
#[verified_engine::verified(alloc_ok = "convenience helper allocates output gradient vectors")]
pub(super) fn compute_prediction_gradients<
    S1: Storage<f32, Dyn, Dyn>,
    S2: Storage<f32, Dyn, Dyn>,
    S3: Storage<f32, Dyn, Dyn>,
>(
    predictor_input: &Matrix<f32, Dyn, Dyn, S1>,
    prediction: &Matrix<f32, Dyn, Dyn, S2>,
    targets: &Matrix<f32, Dyn, Dyn, S3>,
    pred_in_dim: usize,
    output_size: usize,
    lambda_pred: f32,
) -> (DVector<f32>, DVector<f32>) {
    let mut pred_input_vec = DVector::zeros(pred_in_dim);
    let mut pred_loss_grad = DVector::zeros(output_size);
    compute_prediction_gradients_into(
        predictor_input,
        prediction,
        targets,
        lambda_pred,
        &mut pred_input_vec,
        &mut pred_loss_grad,
    );
    (pred_input_vec, pred_loss_grad)
}

/// Clips the L2 norm of a gradient vector if it exceeds `max_norm`.
#[verified_engine::verified]
pub(super) fn clip_gradient_norm(grad: &mut DVector<f32>, max_norm: f32) {
    let norm = grad.norm();
    if norm > max_norm && norm.is_finite() && norm > 0.0 {
        *grad *= max_norm / norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    fn test_compute_reconstruction_gradients_owned_sliced_and_zero_alloc() {
        let control_input =
            DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let control_recon = DMatrix::from_row_slice(2, 2, &[1.1, 2.1, 2.9, 3.9]);
        let warm_input = DMatrix::from_row_slice(4, 2, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let warm_recon = DMatrix::from_row_slice(2, 2, &[1.9, 3.2, 4.1, 4.8]);

        let control_slice = control_input.rows(0, 2);
        let warm_slice = warm_input.rows(0, 2);

        let (in_slice, grad_slice) = compute_reconstruction_gradients(
            &control_slice,
            &control_recon,
            &warm_slice,
            &warm_recon,
            2,
        );
        let control_owned = control_slice.clone_owned();
        let warm_owned = warm_slice.clone_owned();
        let (in_owned, grad_owned) = compute_reconstruction_gradients(
            &control_owned,
            &control_recon,
            &warm_owned,
            &warm_recon,
            2,
        );

        let mut in_into = DVector::zeros(2);
        let mut grad_into = DVector::zeros(2);
        let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            compute_reconstruction_gradients_into(
                &control_slice,
                &control_recon,
                &warm_slice,
                &warm_recon,
                &mut in_into,
                &mut grad_into,
            );
        });
        assert_eq!(metrics.allocations, 0);
        assert_eq!(in_slice, in_owned);
        assert_eq!(grad_slice, grad_owned);
        assert_eq!(in_slice, in_into);
        assert_eq!(grad_slice, grad_into);
    }

    #[test]
    fn test_compute_prediction_gradients_owned_sliced_and_zero_alloc() {
        let pred_in = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let prediction = DMatrix::from_row_slice(2, 2, &[0.5, 1.5, 2.5, 3.5]);
        let full_targets = DMatrix::from_row_slice(4, 2, &[0.6, 1.4, 2.4, 3.6, 9.0, 9.0, 9.0, 9.0]);
        let target_slice = full_targets.rows(0, 2);
        let target_owned = target_slice.clone_owned();

        let (in_slice, grad_slice) =
            compute_prediction_gradients(&pred_in, &prediction, &target_slice, 3, 2, 0.1);
        let (in_owned, grad_owned) =
            compute_prediction_gradients(&pred_in, &prediction, &target_owned, 3, 2, 0.1);

        let mut in_into = DVector::zeros(3);
        let mut grad_into = DVector::zeros(2);
        let ((), metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            compute_prediction_gradients_into(
                &pred_in,
                &prediction,
                &target_slice,
                0.1,
                &mut in_into,
                &mut grad_into,
            );
        });
        assert_eq!(metrics.allocations, 0);
        assert_eq!(in_slice, in_owned);
        assert_eq!(grad_slice, grad_owned);
        assert_eq!(in_slice, in_into);
        assert_eq!(grad_slice, grad_into);
    }
}
