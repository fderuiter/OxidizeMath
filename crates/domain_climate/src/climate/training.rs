//! This module handles the training process for the CERA model.

use crate::climate::autoencoder::AutoencoderModel;
use crate::climate::cera::Cera;
use crate::climate::loss::{cera_loss, earth_movers_distance, mse_loss};
use crate::climate::predictor::PredictorModel;
use nalgebra::{DMatrix, Dyn, Matrix, Storage};
use pure_math::pure_math::analysis::optimization::{ModelOptimizer as Optimizer, SGD};

use pure_math::pure_math::analysis::optimization::Trainable;

/// A trainer for the CERA model.
#[derive(Debug, PartialEq)]
pub struct CeraTrainer<
    'a,
    A: AutoencoderModel + Trainable<f32>,
    P: PredictorModel + Trainable<f32>,
    O: Optimizer<f32>,
> {
    /// Mutable reference to the [`Cera`] model being optimized.
    pub model: &'a mut Cera<A, P>,
    /// The optimizer strategy (e.g., SGD, Adam).
    pub optimizer: O,
}

impl<'a, A: AutoencoderModel + Trainable<f32>, P: PredictorModel + Trainable<f32>>
    CeraTrainer<'a, A, P, SGD<f32>>
{
    /// Creates a new CeraTrainer with default SGD optimizer.
    #[verified_engine::verified]
    pub fn new(model: &'a mut Cera<A, P>) -> Self {
        // Initialize optimizer from config.
        let lr = model.config.learning_rate;
        let optimizer = SGD::new(lr);
        Self { model, optimizer }
    }
}

/// Computes analytical Mean Squared Error gradients and average input for reconstruction batches.
#[verified_engine::verified]
fn compute_reconstruction_gradients<
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
) -> (nalgebra::DVector<f32>, nalgebra::DVector<f32>) {
    let input_rows = control_input.nrows();
    let mut ae_input = nalgebra::DVector::zeros(in_channels);
    let mut ae_loss_grad = nalgebra::DVector::zeros(in_channels);
    if input_rows == 0 || in_channels == 0 {
        return (ae_input, ae_loss_grad);
    }

    let scale = 1.0 / (input_rows * in_channels) as f32;
    for r in 0..input_rows {
        for c in 0..in_channels {
            ae_input[c] += 0.5 * (control_input[(r, c)] + warm_input[(r, c)]);
            ae_loss_grad[c] += scale
                * ((control_recon[(r, c)] - control_input[(r, c)])
                    + (warm_recon[(r, c)] - warm_input[(r, c)]));
        }
    }
    ae_input /= input_rows as f32;
    (ae_input, ae_loss_grad)
}

/// Computes analytical Mean Squared Error gradients and average input for prediction batches.
#[verified_engine::verified]
fn compute_prediction_gradients<
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
) -> (nalgebra::DVector<f32>, nalgebra::DVector<f32>) {
    let batch_size = prediction.nrows();
    let mut pred_input_vec = nalgebra::DVector::zeros(pred_in_dim);
    let mut pred_loss_grad = nalgebra::DVector::zeros(output_size);
    if batch_size == 0 || pred_in_dim == 0 || output_size == 0 {
        return (pred_input_vec, pred_loss_grad);
    }

    for r in 0..batch_size {
        for c in 0..pred_in_dim {
            pred_input_vec[c] += predictor_input[(r, c)];
        }
    }
    pred_input_vec /= batch_size as f32;

    let pred_scale = if lambda_pred.is_finite() {
        lambda_pred * 2.0 / (batch_size * output_size) as f32
    } else {
        0.0
    };
    for r in 0..batch_size {
        for c in 0..output_size {
            pred_loss_grad[c] += pred_scale * (prediction[(r, c)] - targets[(r, c)]);
        }
    }
    (pred_input_vec, pred_loss_grad)
}

/// Clips the L2 norm of a gradient vector if it exceeds `max_norm`.
#[verified_engine::verified]
fn clip_gradient_norm(grad: &mut nalgebra::DVector<f32>, max_norm: f32) {
    let norm = grad.norm();
    if norm > max_norm && norm.is_finite() && norm > 0.0 {
        *grad *= max_norm / norm;
    }
}

impl<
    'a,
    A: AutoencoderModel + Trainable<f32>,
    P: PredictorModel + Trainable<f32>,
    O: Optimizer<f32>,
> CeraTrainer<'a, A, P, O>
{
    /// Creates a new CeraTrainer with a custom optimizer.
    #[verified_engine::verified]
    pub fn new_with_optimizer(model: &'a mut Cera<A, P>, optimizer: O) -> Self {
        Self { model, optimizer }
    }

    /// Performs an analytical backpropagation and optimization step.
    ///
    /// Updates the autoencoder and predictor models using analytical Mean Squared Error
    /// loss gradients. Validates gradient values for finiteness before updating parameters.
    #[verified_engine::verified]
    fn optimizer_step(
        &mut self,
        ae_input: &nalgebra::DVector<f32>,
        ae_loss_grad: &nalgebra::DVector<f32>,
        pred_input: &nalgebra::DVector<f32>,
        pred_loss_grad: &nalgebra::DVector<f32>,
    ) -> Result<(), pure_math::pure_math::analysis::optimization::OptimizationError> {
        if ae_input.iter().all(|v| v.is_finite()) && ae_loss_grad.iter().all(|v| v.is_finite()) {
            self.model
                .autoencoder
                .backward_update(ae_input, ae_loss_grad, &mut self.optimizer)?;
        }

        if pred_input.iter().all(|v| v.is_finite()) && pred_loss_grad.iter().all(|v| v.is_finite())
        {
            self.model.predictor.backward_update(
                pred_input,
                pred_loss_grad,
                &mut self.optimizer,
            )?;
        }

        Ok(())
    }

    /// Trains the CERA model on synthetic data.
    ///
    /// # Arguments
    ///
    /// * `control_inputs` - Input data for the control climate.
    /// * `control_targets` - Target outputs for the control climate.
    /// * `warm_inputs` - Input data for the warm climate.
    ///
    /// # Returns
    ///
    /// A vector of average loss values per epoch.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_climate::__doctest_facade as math_explorer;
    /// use math_explorer::climate::cera::Cera;
    /// use math_explorer::climate::config::CeraConfig;
    /// use math_explorer::climate::training::CeraTrainer;
    /// use math_explorer::nalgebra::DMatrix;
    ///
    /// let config = CeraConfig {
    ///     learning_rate: 0.01,
    ///     lambda_pred: 0.2,
    ///     lambda_emd: 0.1,
    ///     epochs: 2,
    ///     batch_size: 2,
    ///     in_channels: 2,
    ///     latent_channels: 3,
    ///     aligned_channels: 2,
    ///     num_levels: 4,
    ///     output_size: 2,
    /// };
    /// let mut model = Cera::new(config).unwrap();
    /// let control_inputs = DMatrix::from_element(8, 2, 0.5_f32);
    /// let control_targets = DMatrix::from_element(2, 2, 0.25_f32);
    /// let warm_inputs = DMatrix::from_element(8, 2, 0.8_f32);
    ///
    /// let mut trainer = CeraTrainer::new(&mut model);
    /// let losses = trainer.train(&control_inputs, &control_targets, &warm_inputs);
    /// assert_eq!(losses.len(), 2);
    /// assert!(losses.iter().all(|loss| loss.is_finite()));
    /// ```
    #[verified_engine::verified]
    pub fn train(
        &mut self,
        control_inputs: &DMatrix<f32>,
        control_targets: &DMatrix<f32>,
        warm_inputs: &DMatrix<f32>,
    ) -> Vec<f32> {
        let batch_size = self.model.config.batch_size;
        let num_levels = self.model.config.num_levels;
        let aligned_channels = self.model.config.aligned_channels;

        // Ensure we don't divide by zero if inputs are empty, though new() checks dimensions.
        if control_inputs.nrows() == 0 || num_levels == 0 || batch_size == 0 {
            return Vec::new();
        }

        let n_samples = control_inputs.nrows() / num_levels;
        let n_batches = n_samples / batch_size;
        let mut epoch_losses = Vec::with_capacity(self.model.config.epochs);

        for epoch in 0..self.model.config.epochs {
            let mut total_loss = 0.0;
            for i in 0..n_batches {
                // --- Create batches ---
                let input_start = i * batch_size * num_levels;
                let input_rows = batch_size * num_levels;
                let control_input_batch = control_inputs.rows(input_start, input_rows);
                let warm_input_batch = warm_inputs.rows(input_start, input_rows);

                let target_start = i * batch_size;
                let control_target_batch = control_targets.rows(target_start, batch_size);

                // --- Forward pass ---
                let (control_latent, control_recon) =
                    AutoencoderModel::forward(&self.model.autoencoder, &control_input_batch);
                let (warm_latent, warm_recon) =
                    AutoencoderModel::forward(&self.model.autoencoder, &warm_input_batch);

                // --- Reshape and predict ---
                let control_aligned_latent = control_latent.columns(0, aligned_channels);
                // Reuse the method from Cera
                let predictor_input = self
                    .model
                    .reshape_for_predictor(&control_aligned_latent, batch_size);
                let prediction = PredictorModel::forward(&self.model.predictor, &predictor_input);

                // --- Calculate losses ---
                let recon_loss_control = mse_loss(&control_input_batch, &control_recon);
                let recon_loss_warm = mse_loss(&warm_input_batch, &warm_recon);
                let reconstruction_loss = (recon_loss_control + recon_loss_warm) / 2.0;

                let prediction_loss = mse_loss(&control_target_batch, &prediction);

                let warm_aligned_latent = warm_latent.columns(0, aligned_channels);
                let emd_loss = earth_movers_distance(&control_aligned_latent, &warm_aligned_latent);

                let loss = cera_loss(
                    reconstruction_loss,
                    prediction_loss,
                    emd_loss,
                    self.model.config.lambda_pred,
                    self.model.config.lambda_emd,
                );

                // --- Calculate analytical loss gradients ---
                let in_channels = self.model.config.in_channels;
                let (ae_input, ae_loss_grad) = compute_reconstruction_gradients(
                    &control_input_batch,
                    &control_recon,
                    &warm_input_batch,
                    &warm_recon,
                    in_channels,
                );

                let pred_in_dim = num_levels * aligned_channels;
                let output_size = self.model.config.output_size;
                let (pred_input_vec, pred_loss_grad) = compute_prediction_gradients(
                    &predictor_input,
                    &prediction,
                    &control_target_batch,
                    pred_in_dim,
                    output_size,
                    self.model.config.lambda_pred,
                );

                let mut pred_loss_grad = pred_loss_grad;
                let mut ae_loss_grad = ae_loss_grad;
                clip_gradient_norm(&mut ae_loss_grad, 1.0);
                clip_gradient_norm(&mut pred_loss_grad, 1e-4);

                // --- Backward pass and optimization ---
                let _ =
                    self.optimizer_step(&ae_input, &ae_loss_grad, &pred_input_vec, &pred_loss_grad);
                total_loss += loss;
            }
            if n_batches > 0 {
                let avg_loss = total_loss / n_batches as f32;
                println!("Epoch {}, Average Loss: {}", epoch, avg_loss);
                epoch_losses.push(avg_loss);
            }
        }
        epoch_losses
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    #[verified_engine::verified]
    fn test_compute_reconstruction_gradients_owned_and_sliced() {
        let control_input =
            DMatrix::from_row_slice(4, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let control_recon = DMatrix::from_row_slice(2, 2, &[1.1, 2.1, 2.9, 3.9]);
        let warm_input = DMatrix::from_row_slice(4, 2, &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        let warm_recon = DMatrix::from_row_slice(2, 2, &[1.9, 3.2, 4.1, 4.8]);

        let control_slice = control_input.rows(0, 2);
        let warm_slice = warm_input.rows(0, 2);

        // Test with borrowed slices
        let (in_slice, grad_slice) = compute_reconstruction_gradients(
            &control_slice,
            &control_recon,
            &warm_slice,
            &warm_recon,
            2,
        );

        // Test with owned matrices
        let control_owned = control_slice.clone_owned();
        let warm_owned = warm_slice.clone_owned();
        let (in_owned, grad_owned) = compute_reconstruction_gradients(
            &control_owned,
            &control_recon,
            &warm_owned,
            &warm_recon,
            2,
        );

        assert_eq!(in_slice, in_owned);
        assert_eq!(grad_slice, grad_owned);
    }

    #[test]
    #[verified_engine::verified]
    fn test_compute_prediction_gradients_owned_and_sliced() {
        let pred_in = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let prediction = DMatrix::from_row_slice(2, 2, &[0.5, 1.5, 2.5, 3.5]);
        let full_targets = DMatrix::from_row_slice(4, 2, &[0.6, 1.4, 2.4, 3.6, 9.0, 9.0, 9.0, 9.0]);
        let target_slice = full_targets.rows(0, 2);
        let target_owned = target_slice.clone_owned();

        let (in_slice, grad_slice) =
            compute_prediction_gradients(&pred_in, &prediction, &target_slice, 3, 2, 0.1);

        let (in_owned, grad_owned) =
            compute_prediction_gradients(&pred_in, &prediction, &target_owned, 3, 2, 0.1);

        assert_eq!(in_slice, in_owned);
        assert_eq!(grad_slice, grad_owned);
    }
}
