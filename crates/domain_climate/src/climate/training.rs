//! This module handles the training process for the CERA model.

#[path = "training_gradients.rs"]
mod gradients;

use crate::climate::autoencoder::{Autoencoder, AutoencoderModel};
use crate::climate::cera::Cera;
use crate::climate::config::CeraConfig;
use crate::climate::loss::{
    cera_loss, earth_movers_distance, mse_loss, try_earth_movers_distance_into, try_mse_loss,
};
use crate::climate::predictor::{Predictor, PredictorModel};
use crate::climate::scratch::CeraTrainingScratch;
use crate::error::ClimateError;
use gradients::{
    clip_gradient_norm, compute_prediction_gradients, compute_prediction_gradients_into,
    compute_reconstruction_gradients, compute_reconstruction_gradients_into,
};
use nalgebra::{DMatrix, DVector, Dyn, Matrix, Storage};
use pure_math::pure_math::analysis::optimization::{ModelOptimizer as Optimizer, SGD, Trainable};

/// A trainer for the CERA model.
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
        let lr = model.config.learning_rate;
        let optimizer = SGD::new(lr);
        Self { model, optimizer }
    }
}

#[verified_engine::verified]
fn validate_train_step_shapes<
    S1: Storage<f32, Dyn, Dyn>,
    S2: Storage<f32, Dyn, Dyn>,
    S3: Storage<f32, Dyn, Dyn>,
>(
    cfg: &CeraConfig,
    control_input_batch: &Matrix<f32, Dyn, Dyn, S1>,
    control_target_batch: &Matrix<f32, Dyn, Dyn, S2>,
    warm_input_batch: &Matrix<f32, Dyn, Dyn, S3>,
) -> Result<(), ClimateError> {
    let expected_rows =
        cfg.batch_size
            .checked_mul(cfg.num_levels)
            .ok_or(ClimateError::DimensionOverflow {
                context: "train_step_into batch_size * num_levels",
            })?;
    if control_input_batch.shape() != (expected_rows, cfg.in_channels)
        || warm_input_batch.shape() != (expected_rows, cfg.in_channels)
    {
        return Err(ClimateError::ShapeMismatch {
            context: "Input batch shape mismatch in train_step_into.",
            expected: (expected_rows, cfg.in_channels),
            actual: control_input_batch.shape(),
        });
    }
    if control_target_batch.shape() != (cfg.batch_size, cfg.output_size) {
        return Err(ClimateError::ShapeMismatch {
            context: "Target batch shape mismatch in train_step_into.",
            expected: (cfg.batch_size, cfg.output_size),
            actual: control_target_batch.shape(),
        });
    }
    Ok(())
}

#[verified_engine::verified]
fn forward_and_loss_into<
    S1: Storage<f32, Dyn, Dyn>,
    S2: Storage<f32, Dyn, Dyn>,
    S3: Storage<f32, Dyn, Dyn>,
>(
    model: &Cera<Autoencoder, Predictor>,
    control_input_batch: &Matrix<f32, Dyn, Dyn, S1>,
    control_target_batch: &Matrix<f32, Dyn, Dyn, S2>,
    warm_input_batch: &Matrix<f32, Dyn, Dyn, S3>,
    scratch: &mut CeraTrainingScratch,
) -> Result<f32, ClimateError> {
    let cfg = &model.config;
    model.autoencoder.forward_into(
        control_input_batch,
        &mut scratch.ae_scratch,
        &mut scratch.control_latent,
        &mut scratch.control_recon,
    )?;
    model.autoencoder.forward_into(
        warm_input_batch,
        &mut scratch.ae_scratch,
        &mut scratch.warm_latent,
        &mut scratch.warm_recon,
    )?;
    model.reshape_for_predictor_into(
        &scratch.control_latent,
        cfg.batch_size,
        &mut scratch.predictor_input,
    )?;
    model.predictor.forward_into(
        &scratch.predictor_input,
        &mut scratch.pred_scratch,
        &mut scratch.prediction,
    )?;

    let recon_loss = 0.5
        * (try_mse_loss(control_input_batch, &scratch.control_recon)?
            + try_mse_loss(warm_input_batch, &scratch.warm_recon)?);
    let pred_loss = try_mse_loss(control_target_batch, &scratch.prediction)?;
    let ctrl_aligned = scratch.control_latent.columns(0, cfg.aligned_channels);
    let warm_aligned = scratch.warm_latent.columns(0, cfg.aligned_channels);
    let emd =
        try_earth_movers_distance_into(&ctrl_aligned, &warm_aligned, &mut scratch.emd_scratch)?;
    Ok(cera_loss(
        recon_loss,
        pred_loss,
        emd,
        cfg.lambda_pred,
        cfg.lambda_emd,
    ))
}

#[verified_engine::verified]
fn validate_train_into_batches(
    cfg: &CeraConfig,
    control_inputs: &DMatrix<f32>,
    control_targets: &DMatrix<f32>,
    warm_inputs: &DMatrix<f32>,
) -> Result<Option<(usize, usize)>, ClimateError> {
    if control_inputs.nrows() == 0 || cfg.num_levels == 0 || cfg.batch_size == 0 || cfg.epochs == 0
    {
        return Ok(None);
    }
    let batch_rows =
        cfg.batch_size
            .checked_mul(cfg.num_levels)
            .ok_or(ClimateError::DimensionOverflow {
                context: "train_into batch_size * num_levels",
            })?;
    let n_batches = control_inputs.nrows() / batch_rows;
    if n_batches == 0 {
        return Ok(None);
    }
    let required_rows = n_batches * batch_rows;
    let required_target_rows = n_batches * cfg.batch_size;
    if warm_inputs.nrows() < required_rows
        || warm_inputs.ncols() != cfg.in_channels
        || control_inputs.ncols() != cfg.in_channels
    {
        return Err(ClimateError::ShapeMismatch {
            context: "Warm or control inputs shape mismatch in train_into.",
            expected: (required_rows, cfg.in_channels),
            actual: warm_inputs.shape(),
        });
    }
    if control_targets.nrows() < required_target_rows || control_targets.ncols() != cfg.output_size
    {
        return Err(ClimateError::ShapeMismatch {
            context: "Control targets shape mismatch in train_into.",
            expected: (required_target_rows, cfg.output_size),
            actual: control_targets.shape(),
        });
    }
    Ok(Some((batch_rows, n_batches)))
}

impl<'a, O: Optimizer<f32>> CeraTrainer<'a, Autoencoder, Predictor, O> {
    /// Executes a single batch forward, MSE + EMD loss evaluation, and analytical backpropagation
    /// step with zero dynamic heap allocations using [`CeraTrainingScratch`].
    #[verified_engine::verified]
    pub fn train_step_into<
        S1: Storage<f32, Dyn, Dyn>,
        S2: Storage<f32, Dyn, Dyn>,
        S3: Storage<f32, Dyn, Dyn>,
    >(
        &mut self,
        control_input_batch: &Matrix<f32, Dyn, Dyn, S1>,
        control_target_batch: &Matrix<f32, Dyn, Dyn, S2>,
        warm_input_batch: &Matrix<f32, Dyn, Dyn, S3>,
        scratch: &mut CeraTrainingScratch,
    ) -> Result<f32, ClimateError> {
        validate_train_step_shapes(
            &self.model.config,
            control_input_batch,
            control_target_batch,
            warm_input_batch,
        )?;
        let loss = forward_and_loss_into(
            self.model,
            control_input_batch,
            control_target_batch,
            warm_input_batch,
            scratch,
        )?;
        compute_reconstruction_gradients_into(
            control_input_batch,
            &scratch.control_recon,
            warm_input_batch,
            &scratch.warm_recon,
            &mut scratch.ae_input,
            &mut scratch.ae_loss_grad,
        );
        compute_prediction_gradients_into(
            &scratch.predictor_input,
            &scratch.prediction,
            control_target_batch,
            self.model.config.lambda_pred,
            &mut scratch.pred_input_vec,
            &mut scratch.pred_loss_grad,
        );
        clip_gradient_norm(&mut scratch.ae_loss_grad, 1.0);
        clip_gradient_norm(&mut scratch.pred_loss_grad, 1e-4);

        if scratch.ae_input.iter().all(|v| v.is_finite())
            && scratch.ae_loss_grad.iter().all(|v| v.is_finite())
        {
            self.model.autoencoder.backward_update_into(
                &scratch.ae_input,
                &scratch.ae_loss_grad,
                &mut self.optimizer,
                &mut scratch.ae_scratch,
            )?;
        }
        if scratch.pred_input_vec.iter().all(|v| v.is_finite())
            && scratch.pred_loss_grad.iter().all(|v| v.is_finite())
        {
            self.model.predictor.backward_update_into(
                &scratch.pred_input_vec,
                &scratch.pred_loss_grad,
                &mut self.optimizer,
                &mut scratch.pred_scratch,
            )?;
        }
        Ok(loss)
    }

    /// Runs multi-epoch CERA training into caller-provided `epoch_losses_out` with zero dynamic
    /// heap allocations using [`CeraTrainingScratch`], returning the number of epochs recorded.
    #[verified_engine::verified]
    pub fn train_into(
        &mut self,
        control_inputs: &DMatrix<f32>,
        control_targets: &DMatrix<f32>,
        warm_inputs: &DMatrix<f32>,
        scratch: &mut CeraTrainingScratch,
        epoch_losses_out: &mut [f32],
    ) -> Result<usize, ClimateError> {
        let Some((batch_rows, n_batches)) = validate_train_into_batches(
            &self.model.config,
            control_inputs,
            control_targets,
            warm_inputs,
        )?
        else {
            return Ok(0);
        };
        let batch_size = self.model.config.batch_size;
        let max_epochs = self.model.config.epochs.min(epoch_losses_out.len());
        for loss_slot in epoch_losses_out.iter_mut().take(max_epochs) {
            let mut total_loss = 0.0_f32;
            for b in 0..n_batches {
                let input_start = b * batch_rows;
                let target_start = b * batch_size;
                let ctrl_batch = control_inputs.rows(input_start, batch_rows);
                let warm_batch = warm_inputs.rows(input_start, batch_rows);
                let tgt_batch = control_targets.rows(target_start, batch_size);
                total_loss +=
                    self.train_step_into(&ctrl_batch, &tgt_batch, &warm_batch, scratch)?;
            }
            *loss_slot = total_loss / n_batches as f32;
        }
        Ok(max_epochs)
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

    #[verified_engine::verified]
    fn optimizer_step(
        &mut self,
        ae_input: &DVector<f32>,
        ae_loss_grad: &DVector<f32>,
        pred_input: &DVector<f32>,
        pred_loss_grad: &DVector<f32>,
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

    #[verified_engine::verified(
        alloc_ok = "convenience batch helper allocates intermediate tensors"
    )]
    fn train_batch_allocating<
        S1: Storage<f32, Dyn, Dyn>,
        S2: Storage<f32, Dyn, Dyn>,
        S3: Storage<f32, Dyn, Dyn>,
    >(
        &mut self,
        control_input_batch: &Matrix<f32, Dyn, Dyn, S1>,
        control_target_batch: &Matrix<f32, Dyn, Dyn, S2>,
        warm_input_batch: &Matrix<f32, Dyn, Dyn, S3>,
        pred_in_dim: usize,
    ) -> f32 {
        let aligned_channels = self.model.config.aligned_channels;
        let batch_size = self.model.config.batch_size;
        let (control_latent, control_recon) =
            AutoencoderModel::forward(&self.model.autoencoder, control_input_batch);
        let (warm_latent, warm_recon) =
            AutoencoderModel::forward(&self.model.autoencoder, warm_input_batch);
        let control_aligned_latent = control_latent.columns(0, aligned_channels);
        let predictor_input = self
            .model
            .reshape_for_predictor(&control_aligned_latent, batch_size);
        let prediction = PredictorModel::forward(&self.model.predictor, &predictor_input);

        let recon_loss = 0.5
            * (mse_loss(control_input_batch, &control_recon)
                + mse_loss(warm_input_batch, &warm_recon));
        let prediction_loss = mse_loss(control_target_batch, &prediction);
        let warm_aligned_latent = warm_latent.columns(0, aligned_channels);
        let emd_loss = earth_movers_distance(&control_aligned_latent, &warm_aligned_latent);
        let loss = cera_loss(
            recon_loss,
            prediction_loss,
            emd_loss,
            self.model.config.lambda_pred,
            self.model.config.lambda_emd,
        );

        let (ae_input, mut ae_loss_grad) = compute_reconstruction_gradients(
            control_input_batch,
            &control_recon,
            warm_input_batch,
            &warm_recon,
            self.model.config.in_channels,
        );
        let (pred_input_vec, mut pred_loss_grad) = compute_prediction_gradients(
            &predictor_input,
            &prediction,
            control_target_batch,
            pred_in_dim,
            self.model.config.output_size,
            self.model.config.lambda_pred,
        );
        clip_gradient_norm(&mut ae_loss_grad, 1.0);
        clip_gradient_norm(&mut pred_loss_grad, 1e-4);
        let _ = self.optimizer_step(&ae_input, &ae_loss_grad, &pred_input_vec, &pred_loss_grad);
        loss
    }

    /// Trains the CERA model on synthetic data.
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
    #[verified_engine::verified(
        alloc_ok = "convenience train wrapper allocates per-epoch loss vector and batch tensors"
    )]
    pub fn train(
        &mut self,
        control_inputs: &DMatrix<f32>,
        control_targets: &DMatrix<f32>,
        warm_inputs: &DMatrix<f32>,
    ) -> Vec<f32> {
        let batch_size = self.model.config.batch_size;
        let num_levels = self.model.config.num_levels;
        let aligned_channels = self.model.config.aligned_channels;
        let Some(batch_rows) = batch_size.checked_mul(num_levels) else {
            return Vec::new();
        };
        if control_inputs.nrows() == 0 || batch_rows == 0 {
            return Vec::new();
        }
        let n_batches = (control_inputs.nrows() / batch_rows)
            .min(warm_inputs.nrows() / batch_rows)
            .min(control_targets.nrows() / batch_size);
        let Some(pred_in_dim) = num_levels.checked_mul(aligned_channels) else {
            return Vec::new();
        };
        if n_batches == 0 {
            return Vec::new();
        }

        let mut epoch_losses = Vec::with_capacity(self.model.config.epochs);
        for _epoch in 0..self.model.config.epochs {
            let mut total_loss = 0.0;
            for i in 0..n_batches {
                let input_start = i * batch_rows;
                let ctrl_batch = control_inputs.rows(input_start, batch_rows);
                let warm_batch = warm_inputs.rows(input_start, batch_rows);
                let tgt_batch = control_targets.rows(i * batch_size, batch_size);
                total_loss +=
                    self.train_batch_allocating(&ctrl_batch, &tgt_batch, &warm_batch, pred_in_dim);
            }
            epoch_losses.push(total_loss / n_batches as f32);
        }
        epoch_losses
    }
}

#[cfg(test)]
#[path = "training_tests.rs"]
mod tests;
