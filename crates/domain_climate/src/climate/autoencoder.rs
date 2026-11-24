//!This module defines the autoencoder architecture for the CERA framework.
use rand::Rng;

use crate::climate::scratch::{AutoencoderScratch, ConvLayerScratch};
use crate::climate::tensor_ops::{conv1d, try_conv1d_into};
use crate::error::ClimateError;
use nalgebra::{DMatrix, DVector, Dyn, Matrix, Storage};
use pure_math::pure_math::analysis::optimization::{
    ModelOptimizer as Optimizer, OptimizationError, ParamType,
};

/// A trait representing the Autoencoder model interface.
pub trait AutoencoderModel {
    /// Encodes the input data into a latent representation.
    #[verified_engine::verified(alloc_ok = "trait convenience method returns owned latent matrix")]
    fn encode<S: Storage<f32, Dyn, Dyn>>(&self, input: &Matrix<f32, Dyn, Dyn, S>) -> DMatrix<f32>;

    /// Performs a forward pass through the autoencoder.
    #[verified_engine::verified(
        alloc_ok = "trait convenience method returns owned latent and reconstruction matrices"
    )]
    fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
    ) -> (DMatrix<f32>, DMatrix<f32>);
}

#[path = "autoencoder_kernels.rs"]
mod kernels;

use kernels::accumulate_conv_layer_grads;
pub use kernels::leaky_relu;
pub(crate) use kernels::{dense_vec_into, outer_grad_and_backprop};

/// A single layer for the Encoder or Decoder, consisting of a convolution and activation.
pub struct ConvLayer {
    /// The convolution kernel matrix.
    pub kernel: DMatrix<f32>,
    /// The bias vector.
    pub bias: DVector<f32>,
}

impl ConvLayer {
    /// Creates a new convolutional layer with random initialization.
    #[verified_engine::verified(alloc_ok = "initializes kernel and bias tensors")]
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        let mut rng = oxidize_core::rng::OxidizeRng::default();
        let fan_sum = in_channels.saturating_add(out_channels).max(1) as f32;
        let limit = (6.0 / fan_sum).sqrt();
        let kernel = DMatrix::from_fn(out_channels, in_channels, |_, _| {
            (rng.r#gen::<f32>() * 2.0 - 1.0) * limit
        });
        let bias = DVector::from_fn(out_channels, |_, _| {
            (rng.r#gen::<f32>() * 2.0 - 1.0) * 0.1 * limit
        });
        Self { kernel, bias }
    }

    /// Zero-allocation analytical backward pass and parameter update writing input gradients into
    /// `grad_input_out` using pre-allocated [`ConvLayerScratch`].
    #[verified_engine::verified]
    pub fn backward_update_into<O: Optimizer<f32>>(
        &mut self,
        input: &DMatrix<f32>,
        loss_grad: &DMatrix<f32>,
        optimizer: &mut O,
        layer_idx: usize,
        scratch: &mut ConvLayerScratch,
        grad_input_out: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        let n_samples = input.nrows();
        let in_ch = self.kernel.ncols();
        let out_ch = self.kernel.nrows();
        if n_samples == 0 || loss_grad.nrows() == 0 {
            grad_input_out.fill(0.0);
            return Ok(());
        }
        if input.ncols() != in_ch || loss_grad.shape() != (n_samples, out_ch) {
            return Err(ClimateError::ShapeMismatch {
                context: "ConvLayer::backward_update_into input/loss_grad shape mismatch.",
                expected: (n_samples, out_ch),
                actual: loss_grad.shape(),
            });
        }
        if grad_input_out.shape() != (n_samples, in_ch)
            || scratch.grad_w.shape() != (out_ch, in_ch)
            || scratch.grad_b.len() != out_ch
        {
            return Err(ClimateError::ShapeMismatch {
                context: "ConvLayer::backward_update_into scratch or output shape mismatch.",
                expected: (n_samples, in_ch),
                actual: grad_input_out.shape(),
            });
        }
        accumulate_conv_layer_grads(&self.kernel, input, loss_grad, scratch, grad_input_out);
        if scratch.grad_w.iter().all(|v| v.is_finite())
            && scratch.grad_b.iter().all(|v| v.is_finite())
        {
            optimizer.update_matrix(
                (layer_idx, ParamType::Weight),
                &mut self.kernel,
                &scratch.grad_w,
            )?;
            optimizer.update_vector(
                (layer_idx, ParamType::Bias),
                &mut self.bias,
                &scratch.grad_b,
            )?;
        }
        Ok(())
    }

    /// Performs an analytical backward pass and parameter update for the convolutional layer.
    #[verified_engine::verified(
        alloc_ok = "convenience backward_update allocates gradient tensors"
    )]
    pub fn backward_update<O: Optimizer<f32>>(
        &mut self,
        input: &DMatrix<f32>,
        loss_grad: &DMatrix<f32>,
        optimizer: &mut O,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>, OptimizationError> {
        let n_samples = input.nrows();
        let in_ch = self.kernel.ncols();
        if n_samples == 0 || loss_grad.nrows() == 0 {
            return Ok(DMatrix::zeros(0, in_ch));
        }
        let mut scratch = ConvLayerScratch::new(in_ch, self.kernel.nrows());
        let mut grad_input = DMatrix::zeros(n_samples, in_ch);
        match self.backward_update_into(
            input,
            loss_grad,
            optimizer,
            layer_idx,
            &mut scratch,
            &mut grad_input,
        ) {
            Ok(()) => Ok(grad_input),
            Err(ClimateError::Optimization(err)) => Err(err),
            Err(_) => Err(OptimizationError::ConversionError),
        }
    }
}

/// The encoder component of the autoencoder.
pub struct Encoder {
    /// The stack of convolutional layers.
    pub layers: [ConvLayer; 3],
}

impl Encoder {
    /// Creates a new encoder with architecture: `in_channels -> 64 -> 64 -> latent_channels`.
    #[verified_engine::verified(alloc_ok = "initializes 3-layer Encoder weights")]
    pub fn new(in_channels: usize, latent_channels: usize) -> Self {
        let layers = [
            ConvLayer::new(in_channels, 64),
            ConvLayer::new(64, 64),
            ConvLayer::new(64, latent_channels),
        ];
        Self::new_from_layers(layers)
    }

    /// Creates a new encoder with custom layers.
    #[verified_engine::verified]
    pub fn new_from_layers(layers: [ConvLayer; 3]) -> Self {
        Self { layers }
    }

    /// Encodes `input` into `latent_out` without heap allocations using pre-allocated hidden buffers.
    #[verified_engine::verified]
    pub fn forward_into<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
        h0: &mut DMatrix<f32>,
        h1: &mut DMatrix<f32>,
        latent_out: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        try_conv1d_into(input, &self.layers[0].kernel, &self.layers[0].bias, h0)?;
        leaky_relu(h0, 0.01);
        try_conv1d_into(h0, &self.layers[1].kernel, &self.layers[1].bias, h1)?;
        leaky_relu(h1, 0.01);
        try_conv1d_into(h1, &self.layers[2].kernel, &self.layers[2].bias, latent_out)
    }

    /// Encodes the input data into a latent representation.
    #[verified_engine::verified(
        alloc_ok = "convenience forward allocates hidden and output matrices"
    )]
    pub fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
    ) -> DMatrix<f32> {
        let mut x = input.clone_owned();
        for (i, layer) in self.layers.iter().enumerate() {
            x = conv1d(&x, &layer.kernel, &layer.bias);
            if i < self.layers.len() - 1 {
                leaky_relu(&mut x, 0.01);
            }
        }
        x
    }
}

/// The decoder component of the autoencoder.
pub struct Decoder {
    /// The stack of convolutional layers.
    pub layers: [ConvLayer; 3],
}

impl Decoder {
    /// Creates a new decoder with architecture: `latent_channels -> 64 -> 64 -> out_channels`.
    #[verified_engine::verified(alloc_ok = "initializes 3-layer Decoder weights")]
    pub fn new(latent_channels: usize, out_channels: usize) -> Self {
        let layers = [
            ConvLayer::new(latent_channels, 64),
            ConvLayer::new(64, 64),
            ConvLayer::new(64, out_channels),
        ];
        Self::new_from_layers(layers)
    }

    /// Creates a new decoder with custom layers.
    #[verified_engine::verified]
    pub fn new_from_layers(layers: [ConvLayer; 3]) -> Self {
        Self { layers }
    }

    /// Decodes `latent` into `recon_out` without heap allocations using pre-allocated hidden buffers.
    #[verified_engine::verified]
    pub fn forward_into<S: Storage<f32, Dyn, Dyn>>(
        &self,
        latent: &Matrix<f32, Dyn, Dyn, S>,
        h0: &mut DMatrix<f32>,
        h1: &mut DMatrix<f32>,
        recon_out: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        try_conv1d_into(latent, &self.layers[0].kernel, &self.layers[0].bias, h0)?;
        leaky_relu(h0, 0.01);
        try_conv1d_into(h0, &self.layers[1].kernel, &self.layers[1].bias, h1)?;
        leaky_relu(h1, 0.01);
        try_conv1d_into(h1, &self.layers[2].kernel, &self.layers[2].bias, recon_out)
    }

    /// Reconstructs the input data from the latent representation.
    #[verified_engine::verified(
        alloc_ok = "convenience forward allocates hidden and output matrices"
    )]
    pub fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        latent_representation: &Matrix<f32, Dyn, Dyn, S>,
    ) -> DMatrix<f32> {
        let mut x = latent_representation.clone_owned();
        for (i, layer) in self.layers.iter().enumerate() {
            x = conv1d(&x, &layer.kernel, &layer.bias);
            if i < self.layers.len() - 1 {
                leaky_relu(&mut x, 0.01);
            }
        }
        x
    }
}

/// The autoencoder model for the CERA framework.
pub struct Autoencoder {
    /// The encoder component.
    pub encoder: Encoder,
    /// The decoder component.
    pub decoder: Decoder,
}

impl Autoencoder {
    /// Creates a new autoencoder with `in_channels` and `latent_channels`.
    #[verified_engine::verified(alloc_ok = "initializes Autoencoder encoder and decoder weights")]
    pub fn new(in_channels: usize, latent_channels: usize) -> Self {
        let encoder = Encoder::new(in_channels, latent_channels);
        let decoder = Decoder::new(latent_channels, in_channels);
        Self::new_from_components(encoder, decoder)
    }

    /// Creates a new autoencoder from existing encoder and decoder components.
    #[verified_engine::verified]
    pub fn new_from_components(encoder: Encoder, decoder: Decoder) -> Self {
        Self { encoder, decoder }
    }

    /// Zero-allocation latent encoding using [`AutoencoderScratch`].
    #[verified_engine::verified]
    pub fn encode_into<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
        scratch: &mut AutoencoderScratch,
        latent_out: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        self.encoder
            .forward_into(input, &mut scratch.enc_h0, &mut scratch.enc_h1, latent_out)
    }

    /// Zero-allocation full forward pass using [`AutoencoderScratch`].
    #[verified_engine::verified]
    pub fn forward_into<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
        scratch: &mut AutoencoderScratch,
        latent_out: &mut DMatrix<f32>,
        recon_out: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        self.encode_into(input, scratch, latent_out)?;
        self.decoder.forward_into(
            latent_out,
            &mut scratch.dec_h0,
            &mut scratch.dec_h1,
            recon_out,
        )
    }

    #[verified_engine::verified]
    fn forward_vec_into_scratch(&self, x: &DVector<f32>, scratch: &mut AutoencoderScratch) {
        scratch.activations[0].copy_from(x);
        for i in 0..6 {
            let layer = if i < 3 {
                &self.encoder.layers[i]
            } else {
                &self.decoder.layers[i - 3]
            };
            let (prev_slice, rest_slice) = scratch.activations.split_at_mut(i + 1);
            dense_vec_into(
                &layer.kernel,
                &layer.bias,
                &prev_slice[i],
                &mut scratch.zs[i],
                &mut rest_slice[0],
                i != 2 && i != 5,
            );
        }
    }

    /// Zero-allocation 6-layer analytical backpropagation and parameter update using [`AutoencoderScratch`].
    #[verified_engine::verified]
    pub fn backward_update_into(
        &mut self,
        x: &DVector<f32>,
        loss_grad: &DVector<f32>,
        optimizer: &mut dyn Optimizer<f32>,
        scratch: &mut AutoencoderScratch,
    ) -> Result<(), OptimizationError> {
        self.forward_vec_into_scratch(x, scratch);
        let out_len = loss_grad.len();
        scratch.d_z.rows_mut(0, out_len).copy_from(loss_grad);
        for i in (0..6).rev() {
            let layer = if i < 3 {
                &mut self.encoder.layers[i]
            } else {
                &mut self.decoder.layers[i - 3]
            };
            let cur_out = layer.kernel.nrows();
            let cur_in = layer.kernel.ncols();
            let z_prev = if i > 0 && i - 1 != 2 {
                Some(&scratch.zs[i - 1])
            } else {
                None
            };
            let next_slice = if i > 0 {
                &mut scratch.d_next.as_mut_slice()[..cur_in]
            } else {
                &mut [][..]
            };
            outer_grad_and_backprop(
                &layer.kernel,
                &scratch.d_z.as_slice()[..cur_out],
                &scratch.activations[i],
                z_prev,
                &mut scratch.d_w[i],
                next_slice,
            );
            optimizer.update_matrix((i, ParamType::Weight), &mut layer.kernel, &scratch.d_w[i])?;
            scratch.zs[i].copy_from(&scratch.d_z.rows(0, cur_out));
            optimizer.update_vector((i, ParamType::Bias), &mut layer.bias, &scratch.zs[i])?;
            if i > 0 {
                scratch.d_z.as_mut_slice()[..cur_in]
                    .copy_from_slice(&scratch.d_next.as_slice()[..cur_in]);
            }
        }
        Ok(())
    }

    /// Performs a forward pass through the autoencoder.
    #[verified_engine::verified(
        alloc_ok = "convenience forward allocates latent and reconstruction matrices"
    )]
    pub fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
    ) -> (DMatrix<f32>, DMatrix<f32>) {
        let latent = self.encoder.forward(input);
        let reconstruction = self.decoder.forward(&latent);
        (latent, reconstruction)
    }
}

impl AutoencoderModel for Autoencoder {
    #[verified_engine::verified(alloc_ok = "trait convenience method allocates latent matrix")]
    fn encode<S: Storage<f32, Dyn, Dyn>>(&self, input: &Matrix<f32, Dyn, Dyn, S>) -> DMatrix<f32> {
        self.encoder.forward(input)
    }

    #[verified_engine::verified(
        alloc_ok = "trait convenience method allocates latent and reconstruction matrices"
    )]
    fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
    ) -> (DMatrix<f32>, DMatrix<f32>) {
        let latent = self.encoder.forward(input);
        let reconstruction = self.decoder.forward(&latent);
        (latent, reconstruction)
    }
}

impl pure_math::pure_math::analysis::optimization::Trainable<f32> for Autoencoder {
    fn forward(&self, x: &nalgebra::DVector<f32>) -> nalgebra::DVector<f32> {
        let x_mat = nalgebra::DMatrix::from_row_slice(1, x.len(), x.as_slice());
        let (_, recon) = AutoencoderModel::forward(self, &x_mat);
        nalgebra::DVector::from_column_slice(recon.as_slice())
    }

    fn backward_update(
        &mut self,
        x: &nalgebra::DVector<f32>,
        loss_grad: &nalgebra::DVector<f32>,
        optimizer: &mut dyn Optimizer<f32>,
    ) -> Result<(), OptimizationError> {
        let mut scratch = AutoencoderScratch::from_autoencoder(1, self);
        self.backward_update_into(x, loss_grad, optimizer, &mut scratch)
    }
}

#[cfg(test)]
#[path = "autoencoder_tests.rs"]
mod tests;
