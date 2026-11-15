//! This module defines the autoencoder architecture for the CERA framework.
use rand::Rng;

use crate::climate::tensor_ops::conv1d;
use nalgebra::{DMatrix, DVector, Dyn, Matrix, Storage};
use pure_math::pure_math::analysis::optimization::{ModelOptimizer as Optimizer, ParamType};

/// A trait representing the Autoencoder model interface.
pub trait AutoencoderModel {
    /// Encodes the input data into a latent representation.
    #[verified_engine::verified]
    fn encode<S: Storage<f32, Dyn, Dyn>>(&self, input: &Matrix<f32, Dyn, Dyn, S>) -> DMatrix<f32>;

    /// Performs a forward pass through the autoencoder.
    #[verified_engine::verified]
    fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
    ) -> (DMatrix<f32>, DMatrix<f32>);
}

/// A simple leaky ReLU activation function.
///
/// # Arguments
///
/// * `x` - The matrix to apply the activation to (in-place).
/// * `alpha` - The negative slope coefficient.
#[verified_engine::verified]
pub fn leaky_relu(x: &mut DMatrix<f32>, alpha: f32) {
    x.iter_mut().for_each(|val| {
        if *val < 0.0 {
            *val *= alpha;
        }
    });
}

/// A single layer for the Encoder or Decoder, consisting of a convolution and activation.
pub struct ConvLayer {
    /// The convolution kernel matrix.
    pub kernel: DMatrix<f32>,
    /// The bias vector.
    pub bias: DVector<f32>,
}

impl ConvLayer {
    /// Creates a new convolutional layer with random initialization.
    ///
    /// # Arguments
    ///
    /// * `in_channels` - Number of input channels.
    /// * `out_channels` - Number of output channels.
    ///
    /// # Returns
    ///
    /// A new `ConvLayer`.
    #[verified_engine::verified]
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        // Simple random initialization using from_fn
        let kernel = DMatrix::from_fn(out_channels, in_channels, |_, _| {
            oxidize_core::rng::OxidizeRng::default().r#gen::<f32>() * 2.0 - 1.0
        });
        let bias = DVector::from_fn(out_channels, |_, _| {
            oxidize_core::rng::OxidizeRng::default().r#gen::<f32>() * 2.0 - 1.0
        });
        Self { kernel, bias }
    }

    /// Performs an analytical backward pass and parameter update for the convolutional layer.
    ///
    /// Computes weight and bias gradients from layer inputs and output gradients,
    /// verifies that gradients are finite, updates weights and bias using the provided optimizer,
    /// and returns the gradient with respect to the layer inputs.
    #[verified_engine::verified]
    pub fn backward_update<O: Optimizer<f32>>(
        &mut self,
        input: &DMatrix<f32>,
        loss_grad: &DMatrix<f32>,
        optimizer: &mut O,
        layer_idx: usize,
    ) -> Result<DMatrix<f32>, pure_math::pure_math::analysis::optimization::OptimizationError> {
        let n_samples = input.nrows();
        if n_samples == 0 || loss_grad.nrows() == 0 {
            return Ok(DMatrix::zeros(0, self.kernel.ncols()));
        }

        // Input gradient using kernel before parameter update: dX = dZ * W
        let grad_input = loss_grad * &self.kernel;

        // Weight gradient: dW = (1 / N) * dZ^T * X
        let grad_w = (loss_grad.transpose() * input) / (n_samples as f32);

        // Bias gradient: dB = (1 / N) * sum(dZ rows)
        let mut grad_b = DVector::zeros(self.bias.len());
        for row in loss_grad.row_iter() {
            grad_b += row.transpose();
        }
        grad_b /= n_samples as f32;

        if grad_w.iter().all(|v| v.is_finite()) && grad_b.iter().all(|v| v.is_finite()) {
            optimizer.update_matrix((layer_idx, ParamType::Weight), &mut self.kernel, &grad_w)?;
            optimizer.update_vector((layer_idx, ParamType::Bias), &mut self.bias, &grad_b)?;
        }

        Ok(grad_input)
    }

    /// Performs an analytical backward update for a single 1D vector input.
    #[verified_engine::verified]
    pub fn backward_update_vector<O: Optimizer<f32>>(
        &mut self,
        input: &DVector<f32>,
        loss_grad: &DVector<f32>,
        optimizer: &mut O,
        layer_idx: usize,
    ) -> Result<DVector<f32>, pure_math::pure_math::analysis::optimization::OptimizationError> {
        let input_mat = DMatrix::from_row_slice(1, input.len(), input.as_slice());
        let loss_mat = DMatrix::from_row_slice(1, loss_grad.len(), loss_grad.as_slice());
        let grad_in_mat = self.backward_update(&input_mat, &loss_mat, optimizer, layer_idx)?;
        Ok(DVector::from_column_slice(grad_in_mat.as_slice()))
    }

    /// Updates the weights of the layer using the provided optimizer.
    #[verified_engine::verified]
    pub fn update_weights<O: Optimizer<f32>>(
        &mut self,
        optimizer: &mut O,
        layer_idx: usize,
    ) -> Result<(), pure_math::pure_math::analysis::optimization::OptimizationError> {
        let grad_k = DMatrix::zeros(self.kernel.nrows(), self.kernel.ncols());
        let grad_b = DVector::zeros(self.bias.len());
        optimizer.update_matrix((layer_idx, ParamType::Weight), &mut self.kernel, &grad_k)?;
        optimizer.update_vector((layer_idx, ParamType::Bias), &mut self.bias, &grad_b)?;
        Ok(())
    }
}

/// The encoder component of the autoencoder.
pub struct Encoder {
    /// The stack of convolutional layers.
    pub layers: [ConvLayer; 3],
}

impl Encoder {
    /// Creates a new encoder with a hardcoded architecture: Input (2 channels) -> 64 -> 64 -> Latent (3 channels).
    #[verified_engine::verified]
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

    /// Encodes the input data into a latent representation.
    #[verified_engine::verified]
    pub fn forward<S: Storage<f32, Dyn, Dyn>>(
        &self,
        input: &Matrix<f32, Dyn, Dyn, S>,
    ) -> DMatrix<f32> {
        let mut x = input.clone_owned();
        for (i, layer) in self.layers.iter().enumerate() {
            x = conv1d(&x, &layer.kernel, &layer.bias);
            // No activation on the final layer
            if i < self.layers.len() - 1 {
                leaky_relu(&mut x, 0.01);
            }
        }
        x
    }

    /// Updates weights for all layers using the provided optimizer.
    pub fn update_weights<O: Optimizer<f32>>(
        &mut self,
        optimizer: &mut O,
        start_idx: usize,
    ) -> Result<(), pure_math::pure_math::analysis::optimization::OptimizationError> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update_weights(optimizer, start_idx + i)?;
        }
        Ok(())
    }
}

/// The decoder component of the autoencoder.
pub struct Decoder {
    /// The stack of convolutional layers.
    pub layers: [ConvLayer; 3],
}

impl Decoder {
    /// Creates a new decoder with architecture: Latent (3 channels) -> 64 -> 64 -> Output (2 channels).
    #[verified_engine::verified]
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

    /// Reconstructs the input data from the latent representation.
    #[verified_engine::verified]
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

    /// Updates weights for all layers using the provided optimizer.
    pub fn update_weights<O: Optimizer<f32>>(
        &mut self,
        optimizer: &mut O,
        start_idx: usize,
    ) -> Result<(), pure_math::pure_math::analysis::optimization::OptimizationError> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.update_weights(optimizer, start_idx + i)?;
        }
        Ok(())
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
    /// Creates a new autoencoder with 2 input channels and 3 latent channels.
    #[verified_engine::verified]
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

    /// Performs a forward pass through the autoencoder.
    #[verified_engine::verified]
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
    #[verified_engine::verified]
    fn encode<S: Storage<f32, Dyn, Dyn>>(&self, input: &Matrix<f32, Dyn, Dyn, S>) -> DMatrix<f32> {
        self.encoder.forward(input)
    }

    #[verified_engine::verified]
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
    ) -> Result<(), pure_math::pure_math::analysis::optimization::OptimizationError> {
        let mut activations = Vec::with_capacity(7);
        let mut zs = Vec::with_capacity(6);
        let mut current = x.clone();
        activations.push(current.clone());

        for i in 0..6 {
            let layer = if i < 3 {
                &self.encoder.layers[i]
            } else {
                &self.decoder.layers[i - 3]
            };
            let z = &layer.kernel * &current + &layer.bias;
            zs.push(z.clone());
            let mut a = z.clone();
            // Layers 2 (latent) and 5 (reconstruction output) use linear activations
            if i != 2 && i != 5 {
                for v in a.iter_mut() {
                    if *v < 0.0 {
                        *v *= 0.01;
                    }
                }
            }
            current = a.clone();
            activations.push(a);
        }

        let mut d_z = loss_grad.clone();

        for i in (0..6).rev() {
            let a_prev = &activations[i];
            let d_w = &d_z * a_prev.transpose();
            let d_b = d_z.clone();

            let layer = if i < 3 {
                &mut self.encoder.layers[i]
            } else {
                &mut self.decoder.layers[i - 3]
            };

            let d_a_prev = if i > 0 {
                Some(layer.kernel.transpose() * &d_z)
            } else {
                None
            };

            optimizer.update_matrix((i, ParamType::Weight), &mut layer.kernel, &d_w)?;
            optimizer.update_vector((i, ParamType::Bias), &mut layer.bias, &d_b)?;

            if let Some(d_a_prev) = d_a_prev {
                if i - 1 == 2 {
                    // Latent layer has linear activation
                    d_z = d_a_prev;
                } else {
                    let z_prev = &zs[i - 1];
                    let mut d_z_prev = d_a_prev;
                    for (j, val) in z_prev.iter().enumerate() {
                        if *val < 0.0 {
                            d_z_prev[j] *= 0.01;
                        }
                    }
                    d_z = d_z_prev;
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    #[test]
    #[verified_engine::verified]
    fn test_autoencoder_forward_pass() {
        let in_channels = 2;
        let latent_channels = 3;
        let num_levels = 30;
        let batch_size = 4;
        let n_samples = num_levels * batch_size;

        let autoencoder = Autoencoder::new(in_channels, latent_channels);

        let input = DMatrix::from_fn(n_samples, in_channels, |_, _| {
            oxidize_core::rng::OxidizeRng::default().r#gen()
        });

        let (latent, reconstruction) = autoencoder.forward(&input);

        assert_eq!(latent.nrows(), n_samples);
        assert_eq!(latent.ncols(), latent_channels);
        assert_eq!(reconstruction.nrows(), n_samples);
        assert_eq!(reconstruction.ncols(), in_channels);
    }

    #[test]
    #[verified_engine::verified]
    fn test_leaky_relu() {
        let mut matrix = DMatrix::from_row_slice(2, 2, &[-1.0, 2.0, -3.0, 0.0]);
        leaky_relu(&mut matrix, 0.1);
        let expected = DMatrix::from_row_slice(2, 2, &[-0.1, 2.0, -0.3, 0.0]);
        assert!((matrix - expected).abs().max() < math_commons::registry::TOLERANCE_FAST_F32);
    }

    #[test]
    #[verified_engine::verified]
    fn test_autoencoder_backward_update() {
        use pure_math::pure_math::analysis::optimization::SGD;
        use pure_math::pure_math::analysis::optimization::Trainable;

        let in_channels = 2;
        let latent_channels = 3;
        let mut autoencoder = Autoencoder::new(in_channels, latent_channels);

        let initial_kernel_0 = autoencoder.encoder.layers[0].kernel.clone();
        let initial_bias_0 = autoencoder.encoder.layers[0].bias.clone();
        let initial_decoder_kernel_2 = autoencoder.decoder.layers[2].kernel.clone();

        let x = nalgebra::DVector::from_vec(vec![1.0, -0.5]);
        let loss_grad = nalgebra::DVector::from_vec(vec![0.1, -0.2]);
        let mut optimizer = SGD::new(0.01);

        let result = autoencoder.backward_update(&x, &loss_grad, &mut optimizer);
        assert!(result.is_ok());

        assert_ne!(autoencoder.encoder.layers[0].kernel, initial_kernel_0);
        assert_ne!(autoencoder.encoder.layers[0].bias, initial_bias_0);
        assert_ne!(
            autoencoder.decoder.layers[2].kernel,
            initial_decoder_kernel_2
        );
    }
}
