//! This module defines the predictor model for the CERA framework.

use crate::climate::autoencoder::{ConvLayer, leaky_relu};
use nalgebra::{DMatrix, Dyn, Matrix, Storage};
use pure_math::pure_math::analysis::optimization::{ModelOptimizer as Optimizer, ParamType};

/// A trait representing the predictor model interface.
/// This allows for different predictor architectures and decouples the training loop.
pub trait PredictorModel {
    /// Performs a forward pass through the predictor.
    #[verified_engine::verified]
    fn forward<S: Storage<f32, Dyn, Dyn>>(&self, input: &Matrix<f32, Dyn, Dyn, S>) -> DMatrix<f32>;
}

/// A multi-layer perceptron (MLP) used as the predictor in the CERA framework.
///
/// The predictor takes the flattened, aligned latent representation from the
/// autoencoder's encoder and maps it to the target output variables.
pub struct Predictor {
    /// The stack of layers (using `ConvLayer` for simplicity as dense layers).
    pub layers: [ConvLayer; 5],
}

impl Predictor {
    /// Creates a new predictor model with a hardcoded architecture.
    /// Input (60) -> 128 -> 128 -> 128 -> 128 -> Output (148)
    ///
    /// # Arguments
    ///
    /// * `input_size` - The dimension of the input vector.
    /// * `output_size` - The dimension of the output vector.
    ///
    /// # Returns
    ///
    /// A new `Predictor` instance.
    #[verified_engine::verified]
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let layers = [
            ConvLayer::new(input_size, 128),
            ConvLayer::new(128, 128),
            ConvLayer::new(128, 128),
            ConvLayer::new(128, 128),
            ConvLayer::new(128, output_size), // No activation on the final layer
        ];
        Self::new_from_layers(layers)
    }

    /// Creates a new predictor model with custom layers.
    ///
    /// # Arguments
    ///
    /// * `layers` - A fixed-size array of convolutional layers.
    ///
    /// # Returns
    ///
    /// A new `Predictor` instance.
    #[verified_engine::verified]
    pub fn new_from_layers(layers: [ConvLayer; 5]) -> Self {
        Self { layers }
    }
}

impl PredictorModel for Predictor {
    #[verified_engine::verified]
    fn forward<S: Storage<f32, Dyn, Dyn>>(&self, input: &Matrix<f32, Dyn, Dyn, S>) -> DMatrix<f32> {
        let mut x = input.clone_owned();
        for (i, layer) in self.layers.iter().enumerate() {
            x = crate::climate::tensor_ops::conv1d(&x, &layer.kernel, &layer.bias);
            if i < self.layers.len() - 1 {
                leaky_relu(&mut x, 0.01);
            }
        }
        x
    }
}

impl pure_math::pure_math::analysis::optimization::Trainable<f32> for Predictor {
    fn forward(&self, x: &nalgebra::DVector<f32>) -> nalgebra::DVector<f32> {
        let mut current = x.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            current = &layer.kernel * &current + &layer.bias;
            if i < self.layers.len() - 1 {
                for v in current.iter_mut() {
                    if *v < 0.0 {
                        *v *= 0.01;
                    }
                }
            }
        }
        current
    }

    fn backward_update(
        &mut self,
        x: &nalgebra::DVector<f32>,
        loss_grad: &nalgebra::DVector<f32>,
        optimizer: &mut dyn Optimizer<f32>,
    ) -> Result<(), pure_math::pure_math::analysis::optimization::OptimizationError> {
        let mut activations = Vec::new();
        let mut zs = Vec::new();
        let mut current = x.clone();
        activations.push(current.clone());

        for (i, layer) in self.layers.iter().enumerate() {
            let z = &layer.kernel * &current + &layer.bias;
            zs.push(z.clone());
            let mut a = z.clone();
            if i < self.layers.len() - 1 {
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

        for i in (0..self.layers.len()).rev() {
            let a_prev = &activations[i];

            let d_w = &d_z * a_prev.transpose();
            let d_b = d_z.clone();

            let layer = &mut self.layers[i];
            optimizer.update_matrix((i, ParamType::Weight), &mut layer.kernel, &d_w)?;
            optimizer.update_vector((i, ParamType::Bias), &mut layer.bias, &d_b)?;

            if i > 0 {
                let d_a_prev = layer.kernel.transpose() * &d_z;
                let z_prev = &zs[i - 1];
                let mut d_z_prev = d_a_prev.clone();
                for (j, val) in z_prev.iter().enumerate() {
                    if *val < 0.0 {
                        d_z_prev[j] *= 0.01;
                    }
                }
                d_z = d_z_prev;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use pure_math::pure_math::analysis::optimization::SGD;
    use rand::Rng;

    #[test]
    #[verified_engine::verified]
    fn test_predictor_forward_pass() {
        let input_size = 30 * 2; // 30 levels, 2 aligned latent channels
        let output_size = 148;
        let batch_size = 4;

        let predictor = Predictor::new(input_size, output_size);

        let input = DMatrix::from_fn(batch_size, input_size, |_, _| {
            oxidize_core::rng::OxidizeRng::default().r#gen()
        });

        let output = predictor.forward(&input);

        assert_eq!(output.nrows(), batch_size);
        assert_eq!(output.ncols(), output_size);
    }

    #[test]
    #[verified_engine::verified]
    fn test_predictor_update_weights_with_optimizer() {
        use nalgebra::DVector;
        use pure_math::pure_math::analysis::optimization::Trainable;
        let input_size = 10;
        let output_size = 5;
        let mut predictor = Predictor::new(input_size, output_size);
        let mut optimizer = SGD::new(0.1f32);

        let initial_kernel = predictor.layers[0].kernel.clone();

        // Update weights
        let dummy_x = DVector::from_element(input_size, 0.5);
        let dummy_grad = DVector::from_element(output_size, 0.1);
        let _ = predictor.backward_update(&dummy_x, &dummy_grad, &mut optimizer);

        // Check that weights have changed
        let new_kernel = &predictor.layers[0].kernel;
        assert_ne!(initial_kernel, *new_kernel);
    }
}
