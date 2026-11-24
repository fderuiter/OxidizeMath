// [cite:cera_framework]
//! This module defines the core CERA framework, integrating the autoencoder and predictor.

use crate::climate::autoencoder::{Autoencoder, AutoencoderModel};
use crate::climate::predictor::{Predictor, PredictorModel};
use crate::error::ClimateError;
use nalgebra::{DMatrix, Dyn, Matrix, Storage};
// Re-export CeraConfig for backward compatibility (or convenience)
pub use crate::climate::config::CeraConfig;
use verified_engine::Theory;

/// Climate-Emulation and Response Architecture (`Cera`) coupling an encoder-decoder representation
/// $z = E_\phi(x)$ with a linear-nonlinear cloud response predictor $\hat{y} = P_\psi(z_{\text{aligned}})$.
///
/// - **Citation**: `@Liu2025CERA` (`spec:climate_v1`, `papers/cera_framework.tex`).
/// - **Numerical stability**: Single-precision (`f32`) latent projections remain bounded when input columns are standardized and learning rates satisfy $\eta \in (0, 10^{-2}]$.
/// - **Domain of validity**: Requires `1 <= aligned_channels <= latent_channels`, `num_levels >= 1`, and input row counts divisible by `num_levels`.
#[derive(Theory)]
#[theory(
    description = "CERA is a Climate-Emulation and Response Architecture that combines autoencoders with a predictive downstream model to map local atmospheric states to global cloud properties under varying climate forcing.",
    citation = "Liu2025CERA"
)]
pub struct Cera<A: AutoencoderModel, P: PredictorModel> {
    /// The autoencoder component.
    pub autoencoder: A,
    /// The predictor component.
    pub predictor: P,
    /// The model configuration.
    pub config: CeraConfig,
}

impl Cera<Autoencoder, Predictor> {
    /// Creates a new CERA model instantiating default [`Autoencoder`] and [`Predictor`] dimensions
    /// $d_{\text{pred}} = L \cdot C_{\text{aligned}}$ (`@Liu2025CERA`, `spec:climate_v1`).
    ///
    /// - **Numerical stability**: Weight initialization scales inversely with channel fan-in to prevent activation saturation.
    /// - **Domain of validity**: Valid for `1 <= config.aligned_channels <= config.latent_channels` and `config.num_levels >= 1`; returns [`ClimateError`] otherwise.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration struct.
    ///
    /// # Returns
    ///
    /// A result containing the new `Cera` instance or a [`ClimateError`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use domain_climate::__doctest_facade as math_explorer;
    /// use math_explorer::climate::cera::Cera;
    /// use math_explorer::climate::config::CeraConfig;
    /// use math_explorer::nalgebra::DMatrix;
    ///
    /// let valid = CeraConfig {
    ///     learning_rate: 0.001,
    ///     lambda_pred: 0.1,
    ///     lambda_emd: 0.05,
    ///     epochs: 1,
    ///     batch_size: 2,
    ///     in_channels: 2,
    ///     latent_channels: 3,
    ///     aligned_channels: 2,
    ///     num_levels: 4,
    ///     output_size: 3,
    /// };
    /// let model = Cera::new(valid.clone()).unwrap();
    /// let batch = DMatrix::from_element(8, 2, 0.2_f32);
    /// let pred = model.predict(&batch);
    /// assert_eq!(pred.shape(), (2, 3));
    ///
    /// // Invariant: aligned_channels cannot exceed latent_channels
    /// let mut invalid = valid;
    /// invalid.aligned_channels = 5;
    /// assert!(Cera::new(invalid).is_err());
    /// ```
    #[verified_engine::verified]
    pub fn new(config: CeraConfig) -> Result<Self, ClimateError> {
        // Validation logic delegated to new_with_models
        let autoencoder = Autoencoder::new(config.in_channels, config.latent_channels);
        let predictor_input_size = config.num_levels * config.aligned_channels;
        let predictor = Predictor::new(predictor_input_size, config.output_size);

        Self::new_with_models(config, autoencoder, predictor)
    }
}

impl<P: PredictorModel> Cera<Autoencoder, P> {
    /// Creates a new CERA model pairing the default [`Autoencoder`] $z = E_\phi(x)$ with a custom predictor (`@Liu2025CERA`, `spec:climate_v1`).
    ///
    /// - **Numerical stability**: Preserves exact `f32` parameter buffers without intermediate rescaling.
    /// - **Domain of validity**: Valid when `1 <= config.aligned_channels <= config.latent_channels` and `config.num_levels >= 1`.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration struct.
    /// * `predictor` - A custom predictor implementation.
    ///
    /// # Returns
    ///
    /// A result containing the new `Cera` instance or a [`ClimateError`].
    #[verified_engine::verified]
    pub fn new_with_predictor(config: CeraConfig, predictor: P) -> Result<Self, ClimateError> {
        let autoencoder = Autoencoder::new(config.in_channels, config.latent_channels);
        Self::new_with_models(config, autoencoder, predictor)
    }
}

impl<A: AutoencoderModel, P: PredictorModel> Cera<A, P> {
    /// Creates a new CERA model from injected `autoencoder` and `predictor` components verifying $1 \le C_{\text{aligned}} \le C_{\text{latent}}$ (`@Liu2025CERA`, `spec:climate_v1`).
    ///
    /// - **Numerical stability**: Performs exact integer configuration checks with zero floating-point rounding error.
    /// - **Domain of validity**: Requires `config.aligned_channels >= 1`, `config.aligned_channels <= config.latent_channels`, and `config.num_levels >= 1`.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration struct.
    /// * `autoencoder` - The autoencoder model.
    /// * `predictor` - The predictor model.
    ///
    /// # Returns
    ///
    /// A result containing the new `Cera` instance or a [`ClimateError`].
    #[verified_engine::verified]
    pub fn new_with_models(
        config: CeraConfig,
        autoencoder: A,
        predictor: P,
    ) -> Result<Self, ClimateError> {
        if config.aligned_channels > config.latent_channels {
            return Err(ClimateError::ChannelAlignmentExceeded {
                aligned_channels: config.aligned_channels,
                latent_channels: config.latent_channels,
            });
        }
        if config.aligned_channels == 0 {
            return Err(ClimateError::ZeroAlignedChannels);
        }
        if config.num_levels == 0 {
            return Err(ClimateError::ZeroLevels);
        }

        Ok(Self {
            autoencoder,
            predictor,
            config,
        })
    }

    /// Reshapes a batch of latent vectors from $(B \cdot L, C_{\text{aligned}})$ to $(B, L \cdot C_{\text{aligned}})$ via $Z_{b, \ell C + c} = z_{b L + \ell, c}$ (`@Liu2025CERA`, `spec:climate_v1`).
    ///
    /// - **Numerical stability**: Bitwise-exact index permutation with zero arithmetic rounding error.
    /// - **Domain of validity**: Requires `latent_matrix.nrows() >= batch_size * self.config.num_levels` and `latent_matrix.ncols() == self.config.aligned_channels`.
    ///
    /// # Arguments
    ///
    /// * `latent_matrix` - The matrix of latent vectors.
    /// * `batch_size` - The number of samples in the batch.
    ///
    /// # Returns
    ///
    /// The reshaped matrix ready for the predictor.
    #[verified_engine::verified]
    pub fn reshape_for_predictor<S: Storage<f32, Dyn, Dyn>>(
        &self,
        latent_matrix: &Matrix<f32, Dyn, Dyn, S>,
        batch_size: usize,
    ) -> DMatrix<f32> {
        let num_levels = self.config.num_levels;
        let aligned_channels = self.config.aligned_channels;
        let mut reshaped_data = Vec::with_capacity(batch_size * num_levels * aligned_channels);
        for i in 0..batch_size {
            let start_row = i * num_levels;
            let sample_latent = latent_matrix.rows(start_row, num_levels);
            for r in sample_latent.row_iter() {
                for element in r.iter() {
                    reshaped_data.push(*element);
                }
            }
        }
        DMatrix::from_row_slice(batch_size, num_levels * aligned_channels, &reshaped_data)
    }

    /// Evaluates the coupled forward map $\hat{Y} = P_\psi(\text{reshape}(E_\phi(X)_{:, 1:C_{\text{aligned}}}))$ (`@Liu2025CERA`, `spec:climate_v1`).
    ///
    /// - **Numerical stability**: Forward `f32` matrix-vector products accumulate $O(L \cdot C_{\text{aligned}} \varepsilon_{\text{mach}})$ rounding error.
    /// - **Domain of validity**: Requires finite `inputs` with `inputs.nrows()` a positive multiple of `self.config.num_levels` and `inputs.ncols() == self.config.in_channels`.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Input data matrix.
    ///
    /// # Returns
    ///
    /// The predicted output matrix.
    #[verified_engine::verified]
    pub fn predict<S: Storage<f32, Dyn, Dyn>>(
        &self,
        inputs: &Matrix<f32, Dyn, Dyn, S>,
    ) -> DMatrix<f32> {
        let num_levels = self.config.num_levels;
        let aligned_channels = self.config.aligned_channels;
        let batch_size = inputs.nrows() / num_levels;

        let latent = self.autoencoder.encode(inputs);
        let aligned_latent = latent.columns(0, aligned_channels).clone_owned();
        let predictor_input = self.reshape_for_predictor(&aligned_latent, batch_size);
        self.predictor.forward(&predictor_input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::climate::training::CeraTrainer;
    use nalgebra::DMatrix; // Import Trainer
    use rand::Rng;

    // Helper constant for tests
    const TEST_NUM_LEVELS: usize = 30;
    const TEST_IN_CHANNELS: usize = 2;
    const TEST_OUTPUT_SIZE: usize = 148;

    #[verified_engine::verified]
    fn generate_data(n_samples: usize, offset: f32) -> (DMatrix<f32>, DMatrix<f32>) {
        let inputs = DMatrix::from_fn(n_samples * TEST_NUM_LEVELS, TEST_IN_CHANNELS, |_, _| {
            oxidize_core::rng::OxidizeRng::default().r#gen::<f32>() + offset
        });
        let targets = DMatrix::from_fn(n_samples, TEST_OUTPUT_SIZE, |_, _| {
            oxidize_core::rng::OxidizeRng::default().r#gen()
        });
        (inputs, targets)
    }

    #[test]
    #[verified_engine::verified]
    fn test_cera_training_and_prediction() {
        let config = CeraConfig {
            learning_rate: 0.001,
            lambda_pred: 0.1,
            lambda_emd: 0.01,
            epochs: 2, // Keep it short for testing
            batch_size: 4,
            in_channels: TEST_IN_CHANNELS,
            latent_channels: 3,
            aligned_channels: 2,
            num_levels: TEST_NUM_LEVELS,
            output_size: TEST_OUTPUT_SIZE,
        };

        let mut cera = Cera::new(config).expect("Failed to create CERA model");

        let n_samples = 16;
        let (control_inputs, control_targets) = generate_data(n_samples, 0.0);
        let (warm_inputs, _) = generate_data(n_samples, 1.0); // Warm climate has a different distribution

        // Use Trainer
        let mut trainer = CeraTrainer::new(&mut cera);
        trainer.train(&control_inputs, &control_targets, &warm_inputs);

        let (test_inputs, _) = generate_data(4, 0.5);
        let prediction = cera.predict(&test_inputs);

        assert_eq!(prediction.nrows(), 4);
        assert_eq!(prediction.ncols(), TEST_OUTPUT_SIZE);
    }

    #[test]
    #[verified_engine::verified]
    fn test_cera_invalid_config() {
        let config = CeraConfig {
            learning_rate: 0.001,
            lambda_pred: 0.1,
            lambda_emd: 0.01,
            epochs: 1,
            batch_size: 1,
            in_channels: 2,
            latent_channels: 3,
            aligned_channels: 4, // Invalid: > latent_channels
            num_levels: 30,
            output_size: 10,
        };
        assert!(Cera::new(config).is_err());
    }

    #[test]
    #[verified_engine::verified]
    fn test_cera_with_custom_models() {
        let config = CeraConfig {
            learning_rate: 0.001,
            lambda_pred: 0.1,
            lambda_emd: 0.01,
            epochs: 1,
            batch_size: 4,
            in_channels: 2,
            latent_channels: 3,
            aligned_channels: 2,
            num_levels: 30,
            output_size: 148,
        };

        // Create custom components (using default factory for simplicity, but could be manual)
        let encoder_layers = [
            crate::climate::autoencoder::ConvLayer::new(config.in_channels, 32),
            crate::climate::autoencoder::ConvLayer::new(32, 32),
            crate::climate::autoencoder::ConvLayer::new(32, config.latent_channels),
        ];
        let encoder = crate::climate::autoencoder::Encoder::new_from_layers(encoder_layers);
        let decoder =
            crate::climate::autoencoder::Decoder::new(config.latent_channels, config.in_channels);
        let autoencoder =
            crate::climate::autoencoder::Autoencoder::new_from_components(encoder, decoder);

        let predictor_input_size = config.num_levels * config.aligned_channels;
        let predictor = Predictor::new(predictor_input_size, config.output_size);

        // Here we use generic inference
        let cera = Cera::new_with_models(config, autoencoder, predictor);
        assert!(cera.is_ok());
    }
}
