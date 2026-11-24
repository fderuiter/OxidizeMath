// [cite:cera_framework]
//! This module defines the core CERA framework, integrating the autoencoder and predictor.

use crate::climate::autoencoder::{Autoencoder, AutoencoderModel};
use crate::climate::predictor::{Predictor, PredictorModel};
use crate::climate::scratch::CeraScratch;
use crate::error::ClimateError;
use nalgebra::{DMatrix, Dyn, Matrix, Storage};
// Re-export CeraConfig for backward compatibility (or convenience)
pub use crate::climate::config::CeraConfig;
use verified_engine::Theory;

#[verified_engine::verified]
pub(crate) fn validate_cera_config(config: &CeraConfig) -> Result<usize, ClimateError> {
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
    config
        .num_levels
        .checked_mul(config.aligned_channels)
        .ok_or(ClimateError::DimensionOverflow {
            context: "CeraConfig num_levels * aligned_channels",
        })
}

/// The main CERA model.
#[derive(Theory)]
#[theory(
    description = "CERA is a Climate-Emulation and Response Architecture that combines autoencoders with a predictive downstream model to map local atmospheric states to global cloud properties under varying climate forcing.",
    citation = "Climate-Emulation and Response Architecture (CERA) paper"
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
    /// Creates a new CERA model with the given configuration.
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
    #[verified_engine::verified(alloc_ok = "initializes Autoencoder and Predictor weights")]
    pub fn new(config: CeraConfig) -> Result<Self, ClimateError> {
        let predictor_input_size = validate_cera_config(&config)?;
        let autoencoder = Autoencoder::new(config.in_channels, config.latent_channels);
        let predictor = Predictor::new(predictor_input_size, config.output_size);
        Self::new_with_models(config, autoencoder, predictor)
    }

    /// Zero-allocation inference writing predictions into `output` using [`CeraScratch`].
    #[verified_engine::verified]
    pub fn predict_into<S: Storage<f32, Dyn, Dyn>>(
        &self,
        inputs: &Matrix<f32, Dyn, Dyn, S>,
        scratch: &mut CeraScratch,
        output: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        let num_levels = self.config.num_levels;
        if num_levels == 0 {
            return Err(ClimateError::ZeroLevels);
        }
        if inputs.ncols() != self.config.in_channels {
            return Err(ClimateError::ChannelMismatch {
                context: "Cera::predict_into input channel mismatch.",
                expected: self.config.in_channels,
                actual: inputs.ncols(),
            });
        }
        if !inputs.nrows().is_multiple_of(num_levels) {
            return Err(ClimateError::ShapeMismatch {
                context: "Cera::predict_into input rows must be a multiple of num_levels.",
                expected: (num_levels, self.config.in_channels),
                actual: inputs.shape(),
            });
        }
        let batch_size = inputs.nrows() / num_levels;
        self.autoencoder
            .encode_into(inputs, &mut scratch.ae_scratch, &mut scratch.latent)?;
        self.reshape_for_predictor_into(&scratch.latent, batch_size, &mut scratch.predictor_input)?;
        self.predictor
            .forward_into(&scratch.predictor_input, &mut scratch.pred_scratch, output)
    }
}

impl<P: PredictorModel> Cera<Autoencoder, P> {
    /// Creates a new CERA model with a custom predictor.
    #[verified_engine::verified(alloc_ok = "initializes Autoencoder weights")]
    pub fn new_with_predictor(config: CeraConfig, predictor: P) -> Result<Self, ClimateError> {
        validate_cera_config(&config)?;
        let autoencoder = Autoencoder::new(config.in_channels, config.latent_channels);
        Self::new_with_models(config, autoencoder, predictor)
    }
}

impl<A: AutoencoderModel, P: PredictorModel> Cera<A, P> {
    /// Creates a new CERA model with injected dependencies.
    #[verified_engine::verified]
    pub fn new_with_models(
        config: CeraConfig,
        autoencoder: A,
        predictor: P,
    ) -> Result<Self, ClimateError> {
        validate_cera_config(&config)?;
        Ok(Self {
            autoencoder,
            predictor,
            config,
        })
    }

    /// Reshapes the leading `aligned_channels` columns of `latent_matrix` from
    /// `(batch_size * num_levels, >= aligned_channels)` into pre-allocated `output` of shape
    /// `(batch_size, num_levels * aligned_channels)` without heap allocations.
    #[verified_engine::verified]
    pub fn reshape_for_predictor_into<S: Storage<f32, Dyn, Dyn>>(
        &self,
        latent_matrix: &Matrix<f32, Dyn, Dyn, S>,
        batch_size: usize,
        output: &mut DMatrix<f32>,
    ) -> Result<(), ClimateError> {
        let num_levels = self.config.num_levels;
        let aligned_channels = self.config.aligned_channels;
        let required_rows =
            batch_size
                .checked_mul(num_levels)
                .ok_or(ClimateError::DimensionOverflow {
                    context: "reshape_for_predictor batch_size * num_levels",
                })?;
        let pred_cols =
            num_levels
                .checked_mul(aligned_channels)
                .ok_or(ClimateError::DimensionOverflow {
                    context: "reshape_for_predictor num_levels * aligned_channels",
                })?;
        let _ = batch_size
            .checked_mul(pred_cols)
            .ok_or(ClimateError::DimensionOverflow {
                context: "reshape_for_predictor total elements",
            })?;
        if latent_matrix.nrows() < required_rows || latent_matrix.ncols() < aligned_channels {
            return Err(ClimateError::ShapeMismatch {
                context: "Latent matrix too small for batch_size * num_levels or aligned_channels.",
                expected: (required_rows, aligned_channels),
                actual: latent_matrix.shape(),
            });
        }
        if output.shape() != (batch_size, pred_cols) {
            return Err(ClimateError::ShapeMismatch {
                context: "Pre-allocated predictor_input matrix shape mismatch.",
                expected: (batch_size, pred_cols),
                actual: output.shape(),
            });
        }

        for b in 0..batch_size {
            let start_row = b * num_levels;
            for lvl in 0..num_levels {
                let col_base = lvl * aligned_channels;
                for ch in 0..aligned_channels {
                    output[(b, col_base + ch)] = latent_matrix[(start_row + lvl, ch)];
                }
            }
        }
        Ok(())
    }

    /// Fallible reshape of a batch of latent vectors for the predictor with checked dimensions.
    #[verified_engine::verified(alloc_ok = "allocates reshaped predictor input matrix")]
    pub fn try_reshape_for_predictor<S: Storage<f32, Dyn, Dyn>>(
        &self,
        latent_matrix: &Matrix<f32, Dyn, Dyn, S>,
        batch_size: usize,
    ) -> Result<DMatrix<f32>, ClimateError> {
        let required_rows = batch_size.checked_mul(self.config.num_levels).ok_or(
            ClimateError::DimensionOverflow {
                context: "reshape_for_predictor batch_size * num_levels",
            },
        )?;
        let pred_cols = self
            .config
            .num_levels
            .checked_mul(self.config.aligned_channels)
            .ok_or(ClimateError::DimensionOverflow {
                context: "reshape_for_predictor num_levels * aligned_channels",
            })?;
        let _ = batch_size
            .checked_mul(pred_cols)
            .ok_or(ClimateError::DimensionOverflow {
                context: "reshape_for_predictor total elements",
            })?;
        if latent_matrix.nrows() < required_rows
            || latent_matrix.ncols() < self.config.aligned_channels
        {
            return Err(ClimateError::ShapeMismatch {
                context: "Latent matrix too small for batch_size * num_levels or aligned_channels.",
                expected: (required_rows, self.config.aligned_channels),
                actual: latent_matrix.shape(),
            });
        }
        let mut out = DMatrix::zeros(batch_size, pred_cols);
        self.reshape_for_predictor_into(latent_matrix, batch_size, &mut out)?;
        Ok(out)
    }

    /// Reshapes a batch of latent vectors for the predictor.
    /// From `(batch_size * num_levels, channels)` to `(batch_size, num_levels * channels)`.
    #[verified_engine::verified(
        alloc_ok = "convenience reshape wrapper allocates predictor input matrix"
    )]
    pub fn reshape_for_predictor<S: Storage<f32, Dyn, Dyn>>(
        &self,
        latent_matrix: &Matrix<f32, Dyn, Dyn, S>,
        batch_size: usize,
    ) -> DMatrix<f32> {
        self.try_reshape_for_predictor(latent_matrix, batch_size)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Fallible CERA prediction validating input row divisibility and channel dimensions.
    #[verified_engine::verified(
        alloc_ok = "convenience try_predict allocates intermediate and output matrices"
    )]
    pub fn try_predict<S: Storage<f32, Dyn, Dyn>>(
        &self,
        inputs: &Matrix<f32, Dyn, Dyn, S>,
    ) -> Result<DMatrix<f32>, ClimateError> {
        let num_levels = self.config.num_levels;
        if num_levels == 0 {
            return Err(ClimateError::ZeroLevels);
        }
        if inputs.ncols() != self.config.in_channels {
            return Err(ClimateError::ChannelMismatch {
                context: "Cera::try_predict input channel mismatch.",
                expected: self.config.in_channels,
                actual: inputs.ncols(),
            });
        }
        if !inputs.nrows().is_multiple_of(num_levels) {
            return Err(ClimateError::ShapeMismatch {
                context: "Cera::try_predict input rows must be a multiple of num_levels.",
                expected: (num_levels, self.config.in_channels),
                actual: inputs.shape(),
            });
        }
        let batch_size = inputs.nrows() / num_levels;
        let latent = self.autoencoder.encode(inputs);
        let predictor_input = self.try_reshape_for_predictor(&latent, batch_size)?;
        Ok(self.predictor.forward(&predictor_input))
    }

    /// Makes a prediction using the trained CERA model.
    #[verified_engine::verified(
        alloc_ok = "convenience predict wrapper allocates intermediate and output matrices"
    )]
    pub fn predict<S: Storage<f32, Dyn, Dyn>>(
        &self,
        inputs: &Matrix<f32, Dyn, Dyn, S>,
    ) -> DMatrix<f32> {
        self.try_predict(inputs).unwrap_or_else(|e| panic!("{e}"))
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
