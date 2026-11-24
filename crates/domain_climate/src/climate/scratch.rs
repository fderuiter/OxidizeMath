// [cite:cera_framework]
//! Pre-allocated scratch workspaces enabling zero-allocation execution of CERA forward,
//! analytical backpropagation, Earth Mover's Distance (EMD), and training loops inside
//! [`verified_engine::engine::VerifiedEngine::run_verified`].

use crate::climate::config::CeraConfig;
use crate::error::ClimateError;
use nalgebra::{DMatrix, DVector};

/// Pre-allocated column buffers for zero-allocation 1D Earth Mover's Distance evaluation.
#[derive(Debug, Clone)]
pub struct EmdScratch {
    pub(crate) col1: Vec<f32>,
    pub(crate) col2: Vec<f32>,
}

impl EmdScratch {
    /// Pre-allocates paired sorting buffers able to hold columns up to `max_rows` samples.
    #[verified_engine::verified(alloc_ok = "pre-allocates EMD column sort workspace")]
    pub fn new(max_rows: usize) -> Self {
        Self {
            col1: vec![0.0; max_rows],
            col2: vec![0.0; max_rows],
        }
    }

    /// Maximum column sample count supported without reallocation.
    #[verified_engine::verified]
    pub fn max_rows(&self) -> usize {
        self.col1.len().min(self.col2.len())
    }
}

/// Pre-allocated weight and bias gradient buffers for [`crate::climate::autoencoder::ConvLayer`].
#[derive(Debug, Clone)]
pub struct ConvLayerScratch {
    pub(crate) grad_w: DMatrix<f32>,
    pub(crate) grad_b: DVector<f32>,
}

impl ConvLayerScratch {
    /// Pre-allocates gradient matrices for a convolutional layer with `(out_channels, in_channels)`.
    #[verified_engine::verified(alloc_ok = "pre-allocates ConvLayer gradient buffers")]
    pub fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            grad_w: DMatrix::zeros(out_channels, in_channels),
            grad_b: DVector::zeros(out_channels),
        }
    }
}

/// Pre-allocated forward and backward workspace for [`crate::climate::autoencoder::Autoencoder`].
#[derive(Debug, Clone)]
pub struct AutoencoderScratch {
    pub(crate) enc_h0: DMatrix<f32>,
    pub(crate) enc_h1: DMatrix<f32>,
    pub(crate) dec_h0: DMatrix<f32>,
    pub(crate) dec_h1: DMatrix<f32>,
    pub(crate) activations: [DVector<f32>; 7],
    pub(crate) zs: [DVector<f32>; 6],
    pub(crate) d_z: DVector<f32>,
    pub(crate) d_next: DVector<f32>,
    pub(crate) d_w: [DMatrix<f32>; 6],
}

impl AutoencoderScratch {
    /// Pre-allocates hidden batch tensors for `batch_rows` (`batch_size * num_levels`) and
    /// 6-layer single-vector backpropagation buffers.
    #[verified_engine::verified(
        alloc_ok = "pre-allocates Autoencoder forward and backward workspace"
    )]
    pub fn new(batch_rows: usize, in_channels: usize, latent_channels: usize) -> Self {
        let max_dim = in_channels.max(latent_channels).max(64);
        Self {
            enc_h0: DMatrix::zeros(batch_rows, 64),
            enc_h1: DMatrix::zeros(batch_rows, 64),
            dec_h0: DMatrix::zeros(batch_rows, 64),
            dec_h1: DMatrix::zeros(batch_rows, 64),
            activations: [
                DVector::zeros(in_channels),
                DVector::zeros(64),
                DVector::zeros(64),
                DVector::zeros(latent_channels),
                DVector::zeros(64),
                DVector::zeros(64),
                DVector::zeros(in_channels),
            ],
            zs: [
                DVector::zeros(64),
                DVector::zeros(64),
                DVector::zeros(latent_channels),
                DVector::zeros(64),
                DVector::zeros(64),
                DVector::zeros(in_channels),
            ],
            d_z: DVector::zeros(max_dim),
            d_next: DVector::zeros(max_dim),
            d_w: [
                DMatrix::zeros(64, in_channels),
                DMatrix::zeros(64, 64),
                DMatrix::zeros(latent_channels, 64),
                DMatrix::zeros(64, latent_channels),
                DMatrix::zeros(64, 64),
                DMatrix::zeros(in_channels, 64),
            ],
        }
    }

    /// Pre-allocates workspace matching arbitrary layer widths of an existing [`crate::climate::autoencoder::Autoencoder`].
    #[verified_engine::verified(
        alloc_ok = "pre-allocates Autoencoder workspace from model layer shapes"
    )]
    pub fn from_autoencoder(
        batch_rows: usize,
        ae: &crate::climate::autoencoder::Autoencoder,
    ) -> Self {
        let e0_in = ae.encoder.layers[0].kernel.ncols();
        let e0_out = ae.encoder.layers[0].kernel.nrows();
        let e1_out = ae.encoder.layers[1].kernel.nrows();
        let e2_out = ae.encoder.layers[2].kernel.nrows();
        let d0_out = ae.decoder.layers[0].kernel.nrows();
        let d1_out = ae.decoder.layers[1].kernel.nrows();
        let d2_out = ae.decoder.layers[2].kernel.nrows();
        let max_dim = [e0_in, e0_out, e1_out, e2_out, d0_out, d1_out, d2_out]
            .into_iter()
            .max()
            .unwrap_or(1);
        Self {
            enc_h0: DMatrix::zeros(batch_rows, e0_out),
            enc_h1: DMatrix::zeros(batch_rows, e1_out),
            dec_h0: DMatrix::zeros(batch_rows, d0_out),
            dec_h1: DMatrix::zeros(batch_rows, d1_out),
            activations: [
                DVector::zeros(e0_in),
                DVector::zeros(e0_out),
                DVector::zeros(e1_out),
                DVector::zeros(e2_out),
                DVector::zeros(d0_out),
                DVector::zeros(d1_out),
                DVector::zeros(d2_out),
            ],
            zs: [
                DVector::zeros(e0_out),
                DVector::zeros(e1_out),
                DVector::zeros(e2_out),
                DVector::zeros(d0_out),
                DVector::zeros(d1_out),
                DVector::zeros(d2_out),
            ],
            d_z: DVector::zeros(max_dim),
            d_next: DVector::zeros(max_dim),
            d_w: [
                DMatrix::zeros(e0_out, e0_in),
                DMatrix::zeros(e1_out, e0_out),
                DMatrix::zeros(e2_out, e1_out),
                DMatrix::zeros(d0_out, e2_out),
                DMatrix::zeros(d1_out, d0_out),
                DMatrix::zeros(d2_out, d1_out),
            ],
        }
    }
}

/// Pre-allocated forward and backward workspace for [`crate::climate::predictor::Predictor`].
#[derive(Debug, Clone)]
pub struct PredictorScratch {
    pub(crate) h: [DMatrix<f32>; 4],
    pub(crate) activations: [DVector<f32>; 6],
    pub(crate) zs: [DVector<f32>; 5],
    pub(crate) d_z: DVector<f32>,
    pub(crate) d_next: DVector<f32>,
    pub(crate) d_w: [DMatrix<f32>; 5],
}

impl PredictorScratch {
    /// Pre-allocates hidden batch tensors for `batch_size` and 5-layer backpropagation buffers.
    #[verified_engine::verified(
        alloc_ok = "pre-allocates Predictor forward and backward workspace"
    )]
    pub fn new(batch_size: usize, input_size: usize, output_size: usize) -> Self {
        let max_dim = input_size.max(output_size).max(128);
        Self {
            h: [
                DMatrix::zeros(batch_size, 128),
                DMatrix::zeros(batch_size, 128),
                DMatrix::zeros(batch_size, 128),
                DMatrix::zeros(batch_size, 128),
            ],
            activations: [
                DVector::zeros(input_size),
                DVector::zeros(128),
                DVector::zeros(128),
                DVector::zeros(128),
                DVector::zeros(128),
                DVector::zeros(output_size),
            ],
            zs: [
                DVector::zeros(128),
                DVector::zeros(128),
                DVector::zeros(128),
                DVector::zeros(128),
                DVector::zeros(output_size),
            ],
            d_z: DVector::zeros(max_dim),
            d_next: DVector::zeros(max_dim),
            d_w: [
                DMatrix::zeros(128, input_size),
                DMatrix::zeros(128, 128),
                DMatrix::zeros(128, 128),
                DMatrix::zeros(128, 128),
                DMatrix::zeros(output_size, 128),
            ],
        }
    }
}

/// Pre-allocated inference workspace for [`crate::climate::cera::Cera::predict_into`].
#[derive(Debug, Clone)]
pub struct CeraScratch {
    /// Autoencoder forward workspace.
    pub ae_scratch: AutoencoderScratch,
    /// Predictor forward workspace.
    pub pred_scratch: PredictorScratch,
    /// Intermediate latent buffer `(batch_size * num_levels, latent_channels)`.
    pub latent: DMatrix<f32>,
    /// Reshaped aligned latent buffer `(batch_size, num_levels * aligned_channels)`.
    pub predictor_input: DMatrix<f32>,
}

impl CeraScratch {
    /// Pre-allocates inference buffers sized for `config.batch_size` samples.
    #[verified_engine::verified(alloc_ok = "pre-allocates Cera inference workspace")]
    pub fn from_config(config: &CeraConfig) -> Result<Self, ClimateError> {
        Self::new(config, config.batch_size)
    }

    /// Pre-allocates inference buffers sized for `batch_size` samples under `config`.
    #[verified_engine::verified(
        alloc_ok = "pre-allocates Cera inference workspace for explicit batch size"
    )]
    pub fn new(config: &CeraConfig, batch_size: usize) -> Result<Self, ClimateError> {
        let batch_rows =
            batch_size
                .checked_mul(config.num_levels)
                .ok_or(ClimateError::DimensionOverflow {
                    context: "CeraScratch batch_size * num_levels",
                })?;
        let pred_in = config
            .num_levels
            .checked_mul(config.aligned_channels)
            .ok_or(ClimateError::DimensionOverflow {
                context: "CeraScratch num_levels * aligned_channels",
            })?;
        let _ = batch_rows.checked_mul(config.latent_channels).ok_or(
            ClimateError::DimensionOverflow {
                context: "CeraScratch latent matrix elements",
            },
        )?;
        let _ = batch_size
            .checked_mul(pred_in)
            .ok_or(ClimateError::DimensionOverflow {
                context: "CeraScratch predictor_input matrix elements",
            })?;

        Ok(Self {
            ae_scratch: AutoencoderScratch::new(
                batch_rows,
                config.in_channels,
                config.latent_channels,
            ),
            pred_scratch: PredictorScratch::new(batch_size, pred_in, config.output_size),
            latent: DMatrix::zeros(batch_rows, config.latent_channels),
            predictor_input: DMatrix::zeros(batch_size, pred_in),
        })
    }
}

/// Pre-allocated training workspace for [`crate::climate::training::CeraTrainer::train_step_into`]
/// and [`crate::climate::training::CeraTrainer::train_into`].
#[derive(Debug, Clone)]
pub struct CeraTrainingScratch {
    /// Autoencoder forward and backward workspace.
    pub ae_scratch: AutoencoderScratch,
    /// Predictor forward and backward workspace.
    pub pred_scratch: PredictorScratch,
    /// Column-sorting workspace for Earth Mover's Distance.
    pub emd_scratch: EmdScratch,
    /// Control-climate latent batch `(batch_rows, latent_channels)`.
    pub control_latent: DMatrix<f32>,
    /// Control-climate reconstruction batch `(batch_rows, in_channels)`.
    pub control_recon: DMatrix<f32>,
    /// Warm-climate latent batch `(batch_rows, latent_channels)`.
    pub warm_latent: DMatrix<f32>,
    /// Warm-climate reconstruction batch `(batch_rows, in_channels)`.
    pub warm_recon: DMatrix<f32>,
    /// Reshaped aligned latent batch `(batch_size, num_levels * aligned_channels)`.
    pub predictor_input: DMatrix<f32>,
    /// Predictor output batch `(batch_size, output_size)`.
    pub prediction: DMatrix<f32>,
    /// Averaged autoencoder input vector `(in_channels)`.
    pub ae_input: DVector<f32>,
    /// Analytical autoencoder loss gradient `(in_channels)`.
    pub ae_loss_grad: DVector<f32>,
    /// Averaged predictor input vector `(num_levels * aligned_channels)`.
    pub pred_input_vec: DVector<f32>,
    /// Analytical predictor loss gradient `(output_size)`.
    pub pred_loss_grad: DVector<f32>,
}

impl CeraTrainingScratch {
    /// Pre-allocates all forward, loss, and backpropagation buffers sized for `config`.
    #[verified_engine::verified(
        alloc_ok = "pre-allocates CeraTrainer zero-allocation training workspace"
    )]
    pub fn from_config(config: &CeraConfig) -> Result<Self, ClimateError> {
        let batch_rows = config.batch_size.checked_mul(config.num_levels).ok_or(
            ClimateError::DimensionOverflow {
                context: "CeraTrainingScratch batch_size * num_levels",
            },
        )?;
        let pred_in = config
            .num_levels
            .checked_mul(config.aligned_channels)
            .ok_or(ClimateError::DimensionOverflow {
                context: "CeraTrainingScratch num_levels * aligned_channels",
            })?;
        let _ = batch_rows
            .checked_mul(config.latent_channels.max(config.in_channels))
            .ok_or(ClimateError::DimensionOverflow {
                context: "CeraTrainingScratch batch tensor elements",
            })?;
        let _ = config
            .batch_size
            .checked_mul(pred_in.max(config.output_size))
            .ok_or(ClimateError::DimensionOverflow {
                context: "CeraTrainingScratch predictor tensor elements",
            })?;

        Ok(Self {
            ae_scratch: AutoencoderScratch::new(
                batch_rows,
                config.in_channels,
                config.latent_channels,
            ),
            pred_scratch: PredictorScratch::new(config.batch_size, pred_in, config.output_size),
            emd_scratch: EmdScratch::new(batch_rows),
            control_latent: DMatrix::zeros(batch_rows, config.latent_channels),
            control_recon: DMatrix::zeros(batch_rows, config.in_channels),
            warm_latent: DMatrix::zeros(batch_rows, config.latent_channels),
            warm_recon: DMatrix::zeros(batch_rows, config.in_channels),
            predictor_input: DMatrix::zeros(config.batch_size, pred_in),
            prediction: DMatrix::zeros(config.batch_size, config.output_size),
            ae_input: DVector::zeros(config.in_channels),
            ae_loss_grad: DVector::zeros(config.in_channels),
            pred_input_vec: DVector::zeros(pred_in),
            pred_loss_grad: DVector::zeros(config.output_size),
        })
    }
}
