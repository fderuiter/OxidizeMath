//! This module defines the configuration for the CERA model.

#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct CeraConfig {
    /// Learning rate for the optimizer.
    pub learning_rate: f32,
    /// Weight for the prediction loss term.
    pub lambda_pred: f32,
    /// Weight for the Earth Mover's Distance (EMD) loss term.
    pub lambda_emd: f32,
    /// Number of training epochs.
    pub epochs: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Number of input channels.
    pub in_channels: usize,
    /// Dimension of the latent space.
    pub latent_channels: usize,
    /// Number of channels used for alignment/prediction.
    pub aligned_channels: usize,
    /// Number of vertical levels (or time steps) in the input.
    pub num_levels: usize,
    /// Size of the output vector.
    pub output_size: usize,
}
