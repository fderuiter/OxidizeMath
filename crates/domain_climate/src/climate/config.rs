//! This module defines the configuration for the CERA model.

/// Hyperparameters and architectural dimensions for a [`Cera`](crate::climate::cera::Cera) model.
///
/// Configures the input/latent channel widths, vertical atmospheric levels, optimization
/// step size, and loss weights balancing reconstruction, prediction, and Earth Mover's
/// Distance (EMD) representation alignment.
///
/// # Examples
///
/// ```rust
/// # use domain_climate::__doctest_facade as math_explorer;
/// use math_explorer::climate::config::CeraConfig;
///
/// let config = CeraConfig {
///     learning_rate: 0.001,
///     lambda_pred: 0.2,
///     lambda_emd: 0.1,
///     epochs: 5,
///     batch_size: 4,
///     in_channels: 2,
///     latent_channels: 4,
///     aligned_channels: 2,
///     num_levels: 10,
///     output_size: 3,
/// };
/// assert!(config.aligned_channels <= config.latent_channels);
/// assert_eq!(config.num_levels * config.aligned_channels, 20);
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
