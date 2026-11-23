use num_complex::Complex64;
use thiserror::Error;

/// Errors that can occur during spectral transforms.
#[derive(Error, Debug, PartialEq, Clone, Eq)]
pub enum TransformError {
    #[error("Input dimensions must be a power of 2")]
    #[allow(missing_docs)]
    InvalidDimension,
    #[error("Input buffer is empty")]
    #[allow(missing_docs)]
    EmptyInput,
}

/// A trait for performing spectral transforms (Forward and Inverse).
///
/// This trait abstracts over the implementation details (e.g., FFT vs DFT)
/// and allows for different backends.
pub trait SpectralTransform {
    /// Performs the forward transform (e.g., Time -> Frequency).
    ///
    /// # Arguments
    /// * `input` - The input signal (time domain).
    ///
    /// # Returns
    /// * `Result<Vec<Complex64>, TransformError>` - The transformed signal (frequency domain).
    #[verified_engine::verified]
    fn forward(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, TransformError>;

    /// Performs the inverse transform (e.g., Frequency -> Time).
    ///
    /// # Arguments
    /// * `input` - The input signal (frequency domain).
    ///
    /// # Returns
    /// * `Result<Vec<Complex64>, TransformError>` - The reconstructed signal (time domain).
    #[verified_engine::verified]
    fn inverse(&mut self, input: &[Complex64]) -> Result<Vec<Complex64>, TransformError>;
}
