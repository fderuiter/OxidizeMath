//! Error types for the climate modeling domain.

use diagnostics::{Diagnostic, Severity};
use math_commons::error::MathError;
use std::collections::HashMap;
use thiserror::Error;

/// Domain-specific errors for climate emulation, tensor layers, and loss functions.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ClimateError {
    /// Configured `aligned_channels` exceeded `latent_channels`.
    #[error(
        "Aligned channels ({aligned_channels}) cannot exceed latent channels ({latent_channels})"
    )]
    ChannelAlignmentExceeded {
        /// Requested aligned channel count.
        aligned_channels: usize,
        /// Configured latent channel count.
        latent_channels: usize,
    },

    /// Configured `aligned_channels` was zero.
    #[error("Aligned channels must be greater than zero")]
    ZeroAlignedChannels,

    /// Configured `num_levels` was zero.
    #[error("Number of vertical levels must be greater than zero")]
    ZeroLevels,

    /// Tensor channel or bias dimension mismatch in a convolutional layer.
    #[error("{context} Expected {expected}, got {actual}.")]
    ChannelMismatch {
        /// Human-readable layer invariant context.
        context: &'static str,
        /// Expected channel count.
        expected: usize,
        /// Actual channel count.
        actual: usize,
    },

    /// Matrix shape mismatch in a loss or distribution metric.
    #[error(
        "{context} Expected shape ({}, {}), got ({}, {}).",
        expected.0,
        expected.1,
        actual.0,
        actual.1
    )]
    ShapeMismatch {
        /// Human-readable metric invariant context.
        context: &'static str,
        /// Expected `(nrows, ncols)` shape.
        expected: (usize, usize),
        /// Actual `(nrows, ncols)` shape.
        actual: (usize, usize),
    },

    /// Shared mathematical error forwarded from `math_commons`.
    #[error("{0}")]
    Math(#[from] MathError),
}

impl Diagnostic for ClimateError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        match self {
            Self::Math(err) => err.metadata(),
            _ => {
                let mut map = HashMap::new();
                map.insert("error_type".to_string(), "ClimateError".to_string());
                map.insert("description".to_string(), self.to_string());
                map
            }
        }
    }
}
