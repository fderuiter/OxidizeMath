//! Error types for the climate modeling domain.

use math_commons::error::{FitError, LinalgError, MathCommonsError, MathError};
use pure_math::error::{LinearAlgebraError, OptimizationError};
use thiserror::Error;

/// Domain-specific errors for climate emulation, tensor layers, and loss functions.
#[non_exhaustive]
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

    /// Shared `math_commons` umbrella error (AST, QMC, RNG, fit, or linalg).
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),

    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] LinalgError),

    /// Foundational curve-fitting solver error.
    #[error("{0}")]
    Fit(#[from] FitError),

    /// Upstream numerical optimization error.
    #[error("{0}")]
    Optimization(#[from] OptimizationError),

    /// Upstream numerical linear algebra error.
    #[error("{0}")]
    LinearAlgebra(#[from] LinearAlgebraError),
}

mod conversions;

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::{Diagnostic, Severity};
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_climate_error_conversions_preserve_source() {
        let qmc = ClimateError::from(math_commons::error::QmcError::InvalidBase { base: 1 });
        let rng = ClimateError::from(math_commons::error::RngError::ZeroUpperBound);
        let parse = ClimateError::from(math_commons::error::ParseError::EmptyExpression);
        let static_str = ClimateError::from("invalid level");
        assert!(qmc.source().is_some());
        assert!(rng.source().is_some());
        assert!(parse.source().is_some());
        assert!(static_str.source().is_some());
        assert_eq!(qmc.severity(), Severity::Error);
    }
}
