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

    /// Foundational `math_commons` umbrella error.
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
    use math_commons::error::{
        EvalError, GlslError, LexError, ParseError, QmcError, ShaderEmitError,
    };
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_climate_error_umbrella_preserves_sources_and_diagnostics() {
        let cases = [
            ClimateError::from(MathError::DimensionMismatch {
                expected: math_commons::math_kernel::types::Dimension(2),
                actual: math_commons::math_kernel::types::Dimension(3),
            }),
            ClimateError::from(QmcError::InvalidBase { base: 1 }),
            ClimateError::from(LinalgError::SingularMatrix),
            ClimateError::from(FitError::SingularMatrix),
            ClimateError::from(EvalError::UnknownVariable("t".to_string())),
            ClimateError::from(GlslError::DepthLimitExceeded(64)),
            ClimateError::from(LexError::input_too_long(9000)),
            ClimateError::from(ParseError::EmptyExpression),
            ClimateError::from(ShaderEmitError::NonFiniteLiteral),
            ClimateError::from("custom conversion".to_string()),
        ];
        for err in &cases {
            assert!(err.source().is_some());
            assert_eq!(err.severity(), Severity::Error);
            assert_eq!(
                err.metadata().get("umbrella_type").map(String::as_str),
                Some("ClimateError")
            );
        }
    }
}
