//! Error types for the climate modeling domain.

use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_commons::error::MathError;
use pure_math::error::{LinalgError, LinearAlgebraError, OptimizationError};
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

    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] LinalgError),

    /// Upstream numerical optimization error.
    #[error("{0}")]
    Optimization(#[from] OptimizationError),

    /// Upstream numerical linear algebra error.
    #[error("{0}")]
    LinearAlgebra(#[from] LinearAlgebraError),
}

impl Diagnostic for ClimateError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            Self::Linalg(err) => err.severity(),
            Self::Optimization(err) => err.severity(),
            Self::LinearAlgebra(err) => err.severity(),
            _ => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        match self {
            Self::Math(err) => err.visit_fields(visitor),
            Self::Linalg(err) => err.visit_fields(visitor),
            Self::Optimization(err) => err.visit_fields(visitor),
            Self::LinearAlgebra(err) => err.visit_fields(visitor),
            _ => {
                visitor.record_str("error_type", "ClimateError");
                visitor.record_display("description", self);
            }
        }
        visitor.record_str("umbrella_type", "ClimateError");
    }
}
