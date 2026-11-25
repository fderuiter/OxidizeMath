//! Error types for the climate modeling domain.

use diagnostics::{Diagnostic, Severity};
use math_commons::error::{
    EvalError, FitError, GlslError, LexError, LinalgError, MathCommonsError, MathError, ParseError,
    QmcError, ShaderEmitError,
};
use pure_math::error::OptimizationError;
use std::collections::HashMap;
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

    /// Wrapped foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),

    /// Wrapped mathematical optimization error from `pure_math`.
    #[error("{0}")]
    Optimization(#[from] OptimizationError),
}

impl From<FitError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: FitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<EvalError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ShaderEmitError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LinalgError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: LinalgError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<QmcError> for ClimateError {
    #[verified_engine::verified]
    fn from(err: QmcError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<String> for ClimateError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl ClimateError {
    #[verified_engine::verified]
    fn delegated_diagnostic(&self) -> Option<&dyn Diagnostic> {
        match self {
            Self::Math(err) => Some(err),
            Self::MathCommons(err) => Some(err),
            Self::Optimization(err) => Some(err),
            _ => None,
        }
    }
}

impl Diagnostic for ClimateError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.delegated_diagnostic()
            .map_or(Severity::Error, Diagnostic::severity)
    }

    #[verified_engine::verified]
    fn error_type(&self) -> &'static str {
        self.delegated_diagnostic()
            .map_or("ClimateError", Diagnostic::error_type)
    }

    #[verified_engine::verified]
    fn code(&self) -> Option<&'static str> {
        self.delegated_diagnostic().and_then(Diagnostic::code)
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = if let Some(inner) = self.delegated_diagnostic() {
            inner.metadata()
        } else {
            let mut m = HashMap::new();
            m.insert("error_type".to_string(), "ClimateError".to_string());
            m.insert("description".to_string(), self.to_string());
            m
        };
        map.insert("umbrella_type".to_string(), "ClimateError".to_string());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_climate_error_optimization_and_commons_preserve_source() {
        let opt = ClimateError::from(OptimizationError::ConversionError);
        assert!(opt.source().is_some());
        assert_eq!(opt.severity(), Severity::Error);

        let fit = ClimateError::from(FitError::SingularMatrix);
        assert!(fit.source().is_some());
        assert_eq!(fit.code(), Some("MATH-FIT-004"));

        let linalg = ClimateError::from(LinalgError::SingularMatrix);
        assert!(linalg.source().is_some());

        let qmc = ClimateError::from(QmcError::InvalidBase { base: 1 });
        assert!(qmc.source().is_some());
    }
}
