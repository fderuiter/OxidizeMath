use diagnostics::{Diagnostic, Severity};
use math_commons::error::{
    EvalError, FitError, GlslError, LexError, LinalgError, MathCommonsError, MathError, ParseError,
    QmcError, ShaderEmitError,
};
use pure_math::error::OptimizationError;
use std::collections::HashMap;
use thiserror::Error;

/// Error variants produced by AI architectures, optimizers, and training loops.
#[non_exhaustive]
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AIError {
    /// Invalid parameter value (e.g. negative learning rate).
    #[error("Invalid parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter identifier.
        name: String,
        /// Rejected numeric value.
        value: f64,
    },

    /// Optimization failed or diverged.
    #[error("Optimization diverged")]
    OptimizationDivergence,

    /// State is uninitialized.
    #[error("Uninitialized state: {name}")]
    UninitializedState {
        /// Name of the uninitialized state buffer or parameter.
        name: String,
    },

    /// Model dimension is incompatible with the requested attention head count.
    #[error(
        "Model dimension d_model ({d_model}) must be divisible by positive head count h ({heads})"
    )]
    InvalidHeadConfiguration {
        /// Embedding dimension `d_model`.
        d_model: usize,
        /// Requested number of attention heads `h`.
        heads: usize,
    },

    /// Number of per-head weight matrices did not match head count `h`.
    #[error(
        "Expected {expected} weight matrices per projection type, got w_q: {got_q}, w_k: {got_k}, w_v: {got_v}"
    )]
    HeadCountMismatch {
        /// Expected number of heads `h`.
        expected: usize,
        /// Supplied query projection count.
        got_q: usize,
        /// Supplied key projection count.
        got_k: usize,
        /// Supplied value projection count.
        got_v: usize,
    },

    /// Output projection weight matrix had incompatible dimensions.
    #[error(
        "{projection} dimension mismatch: expected ({}, {}), got ({}, {})",
        expected.0,
        expected.1,
        actual.0,
        actual.1
    )]
    ProjectionShapeMismatch {
        /// Name of the projection matrix (`w_o`).
        projection: &'static str,
        /// Expected `(nrows, ncols)` shape.
        expected: (usize, usize),
        /// Actual `(nrows, ncols)` shape.
        actual: (usize, usize),
    },

    /// Per-head projection weight matrix had incompatible dimensions.
    #[error(
        "Head {head} {projection} dimension mismatch: expected ({}, {}), got ({}, {})",
        expected.0,
        expected.1,
        actual.0,
        actual.1
    )]
    HeadProjectionShapeMismatch {
        /// Zero-based attention head index.
        head: usize,
        /// Name of the head projection matrix (`w_q`, `w_k`, or `w_v`).
        projection: &'static str,
        /// Expected `(nrows, ncols)` shape.
        expected: (usize, usize),
        /// Actual `(nrows, ncols)` shape.
        actual: (usize, usize),
    },

    /// Wrapped foundational error from `math_commons`.
    #[error("{0}")]
    Math(#[from] MathError),

    /// Wrapped foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),

    /// Wrapped mathematical optimization error from `pure_math`.
    #[error("{0}")]
    Optimization(#[from] OptimizationError),
}

/// Standard result alias used across `domain_ai` computations.
pub type Result<T> = math_commons::error::MathResult<T>;

impl From<FitError> for AIError {
    #[verified_engine::verified]
    fn from(err: FitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<EvalError> for AIError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for AIError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for AIError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for AIError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ShaderEmitError> for AIError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LinalgError> for AIError {
    #[verified_engine::verified]
    fn from(err: LinalgError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<QmcError> for AIError {
    #[verified_engine::verified]
    fn from(err: QmcError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<String> for AIError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl AIError {
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

impl Diagnostic for AIError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.delegated_diagnostic()
            .map_or(Severity::Error, Diagnostic::severity)
    }

    #[verified_engine::verified]
    fn error_type(&self) -> &'static str {
        self.delegated_diagnostic()
            .map_or("AIError", Diagnostic::error_type)
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
            m.insert("error_type".to_string(), "AIError".to_string());
            m.insert("description".to_string(), self.to_string());
            m
        };
        map.insert("umbrella_type".to_string(), "AIError".to_string());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_ai_error_math_and_optimization_preserve_source() {
        let err = AIError::from(MathError::DimensionMismatch {
            expected: math_commons::math_kernel::types::Dimension(2),
            actual: math_commons::math_kernel::types::Dimension(3),
        });
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);

        let opt = AIError::from(OptimizationError::ConversionError);
        assert!(opt.source().is_some());

        let linalg = AIError::from(LinalgError::SingularMatrix);
        assert!(linalg.source().is_some());
        assert_eq!(linalg.code(), Some("ENGINE-LINALG-004"));

        let qmc = AIError::from(QmcError::InvalidBase { base: 1 });
        assert!(qmc.source().is_some());
    }
}
