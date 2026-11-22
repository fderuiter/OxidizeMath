pub use crate::ast::{EvalError, LexError, ParseError};
use crate::math_kernel::types::Dimension;
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

/// Centralized mathematical error type.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MathError {
    /// Dimension mismatch between tensors/matrices or expected sizes.
    #[error("Dimension mismatch: expected {expected:?}, actual {actual:?}")]
    DimensionMismatch {
        #[allow(missing_docs)]
        expected: Dimension,
        #[allow(missing_docs)]
        actual: Dimension,
    },

    /// Numerical computation failures (e.g., singular matrix, non-positive definite).
    #[error("Numerical error: {reason}")]
    #[allow(missing_docs)]
    NumericalError { reason: String },

    /// Conversion failures (e.g., failed to convert between numeric types).
    #[error("Conversion error: {reason}")]
    #[allow(missing_docs)]
    ConversionError { reason: String },
}

impl Diagnostic for MathError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "MathError".to_string());
        match self {
            Self::DimensionMismatch { expected, actual } => {
                map.insert("expected_dim".to_string(), expected.0.to_string());
                map.insert("actual_dim".to_string(), actual.0.to_string());
            }
            Self::NumericalError { reason } => {
                map.insert("reason".to_string(), reason.clone());
            }
            Self::ConversionError { reason } => {
                map.insert("reason".to_string(), reason.clone());
            }
        }
        map
    }
}

#[allow(missing_docs)]
pub type MathResult<T> = std::result::Result<T, MathError>;

impl From<String> for MathError {
    fn from(reason: String) -> Self {
        Self::ConversionError { reason }
    }
}

mod umbrella;
pub use umbrella::MathCommonsError;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    fn math_commons_umbrella_preserves_sources_and_diagnostics() {
        let lex = LexError {
            start: 0,
            end: 1,
            message: "bad char".to_string(),
        };
        let parse = ParseError::from(lex.clone());
        assert!(parse.source().is_some());

        let variants = [
            MathCommonsError::from(MathError::DimensionMismatch {
                expected: Dimension(2),
                actual: Dimension(3),
            }),
            MathCommonsError::from(lex),
            MathCommonsError::from(parse),
            MathCommonsError::from(EvalError::UnknownVariable("x".to_string())),
        ];

        for err in &variants {
            assert!(err.source().is_some());
            assert_eq!(err.severity(), Severity::Error);
            let meta = err.metadata();
            assert_eq!(
                meta.get("umbrella_type").map(String::as_str),
                Some("MathCommonsError")
            );
            assert!(meta.contains_key("error_type"));
        }
    }
}
