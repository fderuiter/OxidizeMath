use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum AIError {
    /// Invalid parameter value (e.g. negative learning rate).
    #[error("Invalid parameter {name}: {value}")]
    #[allow(missing_docs)]
    InvalidParameter { name: String, value: f64 },

    /// Optimization failed or diverged.
    #[error("Optimization diverged")]
    OptimizationDivergence,

    /// State is uninitialized.
    #[error("Uninitialized state: {name}")]
    #[allow(missing_docs)]
    UninitializedState { name: String },

    /// Model dimension is incompatible with the requested attention head count.
    #[error(
        "Model dimension d_model ({d_model}) must be divisible by positive head count h ({heads})"
    )]
    #[allow(missing_docs)]
    InvalidHeadConfiguration { d_model: usize, heads: usize },

    /// Number of per-head weight matrices did not match head count `h`.
    #[error(
        "Expected {expected} weight matrices per projection type, got w_q: {got_q}, w_k: {got_k}, w_v: {got_v}"
    )]
    #[allow(missing_docs)]
    HeadCountMismatch {
        expected: usize,
        got_q: usize,
        got_k: usize,
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
    #[allow(missing_docs)]
    ProjectionShapeMismatch {
        projection: &'static str,
        expected: (usize, usize),
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
    #[allow(missing_docs)]
    HeadProjectionShapeMismatch {
        head: usize,
        projection: &'static str,
        expected: (usize, usize),
        actual: (usize, usize),
    },

    #[error("{0}")]
    #[allow(missing_docs)]
    Math(#[from] math_commons::error::MathError),
}

#[allow(missing_docs)]
pub type Result<T> = math_commons::error::MathResult<T>;

impl Diagnostic for AIError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(e) => {
                use diagnostics::Severity as MS;
                match diagnostics::Diagnostic::severity(e) {
                    MS::Info => Severity::Info,
                    MS::Warning => Severity::Warning,
                    MS::Error => Severity::Error,
                    MS::Fatal => Severity::Fatal,
                }
            }
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        match self {
            Self::Math(e) => diagnostics::Diagnostic::metadata(e),
            _ => {
                let mut map = HashMap::new();
                map.insert("error_type".to_string(), "AIError".to_string());
                map.insert("description".to_string(), self.to_string());
                map
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_ai_error_math_preserves_source() {
        let err = AIError::from(math_commons::error::MathError::DimensionMismatch {
            expected: math_commons::math_kernel::types::Dimension(2),
            actual: math_commons::math_kernel::types::Dimension(3),
        });
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);
    }
}
