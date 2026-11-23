use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
