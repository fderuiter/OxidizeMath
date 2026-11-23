use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum EpidemiologyError {
    /// Matrix V (Transition Matrix) is singular and cannot be inverted.
    #[error("Transition matrix V is singular and cannot be inverted")]
    SingularTransitionMatrix,
    /// Invalid Parameter (e.g., negative rate).
    #[error("Invalid epidemiological parameter {name}: {value}")]
    #[allow(missing_docs)]
    InvalidParameter { name: String, value: f64 },
    /// Missing Parameter (e.g., required field not set in builder).
    #[error("Missing required epidemiological parameter: {name}")]
    #[allow(missing_docs)]
    MissingParameter { name: String },
    /// Basic reproduction number `R0` was non-positive or non-finite.
    #[error("Basic reproduction number R0 must be finite and positive, got {r0}")]
    #[allow(missing_docs)]
    InvalidReproductionNumber { r0: f64 },
    /// Newton-Raphson derivative approached zero during final-size iteration.
    #[error("Final-size solver derivative ({derivative}) approached zero at S_inf = {x}")]
    #[allow(missing_docs)]
    NearZeroDerivative { x: f64, derivative: f64 },
    /// Newton-Raphson final-size solver failed to converge within maximum iterations.
    #[error("Final-size Newton-Raphson solver failed to converge within {iterations} iterations")]
    #[allow(missing_docs)]
    ConvergenceFailure { iterations: usize },
    /// Wrapped centralized mathematical error.
    #[error(transparent)]
    Math(#[from] math_commons::error::MathError),
}

impl Diagnostic for EpidemiologyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self {
            Self::Math(err) => err.severity(),
            _ => Severity::Error,
        }
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "EpidemiologyError".to_string());
        if let Self::Math(math_err) = self {
            map.extend(math_err.metadata());
        }
        map.insert("description".to_string(), self.to_string());
        map
    }
}
