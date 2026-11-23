// [cite:math_commons]
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

/// Typed diagnostic error returned by zero-allocation curve-fitting solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
pub enum FitError {
    /// Fewer finite data points were supplied than required by the target model.
    #[error("Insufficient finite points: required at least {required}, found {actual}")]
    InsufficientPoints {
        /// Minimum number of finite observations needed.
        required: usize,
        /// Actual number of finite observations found.
        actual: usize,
    },

    /// Polynomial regression degree is outside the supported stack-allocated `1..=4` range.
    #[error("Unsupported polynomial degree {degree}; supported degrees are 1..=4")]
    UnsupportedDegree {
        /// Requested polynomial degree.
        degree: usize,
    },

    /// Input coordinates violated a model domain requirement (e.g., non-positive log/power inputs).
    #[error("Invalid curve-fitting domain: {reason}")]
    InvalidDomain {
        /// Static description of the violated domain precondition.
        reason: &'static str,
    },

    /// Normal equations or damped Jacobian linear system was singular or ill-conditioned.
    #[error("Singular or ill-conditioned linear system during curve fitting")]
    SingularMatrix,

    /// Damped Levenberg-Marquardt failed to satisfy optimality tolerances within its budget.
    #[error("Levenberg-Marquardt failed to converge within {iterations} iterations")]
    NonConvergence {
        /// Number of iterations executed before termination.
        iterations: usize,
    },
}

impl Diagnostic for FitError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "FitError".to_string());
        match self {
            Self::InsufficientPoints { required, actual } => {
                map.insert("code".to_string(), "MATH-FIT-001".to_string());
                map.insert("required".to_string(), required.to_string());
                map.insert("actual".to_string(), actual.to_string());
            }
            Self::UnsupportedDegree { degree } => {
                map.insert("code".to_string(), "MATH-FIT-002".to_string());
                map.insert("degree".to_string(), degree.to_string());
            }
            Self::InvalidDomain { reason } => {
                map.insert("code".to_string(), "MATH-FIT-003".to_string());
                map.insert("reason".to_string(), (*reason).to_string());
            }
            Self::SingularMatrix => {
                map.insert("code".to_string(), "MATH-FIT-004".to_string());
            }
            Self::NonConvergence { iterations } => {
                map.insert("code".to_string(), "MATH-FIT-005".to_string());
                map.insert("iterations".to_string(), iterations.to_string());
            }
        }
        map
    }
}
