use crate::math_kernel::types::Dimension;
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

/// Centralized mathematical error type.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MathError {
    /// Dimension mismatch between tensors/matrices or expected sizes.
    #[error("Dimension mismatch: expected {}, actual {}", expected.0, actual.0)]
    DimensionMismatch {
        #[allow(missing_docs)]
        expected: Dimension,
        #[allow(missing_docs)]
        actual: Dimension,
    },

    /// Floating-point value lies outside an inclusive range `[min, max]` or is non-finite.
    #[error("Value {value} is outside the allowed inclusive range [{min}, {max}]")]
    OutOfBounds {
        #[allow(missing_docs)]
        value: f64,
        #[allow(missing_docs)]
        min: f64,
        #[allow(missing_docs)]
        max: f64,
    },

    /// Floating-point value lies outside a strictly open range `(min, max)` or is non-finite.
    #[error("Value {value} is outside the strictly allowed open range ({min}, {max})")]
    StrictlyOutOfBounds {
        #[allow(missing_docs)]
        value: f64,
        #[allow(missing_docs)]
        min: f64,
        #[allow(missing_docs)]
        max: f64,
    },

    /// Permutation mapping contains an out-of-range or duplicate target index.
    #[error("Invalid permutation element {value} at index {index} for domain size {len}")]
    InvalidPermutation {
        #[allow(missing_docs)]
        index: usize,
        #[allow(missing_docs)]
        value: usize,
        #[allow(missing_docs)]
        len: usize,
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
        map.insert("description".to_string(), self.to_string());
        match self {
            Self::DimensionMismatch { expected, actual } => {
                map.insert("expected_dim".to_string(), expected.0.to_string());
                map.insert("actual_dim".to_string(), actual.0.to_string());
            }
            Self::OutOfBounds { value, min, max }
            | Self::StrictlyOutOfBounds { value, min, max } => {
                map.insert("value".to_string(), value.to_string());
                map.insert("min".to_string(), min.to_string());
                map.insert("max".to_string(), max.to_string());
            }
            Self::InvalidPermutation { index, value, len } => {
                map.insert("index".to_string(), index.to_string());
                map.insert("value".to_string(), value.to_string());
                map.insert("len".to_string(), len.to_string());
            }
            Self::NumericalError { reason } | Self::ConversionError { reason } => {
                map.insert("reason".to_string(), reason.clone());
            }
        }
        map
    }
}

#[allow(missing_docs)]
pub type MathResult<T> = std::result::Result<T, MathError>;
