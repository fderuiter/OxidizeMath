// [cite:engine_diffeq]
use diagnostics::CoreDiagnostic;
#[cfg(feature = "std")]
use diagnostics::{Diagnostic, Severity};
use math_commons::error::MathError;
#[cfg(feature = "std")]
use std::collections::HashMap;
use thiserror::Error;

/// Stack-allocated `#![no_std]` error type for differential and stochastic equation solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum DiffeqError {
    /// Physical or numerical solver parameter violated positivity or finiteness invariants.
    #[error("Invalid differential equation solver parameter: {parameter}")]
    InvalidParameter {
        /// Static identifier of the invalid parameter.
        parameter: &'static str,
    },

    /// Integration step size `dt` was negative or non-finite.
    #[error("Integration step size dt must be finite and non-negative")]
    InvalidStepSize,

    /// Caller-supplied state, gradient, or Wiener increment slice was shorter than required.
    #[error("Buffer '{buffer}' too small: required at least {required}, found {actual}")]
    BufferTooSmall {
        /// Static label of the undersized slice.
        buffer: &'static str,
        /// Minimum slice length required.
        required: usize,
        /// Actual slice length provided.
        actual: usize,
    },

    /// Heap scratch allocation for [`crate::OwnedWorkspace`] failed.
    #[error("Failed to allocate solver workspace of capacity {capacity}")]
    WorkspaceAllocationFailed {
        /// Requested scalar capacity.
        capacity: usize,
    },

    /// Underlying workspace or primitive math error.
    #[error("{0}")]
    Math(#[from] MathError),
}

impl CoreDiagnostic for DiffeqError {
    fn error_type(&self) -> &'static str {
        "DiffeqError"
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidParameter { .. } => "DIFFEQ-001",
            Self::InvalidStepSize => "DIFFEQ-002",
            Self::BufferTooSmall { .. } => "DIFFEQ-003",
            Self::WorkspaceAllocationFailed { .. } => "DIFFEQ-004",
            Self::Math(_) => "DIFFEQ-005",
        }
    }
}

#[cfg(feature = "std")]
impl Diagnostic for DiffeqError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), self.error_type().to_string());
        map.insert("code".to_string(), self.code().to_string());
        map.insert("description".to_string(), self.to_string());
        match self {
            Self::InvalidParameter { parameter } => {
                map.insert("parameter".to_string(), (*parameter).to_string());
            }
            Self::InvalidStepSize => {}
            Self::BufferTooSmall {
                buffer,
                required,
                actual,
            } => {
                map.insert("buffer".to_string(), (*buffer).to_string());
                map.insert("required".to_string(), required.to_string());
                map.insert("actual".to_string(), actual.to_string());
            }
            Self::WorkspaceAllocationFailed { capacity } => {
                map.insert("capacity".to_string(), capacity.to_string());
            }
            Self::Math(inner) => {
                for (k, v) in inner.metadata() {
                    map.entry(k).or_insert(v);
                }
            }
        }
        map
    }
}
