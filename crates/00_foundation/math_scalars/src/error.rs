// [cite:math_scalars]
//! Stack-allocated `#![no_std]` error type for scalar domain and workspace bounds checks.

use core::fmt;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_memory::ArenaError;

/// Stack-allocated `#![no_std]` error returned by [`crate::ScalarWorkspace`] and [`crate::StepSize`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ScalarError {
    /// Physical or numerical grid step size was non-finite or non-positive (`<= 0`).
    InvalidStepSize {
        /// Rejected step size converted to `f64`.
        value: f64,
    },

    /// Caller-supplied [`crate::ScalarWorkspace`] lacked capacity for a requested slice.
    WorkspaceTooSmall {
        /// Requested slice length in scalars.
        required: usize,
        /// Remaining available scalar capacity in the workspace.
        available: usize,
    },
}

impl ScalarError {
    /// Returns the static type identifier `"ScalarError"`.
    #[inline]
    #[must_use]
    pub const fn error_type(&self) -> &'static str {
        "ScalarError"
    }

    /// Returns the stable diagnostic error code (`SCALAR-001`..`SCALAR-002`).
    #[inline]
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidStepSize { .. } => "SCALAR-001",
            Self::WorkspaceTooSmall { .. } => "SCALAR-002",
        }
    }
}

impl fmt::Display for ScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStepSize { value } => {
                write!(
                    f,
                    "StepSize must be finite and strictly positive, got {value}"
                )
            }
            Self::WorkspaceTooSmall {
                required,
                available,
            } => write!(
                f,
                "ScalarWorkspace exhausted: requested {required} scalars, only {available} available"
            ),
        }
    }
}

impl From<ArenaError> for ScalarError {
    fn from(err: ArenaError) -> Self {
        match err {
            ArenaError::InsufficientBytes {
                required,
                available,
                ..
            } => Self::WorkspaceTooSmall {
                required,
                available,
            },
            _ => Self::WorkspaceTooSmall {
                required: usize::MAX,
                available: 0,
            },
        }
    }
}

impl core::error::Error for ScalarError {}

impl Diagnostic for ScalarError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", self.error_type());
        visitor.record_str("code", self.code());
        visitor.record_display("description", self);
        match self {
            Self::InvalidStepSize { value } => {
                visitor.record_f64("value", *value);
            }
            Self::WorkspaceTooSmall {
                required,
                available,
            } => {
                visitor.record_usize("required", *required);
                visitor.record_usize("available", *available);
            }
        }
    }
}
