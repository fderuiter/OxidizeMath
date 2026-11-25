// [cite:engine_linalg]
use crate::{Diagnostic, DiagnosticVisitor, Severity};
use core::fmt;

/// Typed `#![no_std]` error returned by dense and sparse linear algebra factorizations and solvers.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinalgError {
    /// Matrix or vector dimensions do not conform to the requested operation.
    DimensionMismatch,
    /// Caller-supplied solver workspace slice does not have enough capacity.
    WorkspaceTooSmall {
        /// Minimum scalar elements required by the routine.
        required: usize,
        /// Available scalar elements in the workspace.
        available: usize,
    },
    /// Input contains `NaN`, `Infinity`, or an out-of-domain parameter (such as negative ridge $\lambda$).
    InvalidInput,
    /// Matrix is numerically singular or rank-deficient beyond the solver's requirements.
    SingularMatrix,
    /// Matrix is asymmetric or fails positive-definite / positive-semidefinite checks.
    NotPositiveDefinite,
    /// Iterative decomposition or Krylov solver exceeded its iteration budget without converging.
    NonConvergence,
}

impl LinalgError {
    /// Returns the static type identifier `"LinalgError"`.
    #[inline]
    pub const fn error_type(&self) -> &'static str {
        "LinalgError"
    }

    /// Returns the stable diagnostic error code (`LINALG-001`..`LINALG-006`).
    #[inline]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::DimensionMismatch => "LINALG-001",
            Self::WorkspaceTooSmall { .. } => "LINALG-002",
            Self::InvalidInput => "LINALG-003",
            Self::SingularMatrix => "LINALG-004",
            Self::NotPositiveDefinite => "LINALG-005",
            Self::NonConvergence => "LINALG-006",
        }
    }
}

impl fmt::Display for LinalgError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch => f.write_str("matrix or vector dimensions do not match"),
            Self::WorkspaceTooSmall {
                required,
                available,
            } => write!(
                f,
                "solver workspace too small: required {required} scalars, available {available}"
            ),
            Self::InvalidInput => {
                f.write_str("linear algebra input contains non-finite values or invalid parameters")
            }
            Self::SingularMatrix => f.write_str("matrix is numerically singular or rank-deficient"),
            Self::NotPositiveDefinite => {
                f.write_str("matrix is not symmetric positive-definite or positive-semidefinite")
            }
            Self::NonConvergence => {
                f.write_str("iterative linear algebra algorithm failed to converge within budget")
            }
        }
    }
}

impl core::error::Error for LinalgError {}

impl Diagnostic for LinalgError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", self.error_type());
        visitor.record_display("description", self);
        visitor.record_str("code", self.code());
        if let Self::WorkspaceTooSmall {
            required,
            available,
        } = self
        {
            visitor.record_usize("required", *required);
            visitor.record_usize("available", *available);
        }
    }
}
