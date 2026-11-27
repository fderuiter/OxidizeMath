// [cite:math_scalars]
//! Stack-allocated `#![no_std]` linear algebra error types and row-major index helper.

use crate::error::ScalarError;
use core::fmt;
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};

/// Computes the flat row-major index for `(row, col)` in a matrix with `ncols` columns.
#[inline]
pub const fn mat_idx(row: usize, col: usize, ncols: usize) -> usize {
    row * ncols + col
}

/// Typed `#![no_std]` error returned by foundational dense and sparse linear algebra factorizations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinalgError {
    /// Matrix or vector dimensions do not conform to the requested operation.
    DimensionMismatch,
    /// Caller-supplied [`SolverWorkspace`](crate::SolverWorkspace) slice does not have enough capacity.
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
        visitor.record_str("error_type", "LinalgError");
        visitor.record_display("description", self);
        match self {
            Self::DimensionMismatch => {
                visitor.record_str("code", "LINALG-001");
            }
            Self::WorkspaceTooSmall {
                required,
                available,
            } => {
                visitor.record_str("code", "LINALG-002");
                visitor.record_usize("required", *required);
                visitor.record_usize("available", *available);
            }
            Self::InvalidInput => {
                visitor.record_str("code", "LINALG-003");
            }
            Self::SingularMatrix => {
                visitor.record_str("code", "LINALG-004");
            }
            Self::NotPositiveDefinite => {
                visitor.record_str("code", "LINALG-005");
            }
            Self::NonConvergence => {
                visitor.record_str("code", "LINALG-006");
            }
        }
    }
}

impl From<ScalarError> for LinalgError {
    #[inline]
    fn from(err: ScalarError) -> Self {
        match err {
            ScalarError::WorkspaceTooSmall {
                required,
                available,
            } => Self::WorkspaceTooSmall {
                required,
                available,
            },
            ScalarError::InvalidStepSize { .. } => Self::InvalidInput,
        }
    }
}

/// Errors related to high-level matrix and linear system operations.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LinearAlgebraError {
    /// The matrix is singular and cannot be inverted or decomposed.
    SingularMatrix,
    /// Dimensions of operands are incompatible.
    DimensionMismatch,
    /// Solution could not be found.
    SolutionNotFound,
    /// Input contains non-finite (`NaN`/`Infinity`) values or out-of-range parameters.
    InvalidInput(&'static str),
    /// Matrix is not symmetric positive-definite.
    NotPositiveDefinite,
    /// Iterative decomposition or Krylov solver failed to converge within budget.
    NonConvergence,
    /// Foundational `#![no_std]` linear algebra engine error.
    Linalg(LinalgError),
}

impl fmt::Display for LinearAlgebraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SingularMatrix => f.write_str("Matrix is singular"),
            Self::DimensionMismatch => f.write_str("Matrix/Vector dimensions are incompatible"),
            Self::SolutionNotFound => f.write_str("Solution could not be found"),
            Self::InvalidInput(msg) => write!(f, "Invalid linear algebra input: {msg}"),
            Self::NotPositiveDefinite => f.write_str("Matrix is not symmetric positive-definite"),
            Self::NonConvergence => {
                f.write_str("Iterative linear algebra solver failed to converge")
            }
            Self::Linalg(err) => fmt::Display::fmt(err, f),
        }
    }
}

impl core::error::Error for LinearAlgebraError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Linalg(err) => Some(err),
            _ => None,
        }
    }
}

impl From<LinalgError> for LinearAlgebraError {
    #[inline]
    fn from(err: LinalgError) -> Self {
        Self::Linalg(err)
    }
}

impl Diagnostic for LinearAlgebraError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Linalg(linalg_err) = self {
            linalg_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "LinearAlgebraError");
        visitor.record_display("description", self);
    }
}
