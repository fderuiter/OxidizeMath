// [cite:engine_opt]
//! Typed `#![no_std]` curve-fitting error diagnostics.

pub use engine_linalg::LinalgError;
pub use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_scalars::ScalarError;
use thiserror::Error;

#[cfg(feature = "serde")]
mod fit_serde;

/// Typed diagnostic error returned by zero-allocation curve-fitting solvers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
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

    /// Underlying `#![no_std]` linear algebra engine failure preserving causal `source()`.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
}

impl From<ScalarError> for FitError {
    #[inline]
    fn from(err: ScalarError) -> Self {
        Self::Linalg(LinalgError::from(err))
    }
}

impl Diagnostic for FitError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        if let Self::Linalg(linalg_err) = self {
            linalg_err.visit_fields(visitor);
        }
        visitor.record_str("error_type", "FitError");
        visitor.record_display("description", self);
        match self {
            Self::InsufficientPoints { required, actual } => {
                visitor.record_str("code", "MATH-FIT-001");
                visitor.record_usize("required", *required);
                visitor.record_usize("actual", *actual);
            }
            Self::UnsupportedDegree { degree } => {
                visitor.record_str("code", "MATH-FIT-002");
                visitor.record_usize("degree", *degree);
            }
            Self::InvalidDomain { reason } => {
                visitor.record_str("code", "MATH-FIT-003");
                visitor.record_str("reason", reason);
            }
            Self::SingularMatrix => {
                visitor.record_str("code", "MATH-FIT-004");
            }
            Self::NonConvergence { iterations } => {
                visitor.record_str("code", "MATH-FIT-005");
                visitor.record_usize("iterations", *iterations);
            }
            Self::Linalg(_) => {
                visitor.record_str("code", "MATH-FIT-006");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::error::Error as CoreError;

    struct NullVisitor(usize);

    impl DiagnosticVisitor for NullVisitor {
        fn record_str(&mut self, _key: &str, _value: &str) {
            self.0 += 1;
        }

        fn record_display(&mut self, _key: &str, _value: &dyn core::fmt::Display) {
            self.0 += 1;
        }
    }

    #[test]
    fn fit_error_preserves_causal_source_and_diagnostics() {
        let linalg_err = LinalgError::WorkspaceTooSmall {
            required: 32,
            available: 16,
        };
        let fit_err = FitError::from(linalg_err);
        assert!(fit_err.source().is_some());

        let mut v = NullVisitor(0);
        fit_err.visit_fields(&mut v);
        assert!(v.0 >= 4);

        let from_scalar = FitError::from(ScalarError::WorkspaceTooSmall {
            required: 8,
            available: 2,
        });
        assert_eq!(
            from_scalar,
            FitError::Linalg(LinalgError::WorkspaceTooSmall {
                required: 8,
                available: 2,
            })
        );
    }
}
