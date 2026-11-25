// [cite:math_commons]
#[cfg(feature = "std")]
pub use crate::ast::{EvalError, GlslError, LexError, ParseError, ShaderEmitError};
use crate::math_kernel::types::Dimension;
use alloc::string::String;
pub use engine_linalg::LinalgError;
pub use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use thiserror::Error;

/// Centralized mathematical error type.
#[derive(Debug, Clone, PartialEq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathError {
    /// Dimension mismatch between tensors/matrices or expected sizes.
    #[error("Dimension mismatch: expected {}, actual {}", expected.0, actual.0)]
    DimensionMismatch {
        /// Expected structural dimension.
        expected: Dimension,
        /// Actual encountered dimension.
        actual: Dimension,
    },

    /// Floating-point value lies outside an inclusive range `[min, max]` or is non-finite.
    #[error("Value {value} is outside the allowed inclusive range [{min}, {max}]")]
    OutOfBounds {
        /// Rejected floating-point value.
        value: f64,
        /// Inclusive lower bound.
        min: f64,
        /// Inclusive upper bound.
        max: f64,
    },

    /// Floating-point value lies outside a strictly open range `(min, max)` or is non-finite.
    #[error("Value {value} is outside the strictly allowed open range ({min}, {max})")]
    StrictlyOutOfBounds {
        /// Rejected floating-point value.
        value: f64,
        /// Exclusive lower bound.
        min: f64,
        /// Exclusive upper bound.
        max: f64,
    },

    /// Permutation mapping contains an out-of-range or duplicate target index.
    #[error("Invalid permutation element {value} at index {index} for domain size {len}")]
    InvalidPermutation {
        /// Position in the input slice where the invalid element occurred.
        index: usize,
        /// Out-of-range or duplicate target element.
        value: usize,
        /// Expected permutation domain size `n`.
        len: usize,
    },

    /// Pre-allocated solver scratch workspace had fewer elements than required.
    #[error("Solver workspace too small: required {required} elements, available {available}")]
    WorkspaceTooSmall {
        /// Number of elements requested by the caller.
        required: usize,
        /// Remaining elements available in the workspace slice.
        available: usize,
    },

    /// Numerical computation failures (e.g., singular matrix, non-positive definite).
    #[error("Numerical error: {reason}")]
    NumericalError {
        /// Human-readable description of the numerical failure.
        reason: String,
    },

    /// Conversion failures (e.g., failed to convert between numeric types).
    #[error("Conversion error: {reason}")]
    ConversionError {
        /// Human-readable description of the conversion failure.
        reason: String,
    },
}

impl Diagnostic for MathError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "MathError");
        visitor.record_display("description", self);
        match self {
            Self::DimensionMismatch { expected, actual } => {
                visitor.record_usize("expected_dim", expected.0);
                visitor.record_usize("actual_dim", actual.0);
            }
            Self::OutOfBounds { value, min, max }
            | Self::StrictlyOutOfBounds { value, min, max } => {
                visitor.record_f64("value", *value);
                visitor.record_f64("min", *min);
                visitor.record_f64("max", *max);
            }
            Self::InvalidPermutation { index, value, len } => {
                visitor.record_usize("index", *index);
                visitor.record_usize("value", *value);
                visitor.record_usize("len", *len);
            }
            Self::WorkspaceTooSmall {
                required,
                available,
            } => {
                visitor.record_usize("required", *required);
                visitor.record_usize("available", *available);
            }
            Self::NumericalError { reason } | Self::ConversionError { reason } => {
                visitor.record_str("reason", reason);
            }
        }
    }
}

/// Convenience result alias returning [`MathError`] on failure.
pub type MathResult<T> = core::result::Result<T, MathError>;

impl From<String> for MathError {
    fn from(reason: String) -> Self {
        Self::ConversionError { reason }
    }
}

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

mod umbrella;
pub use umbrella::MathCommonsError;

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::error::Error as StdError;
    use verified_engine::engine::VerifiedEngine;

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
    fn zero_allocation_diagnostic_visitation_for_linalg_math_and_fit() {
        let linalg_err = LinalgError::WorkspaceTooSmall {
            required: 32,
            available: 16,
        };
        let math_err = MathError::DimensionMismatch {
            expected: Dimension(3),
            actual: Dimension(2),
        };
        let fit_err = FitError::from(linalg_err);
        assert!(fit_err.source().is_some());

        let (count, _metrics) = VerifiedEngine::run_verified(|| {
            let mut v = NullVisitor(0);
            linalg_err.visit_fields(&mut v);
            math_err.visit_fields(&mut v);
            fit_err.visit_fields(&mut v);
            v.0
        });
        assert!(count >= 12);
    }

    #[test]
    fn math_commons_umbrella_preserves_sources_and_diagnostics() {
        let lex = LexError {
            start: 0,
            end: 1,
            message: "bad char".to_string(),
        };
        let parse = ParseError::from(lex.clone());
        assert!(parse.source().is_some());

        let unknown_fn = ParseError::UnknownFunction {
            name: "foo".to_string(),
            span: crate::ast::Span { start: 3, end: 6 },
        };
        let unknown_meta = unknown_fn.metadata();
        assert_eq!(
            unknown_meta.get("function").map(String::as_str),
            Some("foo")
        );
        assert_eq!(unknown_meta.get("start").map(String::as_str), Some("3"));

        let wrong_arity = ParseError::WrongArgumentCount {
            func: crate::ast::Func::Atan2,
            expected: 2,
            found: 1,
            span: crate::ast::Span { start: 4, end: 12 },
        };
        let arity_meta = wrong_arity.metadata();
        assert_eq!(
            arity_meta.get("function").map(String::as_str),
            Some("atan2")
        );
        assert_eq!(arity_meta.get("expected").map(String::as_str), Some("2"));
        assert_eq!(arity_meta.get("found").map(String::as_str), Some("1"));
        assert_eq!(arity_meta.get("start").map(String::as_str), Some("4"));

        let variants = [
            MathCommonsError::from(MathError::DimensionMismatch {
                expected: Dimension(2),
                actual: Dimension(3),
            }),
            MathCommonsError::from(FitError::SingularMatrix),
            MathCommonsError::from(LinalgError::SingularMatrix),
            MathCommonsError::from(lex),
            MathCommonsError::from(parse),
            MathCommonsError::from(unknown_fn),
            MathCommonsError::from(wrong_arity),
            MathCommonsError::from(EvalError::UnknownVariable("x".to_string())),
            MathCommonsError::from(GlslError::DepthLimitExceeded(64)),
            MathCommonsError::from(ShaderEmitError::RecursionLimitExceeded(64)),
        ];

        for err in &variants {
            assert!(err.source().is_some());
            assert_eq!(err.severity(), Severity::Error);
            let meta = err.metadata();
            assert_eq!(
                meta.get("umbrella_type").map(String::as_str),
                Some("MathCommonsError")
            );
            assert!(meta.contains_key("error_type"));
        }
    }
}
