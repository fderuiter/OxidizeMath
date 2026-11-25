// [cite:math_commons]
#[cfg(feature = "std")]
pub use crate::ast::{EvalError, GlslError, LexError, ParseError, ShaderEmitError};
use crate::math_kernel::types::Dimension;
use diagnostics::CoreDiagnostic;
#[cfg(feature = "std")]
use diagnostics::{Diagnostic, Severity};
#[cfg(feature = "std")]
use std::collections::HashMap;
use thiserror::Error;

#[inline]
fn f64_eq_canonical(a: f64, b: f64) -> bool {
    (a == 0.0 && b == 0.0) || a.to_bits() == b.to_bits()
}

/// Centralized `#![no_std]` `Copy` stack mathematical error type.
#[derive(Debug, Clone, Copy, Error)]
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

    /// Preallocated solver scratch buffer could not satisfy the requested slice length.
    #[error("Workspace buffer too small: required {required} elements, available {available}")]
    WorkspaceTooSmall {
        /// Minimum number of elements requested by the caller.
        required: usize,
        /// Remaining elements available in the workspace buffer.
        available: usize,
    },

    /// Numerical computation failures (e.g., singular matrix, non-positive definite).
    #[error("Numerical error: {reason}")]
    #[allow(missing_docs)]
    NumericalError { reason: &'static str },

    /// Conversion failures (e.g., failed to convert between numeric types).
    #[error("Conversion error: {reason}")]
    #[allow(missing_docs)]
    ConversionError { reason: &'static str },
}

#[inline]
fn bounds_key(err: MathError) -> Option<(bool, f64, f64, f64)> {
    match err {
        MathError::OutOfBounds { value, min, max } => Some((false, value, min, max)),
        MathError::StrictlyOutOfBounds { value, min, max } => Some((true, value, min, max)),
        _ => None,
    }
}

impl PartialEq for MathError {
    fn eq(&self, other: &Self) -> bool {
        if let (Some((s1, v1, mn1, mx1)), Some((s2, v2, mn2, mx2))) =
            (bounds_key(*self), bounds_key(*other))
        {
            return s1 == s2
                && f64_eq_canonical(v1, v2)
                && f64_eq_canonical(mn1, mn2)
                && f64_eq_canonical(mx1, mx2);
        }
        match (*self, *other) {
            (
                Self::DimensionMismatch {
                    expected: e1,
                    actual: a1,
                },
                Self::DimensionMismatch {
                    expected: e2,
                    actual: a2,
                },
            ) => e1 == e2 && a1 == a2,
            (
                Self::InvalidPermutation {
                    index: i1,
                    value: v1,
                    len: l1,
                },
                Self::InvalidPermutation {
                    index: i2,
                    value: v2,
                    len: l2,
                },
            ) => (i1, v1, l1) == (i2, v2, l2),
            (
                Self::WorkspaceTooSmall {
                    required: r1,
                    available: a1,
                },
                Self::WorkspaceTooSmall {
                    required: r2,
                    available: a2,
                },
            ) => (r1, a1) == (r2, a2),
            (Self::NumericalError { reason: r1 }, Self::NumericalError { reason: r2 })
            | (Self::ConversionError { reason: r1 }, Self::ConversionError { reason: r2 }) => {
                r1 == r2
            }
            _ => false,
        }
    }
}

impl Eq for MathError {}

impl CoreDiagnostic for MathError {
    fn error_type(&self) -> &'static str {
        "MathError"
    }

    fn code(&self) -> &'static str {
        match self {
            Self::DimensionMismatch { .. } => "MATH-001",
            Self::OutOfBounds { .. } => "MATH-002",
            Self::StrictlyOutOfBounds { .. } => "MATH-003",
            Self::InvalidPermutation { .. } => "MATH-004",
            Self::WorkspaceTooSmall { .. } => "MATH-005",
            Self::NumericalError { .. } => "MATH-006",
            Self::ConversionError { .. } => "MATH-007",
        }
    }
}

#[cfg(feature = "std")]
impl Diagnostic for MathError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), self.error_type().to_string());
        map.insert("code".to_string(), self.code().to_string());
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
            Self::WorkspaceTooSmall {
                required,
                available,
            } => {
                map.insert("required".to_string(), required.to_string());
                map.insert("available".to_string(), available.to_string());
            }
            Self::NumericalError { reason } | Self::ConversionError { reason } => {
                map.insert("reason".to_string(), (*reason).to_string());
            }
        }
        map
    }
}

#[allow(missing_docs)]
pub type MathResult<T> = core::result::Result<T, MathError>;

impl From<&'static str> for MathError {
    fn from(reason: &'static str) -> Self {
        Self::ConversionError { reason }
    }
}

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

impl CoreDiagnostic for FitError {
    fn error_type(&self) -> &'static str {
        "FitError"
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InsufficientPoints { .. } => "MATH-FIT-001",
            Self::UnsupportedDegree { .. } => "MATH-FIT-002",
            Self::InvalidDomain { .. } => "MATH-FIT-003",
            Self::SingularMatrix => "MATH-FIT-004",
            Self::NonConvergence { .. } => "MATH-FIT-005",
        }
    }
}

#[cfg(feature = "std")]
impl Diagnostic for FitError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), self.error_type().to_string());
        map.insert("code".to_string(), self.code().to_string());
        map.insert("description".to_string(), self.to_string());
        match self {
            Self::InsufficientPoints { required, actual } => {
                map.insert("required".to_string(), required.to_string());
                map.insert("actual".to_string(), actual.to_string());
            }
            Self::UnsupportedDegree { degree } => {
                map.insert("degree".to_string(), degree.to_string());
            }
            Self::InvalidDomain { reason } => {
                map.insert("reason".to_string(), (*reason).to_string());
            }
            Self::SingularMatrix => {}
            Self::NonConvergence { iterations } => {
                map.insert("iterations".to_string(), iterations.to_string());
            }
        }
        map
    }
}

#[cfg(feature = "std")]
mod umbrella;
#[cfg(feature = "std")]
pub use umbrella::MathCommonsError;

#[cfg(test)]
mod core_tests {
    use super::*;

    fn assert_stack_error_traits<
        E: Copy + Clone + PartialEq + Eq + core::fmt::Display + core::error::Error + CoreDiagnostic,
    >() {
    }

    #[test]
    fn math_error_is_copy_eq_and_core_diagnostic_without_alloc() {
        assert_stack_error_traits::<MathError>();
        let ws = MathError::WorkspaceTooSmall {
            required: 16,
            available: 8,
        };
        let ws_copy = ws;
        assert_eq!(ws, ws_copy);
        assert_eq!(ws.code(), "MATH-005");
        assert_eq!(ws.error_type(), "MathError");
        assert!(core::error::Error::source(&ws).is_none());

        let nan_bounds = MathError::OutOfBounds {
            value: f64::NAN,
            min: -0.0,
            max: 1.0,
        };
        let pos_zero_bounds = MathError::OutOfBounds {
            value: f64::NAN,
            min: 0.0,
            max: 1.0,
        };
        assert_eq!(nan_bounds, pos_zero_bounds);
        assert_eq!(
            MathError::from("overflow"),
            MathError::ConversionError { reason: "overflow" }
        );
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use std::error::Error as StdError;

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
            MathCommonsError::from(MathError::WorkspaceTooSmall {
                required: 10,
                available: 4,
            }),
            MathCommonsError::from(FitError::SingularMatrix),
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
