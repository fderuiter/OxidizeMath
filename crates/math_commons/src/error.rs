// [cite:math_commons]
#[cfg(feature = "std")]
pub use crate::ast::{EvalError, GlslError, LexError, ParseError, ShaderEmitError};
use crate::math_kernel::types::Dimension;
use alloc::string::String;
#[cfg(feature = "std")]
use diagnostics::{Diagnostic, Severity};
#[cfg(feature = "std")]
use std::collections::HashMap;
use thiserror::Error;

/// Centralized mathematical error type.
#[derive(Debug, Clone, PartialEq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Numerical computation failures (e.g., singular matrix, non-positive definite).
    #[error("Numerical error: {reason}")]
    #[allow(missing_docs)]
    NumericalError { reason: String },

    /// Conversion failures (e.g., failed to convert between numeric types).
    #[error("Conversion error: {reason}")]
    #[allow(missing_docs)]
    ConversionError { reason: String },
}

#[cfg(feature = "std")]
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
pub type MathResult<T> = core::result::Result<T, MathError>;

impl From<String> for MathError {
    fn from(reason: String) -> Self {
        Self::ConversionError { reason }
    }
}

#[cfg(feature = "serde")]
const KNOWN_INVALID_DOMAIN_REASONS: &[&str] = &[
    "non-finite exponential seed parameters",
    "exponential fit requires at least 2 points with y > 0",
    "power-law fit requires at least 2 points with x > 0 and y > 0",
    "non-finite power-law seed parameters",
    "logarithmic fit requires at least 2 points with x > 0",
    "logistic fit requires positive response values",
    "sinusoidal fit requires non-degenerate x span",
    "polynomial FitModel passed to fit_nonlinear",
    "exponential log-space fit requires at least 2 points with y > 0",
    "power-law log-space fit requires at least 2 points with x > 0 and y > 0",
    "insufficient positive coordinates for logarithmic polynomial fit",
    "non-finite initial residual sum of squares",
    "cycle count in calibration series must be non-negative",
    "state-of-health capacity fraction must lie within [0.0, 1.2]",
    "calibration series must include positive cycle counts",
    "depth of discharge must be strictly positive for cycle calibration",
    "non-finite calibrated battery aging parameters",
    "non-finite calibrated residual sum of squares",
    "custom formula requires fit_custom_evaluator",
    "invalid curve-fitting domain",
];

#[cfg(feature = "serde")]
struct StaticReason(&'static str);

#[cfg(feature = "serde")]
fn deserialize_static_reason<'de, D>(deserializer: D) -> Result<StaticReason, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct StaticReasonVisitor;

    impl serde::de::Visitor<'_> for StaticReasonVisitor {
        type Value = StaticReason;

        fn expecting(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            formatter.write_str("a curve-fitting domain reason string")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            for &known in KNOWN_INVALID_DOMAIN_REASONS {
                if known == v {
                    return Ok(StaticReason(known));
                }
            }
            Ok(StaticReason("invalid curve-fitting domain"))
        }
    }

    deserializer.deserialize_str(StaticReasonVisitor)
}

#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
enum FitErrorRepr {
    InsufficientPoints {
        required: usize,
        actual: usize,
    },
    UnsupportedDegree {
        degree: usize,
    },
    InvalidDomain {
        #[serde(deserialize_with = "deserialize_static_reason")]
        reason: StaticReason,
    },
    SingularMatrix,
    NonConvergence {
        iterations: usize,
    },
}

#[cfg(feature = "serde")]
impl From<FitErrorRepr> for FitError {
    fn from(repr: FitErrorRepr) -> Self {
        match repr {
            FitErrorRepr::InsufficientPoints { required, actual } => {
                Self::InsufficientPoints { required, actual }
            }
            FitErrorRepr::UnsupportedDegree { degree } => Self::UnsupportedDegree { degree },
            FitErrorRepr::InvalidDomain {
                reason: StaticReason(reason),
            } => Self::InvalidDomain { reason },
            FitErrorRepr::SingularMatrix => Self::SingularMatrix,
            FitErrorRepr::NonConvergence { iterations } => Self::NonConvergence { iterations },
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for FitError {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        FitErrorRepr::deserialize(deserializer).map(Self::from)
    }
}

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
}

#[cfg(feature = "std")]
impl Diagnostic for FitError {
    fn severity(&self) -> Severity {
        Severity::Error
    }

    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "FitError".to_string());
        map.insert("description".to_string(), self.to_string());
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

#[cfg(feature = "std")]
mod umbrella;
#[cfg(feature = "std")]
pub use umbrella::MathCommonsError;

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
