//! Error types for the `#![no_std]` probability and statistical inference engine.
#![cfg_attr(any(), verified(opt_out = "error_types"))]

use core::fmt;

/// Domain, dimension, and numerical convergence errors produced by `engine_prob`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProbError {
    /// Distribution or model parameter violates mathematical domain constraints.
    InvalidParameter {
        /// Parameter identifier.
        name: &'static str,
        /// Invalid numeric value supplied.
        value: f64,
    },
    /// Probability argument lies outside `[0, 1]` or `(0, 1)` as required.
    InvalidProbability {
        /// Invalid probability scalar.
        value: f64,
    },
    /// Input slice is unexpectedly empty.
    EmptyInput {
        /// Context describing the empty slice.
        context: &'static str,
    },
    /// Slice lengths do not match expected dimensions.
    DimensionMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count provided.
        actual: usize,
    },
    /// Caller-supplied scratch or output slice is smaller than required.
    BufferTooSmall {
        /// Minimum required buffer length.
        required: usize,
        /// Actual buffer length provided.
        actual: usize,
    },
    /// Iterative numerical algorithm failed to converge or encountered non-finite values.
    NumericalFailure {
        /// Static description of the failure mode.
        reason: &'static str,
    },
}

impl fmt::Display for ProbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::InvalidParameter { name, value } => {
                write!(f, "invalid parameter '{name}': {value}")
            }
            Self::InvalidProbability { value } => {
                write!(
                    f,
                    "invalid probability: {value} (must be finite and in valid interval)"
                )
            }
            Self::EmptyInput { context } => {
                write!(f, "empty input in {context}")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
            Self::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "scratch/output buffer too small: required at least {required}, got {actual}"
                )
            }
            Self::NumericalFailure { reason } => {
                write!(f, "numerical failure: {reason}")
            }
        }
    }
}

impl core::error::Error for ProbError {}
