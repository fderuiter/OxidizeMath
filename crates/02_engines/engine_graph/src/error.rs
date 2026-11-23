//! Error types for the `#![no_std]` graphical model and belief propagation engine.
#![cfg_attr(any(), verified(opt_out = "error_types"))]

use core::fmt;

/// Structural, dimension, and numerical errors produced by `engine_graph`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GraphEngineError {
    /// Graph or factor specification is empty.
    EmptyGraph {
        /// Context describing the empty component.
        context: &'static str,
    },
    /// Variable index or state cardinality violates graph bounds.
    InvalidVariable {
        /// Variable index.
        var: usize,
        /// Invalid value or cardinality.
        value: usize,
    },
    /// Factor specification has invalid arity, duplicate variables, or out-of-range table slice.
    InvalidFactor {
        /// Static description of the factor violation.
        reason: &'static str,
    },
    /// Potential or parameter value is negative, NaN, or identically zero across a factor.
    InvalidPotential {
        /// Static description of the numerical violation.
        reason: &'static str,
    },
    /// Caller-supplied output or scratch slice is smaller than required.
    BufferTooSmall {
        /// Minimum required length.
        required: usize,
        /// Actual slice length provided.
        actual: usize,
    },
    /// Dimension mismatch across state, transition, or observation buffers.
    DimensionMismatch {
        /// Expected length.
        expected: usize,
        /// Actual length provided.
        actual: usize,
    },
}

impl fmt::Display for GraphEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::EmptyGraph { context } => write!(f, "empty graph component in {context}"),
            Self::InvalidVariable { var, value } => {
                write!(f, "invalid variable {var} (cardinality/index {value})")
            }
            Self::InvalidFactor { reason } => write!(f, "invalid factor: {reason}"),
            Self::InvalidPotential { reason } => write!(f, "invalid potential: {reason}"),
            Self::BufferTooSmall { required, actual } => {
                write!(
                    f,
                    "buffer too small: required at least {required}, got {actual}"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl core::error::Error for GraphEngineError {}
