//! Typed `#![no_std]` error taxonomy for `engine_prob`.

use core::fmt;

/// Error variants returned by `#![no_std]` cubature, variance-reduction, and bootstrap solvers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProbError {
    /// Input sample slice was empty.
    EmptySample,
    /// Iteration, point, pair, or resample count was zero or invalid.
    InvalidCount {
        /// Parameter name that failed validation.
        name: &'static str,
        /// Provided invalid count.
        value: usize,
    },
    /// Requested dimensionality was zero or exceeded sequence generator limits.
    InvalidDimension {
        /// Requested dimension.
        dim: usize,
        /// Maximum supported dimension for the selected algorithm.
        max_dim: usize,
    },
    /// Hyperrectangle integration bounds were inverted (`lower > upper`) or mismatched.
    InvalidBounds {
        /// Coordinate axis index where bounds failed validation.
        dim_index: usize,
        /// Lower integration bound.
        lower: f64,
        /// Upper integration bound.
        upper: f64,
    },
    /// Confidence level was outside the open unit interval $(0, 1)$ or non-finite.
    InvalidConfidence {
        /// Provided confidence level.
        confidence: f64,
    },
    /// Input data, weights, or integrand evaluation produced `NaN` or `±Inf`.
    NonFiniteInput {
        /// Diagnostic label identifying the non-finite input source.
        context: &'static str,
    },
    /// Caller-provided [`crate::workspace::SolverWorkspace`] lacked sufficient scratch elements.
    InsufficientWorkspace {
        /// Minimum `f64` elements required.
        required_floats: usize,
        /// Available `f64` elements in workspace.
        available_floats: usize,
        /// Minimum `usize` index elements required.
        required_indices: usize,
        /// Available `usize` index elements in workspace.
        available_indices: usize,
    },
    /// Correlation or covariance matrix failed positive-definiteness or shape checks.
    DegenerateMatrix {
        /// Static explanation of the matrix failure.
        reason: &'static str,
    },
    /// Importance weights were negative or summed to zero.
    ZeroEffectiveWeight,
}

impl fmt::Display for ProbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptySample => f.write_str("sample slice must not be empty"),
            Self::InvalidCount { name, value } => {
                write!(f, "invalid count for {name}: {value} (must be >= 1)")
            }
            Self::InvalidDimension { dim, max_dim } => {
                write!(f, "dimension {dim} outside supported range 1..={max_dim}")
            }
            Self::InvalidBounds {
                dim_index,
                lower,
                upper,
            } => write!(
                f,
                "invalid integration bounds at axis {dim_index}: [{lower}, {upper}]"
            ),
            Self::InvalidConfidence { confidence } => {
                write!(f, "confidence level {confidence} must lie strictly in (0, 1)")
            }
            Self::NonFiniteInput { context } => {
                write!(f, "non-finite floating-point value encountered in {context}")
            }
            Self::InsufficientWorkspace {
                required_floats,
                available_floats,
                required_indices,
                available_indices,
            } => write!(
                f,
                "insufficient SolverWorkspace capacity: need ({required_floats} floats, {required_indices} indices), got ({available_floats} floats, {available_indices} indices)"
            ),
            Self::DegenerateMatrix { reason } => write!(f, "degenerate matrix: {reason}"),
            Self::ZeroEffectiveWeight => {
                f.write_str("self-normalized importance weights must be >= 0 with positive sum")
            }
        }
    }
}

impl core::error::Error for ProbError {}
