//! # Shannon Information Theory & Channel Capacity
//!
//! Zero-allocation `f64` routines (in base-2 bits) for discrete Shannon entropy, Rényi entropy,
//! Kullback-Leibler and Jensen-Shannon divergences, joint mutual information, closed-form channel
//! capacities (`BSC`, `BEC`, `AWGN`), and the scratch-buffered Blahut-Arimoto solver for general
//! discrete memoryless channels (`DMC`).
//!
//! ## Quick Start
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::algorithmic_information::shannon::{
//!     BlahutArimotoWorkspace, blahut_arimoto_capacity, bsc_capacity, categorical_entropy,
//! };
//!
//! let fair = [0.25, 0.25, 0.25, 0.25];
//! assert!((categorical_entropy(&fair).unwrap() - 2.0).abs() < 1e-12);
//!
//! let p = 0.11;
//! let closed = bsc_capacity(p).unwrap();
//! let transition = [1.0 - p, p, p, 1.0 - p];
//! let mut ws = BlahutArimotoWorkspace::new(2, 2);
//! let mut opt_input = [0.0; 2];
//! let res = blahut_arimoto_capacity(&transition, 2, 2, 1e-10, 200, &mut ws, &mut opt_input)
//!     .unwrap();
//! assert!((res.capacity_bits - closed).abs() < 1e-9);
//! ```

mod channel;
mod entropy;

pub use channel::{
    BlahutArimotoWorkspace, ChannelCapacityResult, awgn_bandwidth_capacity, awgn_capacity,
    bec_capacity, binary_entropy, blahut_arimoto_capacity, bsc_capacity,
};
pub use entropy::{
    JointMiWorkspace, categorical_entropy, jensen_shannon_divergence, joint_mutual_information,
    kl_divergence, renyi_entropy, validate_pmf,
};

use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

/// Stack-only error type for discrete Shannon entropy, divergence, and channel capacity routines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ShannonError {
    /// Probability distribution slice cannot be empty.
    EmptyDistribution,
    /// Probability mass entry lies outside `[0, 1]` or is non-finite.
    InvalidProbability {
        /// Offending probability value.
        value: f64,
    },
    /// Probability distribution entries do not sum to `1.0` within numerical tolerance.
    UnnormalizedDistribution {
        /// Observed probability mass sum.
        sum: f64,
    },
    /// Slice or matrix dimensions did not match expected sizes.
    DimensionMismatch {
        /// Expected element count.
        expected: usize,
        /// Actual element count.
        actual: usize,
    },
    /// Reference distribution `Q` assigns zero mass where `P` assigns positive mass.
    SupportMismatch {
        /// First offending outcome index where `P[index] > 0` and `Q[index] == 0`.
        index: usize,
    },
    /// Rényi order `alpha` must be non-negative and not `NaN` (`+inf` is allowed for min-entropy).
    InvalidRenyiOrder {
        /// Offending Rényi order parameter.
        alpha: f64,
    },
    /// Closed-form channel parameter (crossover probability, erasure probability, SNR, or bandwidth) is invalid.
    InvalidChannelParameter {
        /// Offending channel parameter value.
        value: f64,
    },
    /// Discrete memoryless channel transition matrix row is invalid or unnormalized.
    InvalidChannelMatrix {
        /// Offending input symbol row index.
        row: usize,
        /// Observed row sum (or `NaN` if an entry was invalid).
        sum: f64,
    },
    /// Preallocated scratch buffer or caller output slice is smaller than required.
    WorkspaceTooSmall {
        /// Minimum required capacity.
        required: usize,
        /// Provided capacity.
        actual: usize,
    },
    /// Iterative Blahut-Arimoto solver reached `max_iterations` before closing the duality gap.
    ConvergenceFailed {
        /// Number of iterations executed.
        iterations: usize,
        /// Remaining upper-minus-lower bound gap in bits.
        gap: f64,
    },
}

impl std::fmt::Display for ShannonError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyDistribution => write!(f, "probability distribution cannot be empty"),
            Self::InvalidProbability { value } => {
                write!(
                    f,
                    "probability entry must be finite and in [0, 1] (got {value})"
                )
            }
            Self::UnnormalizedDistribution { sum } => {
                write!(f, "probability distribution must sum to 1.0 (got {sum})")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "distribution length mismatch (expected {expected}, got {actual})"
                )
            }
            Self::SupportMismatch { index } => {
                write!(
                    f,
                    "kl divergence undefined: p[{index}] > 0 while reference q[{index}] == 0"
                )
            }
            Self::InvalidRenyiOrder { alpha } => {
                write!(
                    f,
                    "renyi order alpha must be >= 0 and not NaN (got {alpha})"
                )
            }
            Self::InvalidChannelParameter { value } => {
                write!(f, "invalid channel capacity parameter (got {value})")
            }
            Self::InvalidChannelMatrix { row, sum } => {
                write!(
                    f,
                    "channel transition matrix row {row} must be a valid distribution (sum {sum})"
                )
            }
            Self::WorkspaceTooSmall { required, actual } => {
                write!(
                    f,
                    "scratch buffer too small (required at least {required}, got {actual})"
                )
            }
            Self::ConvergenceFailed { iterations, gap } => {
                write!(
                    f,
                    "blahut-arimoto failed to converge after {iterations} iterations (gap {gap})"
                )
            }
        }
    }
}

impl std::error::Error for ShannonError {}

impl Diagnostic for ShannonError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "ShannonError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

#[cfg(test)]
mod tests;

// [cite:algorithmic_information_rust]
