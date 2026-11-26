//! Typed zero-allocation error taxonomy for Monte Carlo, quasi-Monte Carlo, and bootstrap engines.

use thiserror::Error;

/// Errors raised by Monte Carlo, quasi-Monte Carlo, and nonparametric bootstrap routines.
///
/// Every variant is `Copy` so error construction and `?` propagation perform zero dynamic heap
/// allocations inside `VerifiedAllocator`-locked simulation loops.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum MonteCarloError {
    /// Requested dimension is zero or exceeds the sequence/workspace capacity.
    #[error("invalid dimension {dimension} (must be in 1..={max_dimension})")]
    InvalidDimension {
        /// Requested coordinate dimension.
        dimension: usize,
        /// Maximum supported dimension for the target generator or workspace.
        max_dimension: usize,
    },

    /// Sample count is zero or exceeds the preallocated workspace capacity.
    #[error("invalid sample count {samples} (must be positive and within workspace capacity)")]
    InvalidSampleCount {
        /// Requested sample count.
        samples: usize,
    },

    /// Confidence level lies outside the open interval `(0.0, 1.0)` or is non-finite.
    #[error("invalid confidence level {confidence} (must be finite and strictly in (0, 1))")]
    InvalidConfidenceLevel {
        /// Requested confidence level.
        confidence: f64,
    },

    /// Custom Halton radical-inverse base is composite or less than `2`.
    #[error("invalid Halton base {base} (must be a prime >= 2)")]
    InvalidBase {
        /// Supplied base value.
        base: u64,
    },

    /// Caller-supplied slice length did not match the expected dimension or sample count.
    #[error("buffer length mismatch: expected {expected}, got {actual}")]
    BufferLengthMismatch {
        /// Required slice length.
        expected: usize,
        /// Actual slice length provided.
        actual: usize,
    },

    /// Integrand, statistic, or proposal weight evaluated to `NaN` or `+-inf`.
    #[error("non-finite integrand, weight, or statistic evaluation encountered")]
    NonFiniteEvaluation,

    /// Self-normalized importance weights collapsed onto too few particles or summed to zero.
    #[error("degenerate importance weights (effective sample size = {effective_sample_size})")]
    DegenerateWeights {
        /// Computed Kong effective sample size (`(sum w)^2 / sum w^2`).
        effective_sample_size: f64,
    },

    /// Integration hyperrectangle bounds are non-finite or inverted (`lower > upper`).
    #[error("invalid integration bounds at dimension {dimension}: [{lower}, {upper}]")]
    InvalidBounds {
        /// Coordinate axis index with invalid bounds.
        dimension: usize,
        /// Lower bound supplied for the axis.
        lower: f64,
        /// Upper bound supplied for the axis.
        upper: f64,
    },

    /// Input dataset has fewer observations than required by the estimator.
    #[error("insufficient observations: required at least {required}, got {actual}")]
    InsufficientData {
        /// Minimum required number of observations.
        required: usize,
        /// Actual number of observations provided.
        actual: usize,
    },

    /// Bootstrap replicate count was zero or exceeded preallocated capacity.
    #[error("invalid bootstrap replicate count (must be positive and within workspace capacity)")]
    ZeroReplicates,
}

// [cite:clinical_trials_statistics]
