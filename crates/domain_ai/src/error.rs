use thiserror::Error;

/// Error variants produced by AI architectures, optimizers, and training loops.
#[non_exhaustive]
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AIError {
    /// Invalid parameter value (e.g. negative learning rate).
    #[error("Invalid parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter identifier.
        name: String,
        /// Rejected numeric value.
        value: f64,
    },

    /// Optimization failed or diverged.
    #[error("Optimization diverged")]
    OptimizationDivergence,

    /// State is uninitialized.
    #[error("Uninitialized state: {name}")]
    UninitializedState {
        /// Name of the uninitialized state buffer or parameter.
        name: String,
    },

    /// Model dimension is incompatible with the requested attention head count.
    #[error(
        "Model dimension d_model ({d_model}) must be divisible by positive head count h ({heads})"
    )]
    InvalidHeadConfiguration {
        /// Embedding dimension `d_model`.
        d_model: usize,
        /// Requested number of attention heads `h`.
        heads: usize,
    },

    /// Number of per-head weight matrices did not match head count `h`.
    #[error(
        "Expected {expected} weight matrices per projection type, got w_q: {got_q}, w_k: {got_k}, w_v: {got_v}"
    )]
    HeadCountMismatch {
        /// Expected number of heads `h`.
        expected: usize,
        /// Supplied query projection count.
        got_q: usize,
        /// Supplied key projection count.
        got_k: usize,
        /// Supplied value projection count.
        got_v: usize,
    },

    /// Output projection weight matrix had incompatible dimensions.
    #[error(
        "{projection} dimension mismatch: expected ({}, {}), got ({}, {})",
        expected.0,
        expected.1,
        actual.0,
        actual.1
    )]
    ProjectionShapeMismatch {
        /// Name of the projection matrix (`w_o`).
        projection: &'static str,
        /// Expected `(nrows, ncols)` shape.
        expected: (usize, usize),
        /// Actual `(nrows, ncols)` shape.
        actual: (usize, usize),
    },

    /// Per-head projection weight matrix had incompatible dimensions.
    #[error(
        "Head {head} {projection} dimension mismatch: expected ({}, {}), got ({}, {})",
        expected.0,
        expected.1,
        actual.0,
        actual.1
    )]
    HeadProjectionShapeMismatch {
        /// Zero-based attention head index.
        head: usize,
        /// Name of the head projection matrix (`w_q`, `w_k`, or `w_v`).
        projection: &'static str,
        /// Expected `(nrows, ncols)` shape.
        expected: (usize, usize),
        /// Actual `(nrows, ncols)` shape.
        actual: (usize, usize),
    },

    /// Row-major cell mask slice length did not match `width * height`.
    #[error("Cell mask length {actual} does not match expected grid size {expected}")]
    CellMaskLengthMismatch {
        /// Expected cell count (`width * height`).
        expected: usize,
        /// Actual slice length supplied.
        actual: usize,
    },

    /// Byte tag in a serialized grid mask did not map to a valid `GridCellRole`.
    #[error("Invalid GridCellRole tag {code} at cell index {index}")]
    InvalidCellRoleTag {
        /// Zero-based cell index where the invalid tag occurred.
        index: usize,
        /// Rejected raw byte tag.
        code: u8,
    },

    /// Wrapped foundational error from `math_commons`.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),

    /// Wrapped `math_commons` umbrella error (AST, QMC, RNG, fit, or linalg).
    #[error("{0}")]
    MathCommons(#[from] math_commons::error::MathCommonsError),

    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] math_commons::error::LinalgError),

    /// Foundational `#![no_std]` differential-equation solver error.
    #[error("{0}")]
    Diffeq(#[from] engine_diffeq::DiffeqError),

    /// Foundational curve-fitting solver error.
    #[error("{0}")]
    Fit(#[from] math_commons::error::FitError),

    /// Upstream numerical optimization error.
    #[error("{0}")]
    Optimization(#[from] pure_math::error::OptimizationError),

    /// Upstream numerical linear algebra error.
    #[error("{0}")]
    LinearAlgebra(#[from] pure_math::error::LinearAlgebraError),
}

/// Standard result alias used across `domain_ai` computations.
pub type Result<T> = math_commons::error::MathResult<T>;

mod conversions;

#[cfg(test)]
mod tests {
    use super::*;
    use diagnostics::{Diagnostic, Severity};
    use std::error::Error as StdError;

    #[test]
    #[verified_engine::verified]
    fn test_ai_error_math_and_commons_preserve_source() {
        let err = AIError::from(math_commons::error::MathError::DimensionMismatch {
            expected: math_commons::math_kernel::types::Dimension(2),
            actual: math_commons::math_kernel::types::Dimension(3),
        });
        assert!(err.source().is_some());
        assert_eq!(err.severity(), Severity::Error);

        let qmc = AIError::from(math_commons::error::QmcError::InvalidBase { base: 1 });
        let rng = AIError::from(math_commons::error::RngError::ZeroUpperBound);
        let parse = AIError::from(math_commons::error::ParseError::EmptyExpression);
        let static_str = AIError::from("invalid grid");
        assert!(qmc.source().is_some());
        assert!(rng.source().is_some());
        assert!(parse.source().is_some());
        assert!(static_str.source().is_some());
    }
}
