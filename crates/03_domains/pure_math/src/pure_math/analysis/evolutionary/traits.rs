// [cite:pure_math]
use nalgebra::DVector;
use thiserror::Error;

/// Errors returned by game-theoretic equilibrium, simplex, and mean-field evolutionary solvers.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum GameTheoryError {
    /// Payoff matrix is not square.
    #[error("Payoff matrix must be square, got {rows}x{cols}")]
    NonSquarePayoffMatrix {
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// Invalid parameter value (e.g. frequency outside `[0, 1]`).
    #[error("Invalid game theory parameter {name}: {value}")]
    InvalidParameter {
        /// Parameter identifier.
        name: String,
        /// Rejected numeric value.
        value: f64,
    },
    /// Lower and upper bound vectors had different lengths.
    #[error(
        "Box constraint bound lengths must match: min has {min_len} elements, max has {max_len}"
    )]
    BoundsDimensionMismatch {
        /// Length of the lower-bound vector.
        min_len: usize,
        /// Length of the upper-bound vector.
        max_len: usize,
    },
    /// Lower bound exceeded upper bound or was non-finite along a dimension.
    #[error(
        "Invalid box bounds at dimension {dimension}: min ({min}) must be finite and <= max ({max})"
    )]
    InvalidBoxBounds {
        /// Zero-based coordinate index with invalid bounds.
        dimension: usize,
        /// Lower bound at `dimension`.
        min: f64,
        /// Upper bound at `dimension`.
        max: f64,
    },
    /// Wrapped centralized mathematical error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
    /// Wrapped Monte Carlo estimation error.
    #[error("{0}")]
    MonteCarlo(#[from] crate::error::MonteCarloError),
}

/// Frequency-dependent population fitness operator $\mathbf{f}: \Delta^{n-1} \to \mathbb{R}^n$.
///
/// Evaluates strategy fitness $f_i(\mathbf{x})$ in-place into caller-owned `out` for replicator
/// and mean-field evolutionary dynamics:
///
/// $$\dot{x}_i = x_i \left(f_i(\mathbf{x}) - \sum_{j=1}^{n} x_j f_j(\mathbf{x})\right)$$
///
/// Reference: Taylor & Jonker (1978), *Math. Biosci.* 40(1–2), 145–156, DOI: 10.1016/0025-5564(78)90077-9; see
/// [`docs/math_specs/pure_math.md`](../../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Implementations writing finite payoffs into `out` without heap allocation preserve zero-allocation
/// simplex integration under explicit Runge-Kutta steppers.
///
/// # Domain of Validity
/// Population state `x` and output buffer `out` must have equal strategy dimension $n \ge 1$.
pub trait FitnessStrategy {
    /// Computes the fitness vector $\mathbf{f}(\mathbf{x})$ for population state `x` into `out`.
    #[verified_engine::verified]
    fn fitness(&self, x: &DVector<f64>, out: &mut DVector<f64>);
}
