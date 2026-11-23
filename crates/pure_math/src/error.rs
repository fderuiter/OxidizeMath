use thiserror::Error;

#[allow(missing_docs)]
pub type Result<T> = math_commons::error::MathResult<T>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum NumberTheoryError {
    #[error("Failed to parse integer from string: {0}")]
    #[allow(missing_docs)]
    ParseError(String),

    #[error("Invalid integer radix {radix} (must be between 2 and 36)")]
    #[allow(missing_docs)]
    InvalidRadix { radix: u32 },

    #[error("Failed to parse integer literal in base {radix}")]
    #[allow(missing_docs)]
    InvalidIntegerLiteral { radix: u32 },

    #[error("Modulus must be non-zero")]
    #[allow(missing_docs)]
    ZeroModulus,

    #[error("Element has no multiplicative inverse for the given modulus")]
    #[allow(missing_docs)]
    NoModularInverse,

    #[error("Cannot invert zero element in finite field")]
    #[allow(missing_docs)]
    DivisionByZeroFieldElement,

    #[error("Modulo operation failed")]
    #[allow(missing_docs)]
    ModuloError,

    #[error("Conversion to usize failed")]
    #[allow(missing_docs)]
    ConversionError,

    #[error("Division by zero QSeries")]
    #[allow(missing_docs)]
    DivisionByZeroQSeries,

    #[error("Division by a QSeries with zero constant term")]
    #[allow(missing_docs)]
    DivisionByZeroConstantTerm,
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum KellyError {
    /// Invalid probability (must be between 0 and 1).
    #[error("Invalid probability: {value} (must be between 0 and 1)")]
    #[allow(missing_docs)]
    InvalidProbability { value: f64 },
    /// Invalid odds (must be greater than 1.0 for decimal odds).
    #[error("Invalid odds: {value} (must be > 1.0)")]
    #[allow(missing_docs)]
    InvalidOdds { value: f64 },
    /// Invalid fraction (must be between 0 and 1).
    #[error("Invalid fraction: {value} (must be between 0 and 1)")]
    #[allow(missing_docs)]
    InvalidFraction { value: f64 },
    /// No edge (negative expected value - should not bet).
    #[error("No edge: p={probability}, odds={odds} results in negative expectation")]
    #[allow(missing_docs)]
    NoEdge { probability: f64, odds: f64 },
    /// Invalid bankroll amount (must be positive).
    #[error("Invalid bankroll: {value} (must be positive)")]
    #[allow(missing_docs)]
    InvalidBankroll { value: f64 },
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum MarkovError {
    /// Invalid probability value (must be in [0, 1] and finite).
    #[error("Invalid probability {value}: must be finite and within [0, 1]")]
    InvalidProbability {
        /// The invalid probability value.
        value: f64,
    },

    /// Matrix is not stochastic (rows don't sum to 1).
    #[error("Transition matrix is not stochastic: {reason}")]
    NotStochastic {
        /// Description of the issue.
        reason: String,
    },

    /// Matrix is not a valid generator (rows don't sum to 0 or invalid rates).
    #[error("Invalid generator matrix: {reason}")]
    InvalidGenerator {
        /// Description of the issue.
        reason: String,
    },

    /// Invalid state specification.
    #[error("Invalid Markov state: {reason}")]
    InvalidState {
        /// Description of the issue.
        reason: String,
    },

    /// Time index out of bounds.
    #[error(
        "Time index {time} is out of valid range [{}, {}]",
        valid_range.0,
        valid_range.1
    )]
    TimeIndexOutOfBounds {
        /// The requested time.
        time: f64,
        /// Valid time range.
        valid_range: (f64, f64),
    },

    /// Observation sequence incompatible with model.
    #[error("Invalid observation sequence: {reason}")]
    InvalidObservation {
        /// Description of the issue.
        reason: String,
    },

    /// Matrix not invertible or singular.
    #[error("Singular matrix in {context}")]
    SingularMatrix {
        /// Context about which matrix.
        context: String,
    },

    #[error("Mathematical error: {0}")]
    #[allow(missing_docs)]
    Math(#[from] math_commons::error::MathError),
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum OuError {
    /// Invalid mean reversion rate (must be positive).
    #[error("Invalid mean reversion rate: {value} (must be positive)")]
    #[allow(missing_docs)]
    InvalidMeanReversionRate { value: f64 },
    /// Invalid volatility (must be non-negative).
    #[error("Invalid volatility: {value} (must be non-negative)")]
    #[allow(missing_docs)]
    InvalidVolatility { value: f64 },
    /// Invalid time step (must be positive).
    #[error("Invalid time step: {value} (must be positive)")]
    #[allow(missing_docs)]
    InvalidTimeStep { value: f64 },
    /// Invalid simulation parameters.
    #[error("Invalid simulation parameters: {reason}")]
    #[allow(missing_docs)]
    InvalidSimulationParams { reason: String },
    /// Insufficient data for parameter estimation.
    #[error("Insufficient data: required at least {required}, got {actual}")]
    #[allow(missing_docs)]
    InsufficientData { required: usize, actual: usize },
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum ZipError {
    /// Invalid probability parameter (must be in [0, 1]).
    #[error("Invalid probability {value} for parameter '{parameter}' (must be in [0, 1])")]
    #[allow(missing_docs)]
    InvalidProbability { value: f64, parameter: String },
    /// Invalid rate parameter (must be positive).
    #[error("Invalid Poisson rate {value} (must be strictly positive)")]
    #[allow(missing_docs)]
    InvalidRate { value: f64 },
    /// Invalid count value (must be non-negative integer).
    #[error("Invalid count {value} (must be a finite non-negative integer)")]
    #[allow(missing_docs)]
    InvalidCount { value: f64 },
    /// Insufficient data for regression.
    #[error(
        "Insufficient observations for ZIP regression: required at least {required}, got {actual}"
    )]
    #[allow(missing_docs)]
    InsufficientData { required: usize, actual: usize },
    /// Regression convergence failed.
    #[error("ZIP regression failed to converge after {iterations} iterations: {reason}")]
    #[allow(missing_docs)]
    ConvergenceFailed { iterations: usize, reason: String },
    /// Invalid predictor matrix dimensions.
    #[error("Invalid predictor matrix dimensions: expected {expected}, got {actual}")]
    #[allow(missing_docs)]
    InvalidDimensions { expected: String, actual: String },
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum TdaError {
    /// Empty point cloud provided.
    #[error("point cloud cannot be empty")]
    EmptyPointCloud,
    /// Invalid radius (must be non-negative).
    #[error("filtration radius must be non-negative and finite (got {value})")]
    #[allow(missing_docs)]
    InvalidRadius { value: f64 },
    /// Invalid dimension for Betti number computation.
    #[error("invalid homology dimension {dimension}")]
    #[allow(missing_docs)]
    InvalidDimension { dimension: usize },
    /// Insufficient points for the requested operation.
    #[error("insufficient points (required at least {required}, got {actual})")]
    #[allow(missing_docs)]
    InsufficientPoints { required: usize, actual: usize },
    /// Invalid simplex (e.g., duplicate vertices).
    #[error("invalid simplex: {reason}")]
    #[allow(missing_docs)]
    InvalidSimplex { reason: String },
    /// Matrix computation error.
    #[error("matrix computation failed: {reason}")]
    #[allow(missing_docs)]
    MatrixError { reason: String },
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum CopulaError {
    /// Invalid probability (must be in [0, 1]).
    #[error("Invalid copula probability {value} (must be in [0, 1])")]
    InvalidProbability {
        #[allow(missing_docs)]
        value: f64,
    },
    /// Invalid correlation (must be in [-1, 1]).
    #[error("Invalid correlation coefficient {value} (must be in [-1, 1])")]
    InvalidCorrelation {
        #[allow(missing_docs)]
        value: f64,
    },

    /// Matrix is not positive definite.
    #[error("Correlation matrix is not positive definite")]
    NotPositiveDefinite,
    /// Matrix is not symmetric.
    #[error("Correlation matrix is not symmetric")]
    NotSymmetric,

    #[error("Mathematical error: {0}")]
    #[allow(missing_docs)]
    Math(#[from] math_commons::error::MathError),
}

#[derive(Debug, Clone, PartialEq, Error)]
#[allow(missing_docs)]
pub enum Glicko2Error {
    /// Invalid rating value.
    #[error("rating must be finite (got {value})")]
    #[allow(missing_docs)]
    InvalidRating { value: f64 },
    /// Invalid rating deviation (must be positive).
    #[error("rating deviation must be positive and finite (got {value})")]
    #[allow(missing_docs)]
    InvalidRatingDeviation { value: f64 },
    /// Invalid volatility (must be positive).
    #[error("volatility must be positive and finite (got {value})")]
    #[allow(missing_docs)]
    InvalidVolatility { value: f64 },
    /// Invalid system constant tau (must be between 0.3 and 1.2).
    #[error("system constant tau must be in [0.3, 1.2] and finite (got {value})")]
    #[allow(missing_docs)]
    InvalidSystemConstant { value: f64 },
    /// Invalid opponent count (must be at least 1).
    #[error("invalid opponent count ({count})")]
    #[allow(missing_docs)]
    InvalidOpponentCount { count: usize },
    /// Volatility convergence failed after maximum iterations.
    #[error("volatility iteration failed to converge after {iterations} steps")]
    #[allow(missing_docs)]
    VolatilityConvergenceFailed { iterations: usize },
    /// Invalid score (must be between 0 and 1).
    #[error("match score must be in [0.0, 1.0] and finite (got {value})")]
    #[allow(missing_docs)]
    InvalidScore { value: f64 },
    /// Empty match results provided.
    #[error("match results cannot be empty")]
    EmptyMatchResults,
    /// Roster index referenced by a tournament match or pairing is out of bounds.
    #[error("player index {index} is out of bounds for roster of length {roster_len}")]
    #[allow(missing_docs)]
    PlayerIndexOutOfBounds { index: usize, roster_len: usize },
    /// A tournament match or pairing pitted a player against themselves.
    #[error("self-match is not permitted for player index {player_index}")]
    #[allow(missing_docs)]
    SelfMatch { player_index: usize },
    /// Tournament schedule or pairing requires at least two players.
    #[error("at least 2 players are required for tournament pairings (got {count})")]
    #[allow(missing_docs)]
    InsufficientPlayers { count: usize },
    /// Slice dimensions passed to a batch tournament routine did not match.
    #[error("buffer length mismatch (expected {expected}, got {actual})")]
    #[allow(missing_docs)]
    DimensionMismatch { expected: usize, actual: usize },
}

mod submodule_diagnostics;
mod umbrella;

pub use crate::pure_math::{
    algebra::linear_algebra::numerical::LinearAlgebraError,
    algorithmic_information::FractranError,
    analysis::{
        evolution::EvolutionError, optimization::OptimizationError, roots::AnalysisError,
        stochastic::StochasticError, transforms::TransformError,
    },
    elliptic_curves::EllipticCurveError,
    number_theory::ambs::AmbsError,
    tensor::types::TensorError,
};
pub use umbrella::PureMathError;

#[cfg(test)]
mod tests;
