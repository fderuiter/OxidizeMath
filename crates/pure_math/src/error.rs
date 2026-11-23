use thiserror::Error;

#[allow(missing_docs)]
pub type Result<T> = math_commons::error::MathResult<T>;

#[derive(Debug, Error)]
#[allow(missing_docs)]
pub enum NumberTheoryError {
    #[error("Failed to parse integer from string: {0}")]
    #[allow(missing_docs)]
    ParseError(String),

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

/// Errors returned by bookmaker devigging, `+EV`, arbitrage, Dutching, middle, and hedge calculators.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
pub enum MarketPricingError {
    /// Fewer outcomes were supplied than required by the market or calculator.
    #[error("Insufficient outcomes: got {count}, require at least {min_required}")]
    InsufficientOutcomes {
        /// Supplied outcome count.
        count: usize,
        /// Minimum required outcome count.
        min_required: usize,
    },
    /// Supplied outcome slice exceeds fixed stack capacity `MAX_MARKET_OUTCOMES`.
    #[error("Too many outcomes: got {count}, maximum zero-allocation capacity is {max_supported}")]
    TooManyOutcomes {
        /// Supplied outcome count.
        count: usize,
        /// Maximum supported stack outcome count.
        max_supported: usize,
    },
    /// Input and output slices do not have matching lengths.
    #[error("Slice length mismatch: expected {expected}, got {actual}")]
    SliceLengthMismatch {
        /// Expected slice length.
        expected: usize,
        /// Actual slice length.
        actual: usize,
    },
    /// Outcome index exceeds the market slice length.
    #[error("Outcome index {index} is out of bounds for market of {outcome_count} outcomes")]
    OutcomeIndexOutOfBounds {
        /// Requested zero-based outcome index.
        index: usize,
        /// Total outcomes in the market slice.
        outcome_count: usize,
    },
    /// Duplicate outcome index supplied in a Dutching subset selection.
    #[error("Duplicate outcome index {index} in subset selection")]
    DuplicateOutcomeIndex {
        /// Repeated zero-based outcome index.
        index: usize,
    },
    /// Non-finite or non-playable decimal odds (`<= 1.0`).
    #[error("Invalid decimal odds: {value} (must be finite and > 1.0)")]
    InvalidOdds {
        /// Invalid decimal odds value.
        value: f64,
    },
    /// Non-positive or non-finite market booksum (`sum(1 / O_i)`).
    #[error("Market booksum must be finite and positive (got {booksum})")]
    NonPositiveOverround {
        /// Computed raw booksum.
        booksum: f64,
    },
    /// Cross-book reciprocal odds sum is `>= 1.0`, leaving no risk-free arbitrage.
    #[error("No arbitrage opportunity: reciprocal sum {reciprocal_sum} >= 1.0")]
    NoArbitrage {
        /// Combined implied probability sum `sum(1 / O_i)`.
        reciprocal_sum: f64,
    },
    /// Positive target profit cannot be locked when selected reciprocal sum is `>= 1.0`.
    #[error("Target profit is infeasible when reciprocal sum {reciprocal_sum} >= 1.0")]
    InfeasibleTargetProfit {
        /// Combined implied probability sum across selected contenders.
        reciprocal_sum: f64,
    },
    /// Non-finite or non-positive stake, budget, or target profit parameter.
    #[error("Stake or budget must be finite and positive (got {value})")]
    InvalidStakeOrBudget {
        /// Invalid stake or budget scalar.
        value: f64,
    },
    /// Requested downside profit floor exceeds the equal-profit hedge ceiling.
    #[error(
        "Target profit floor {floor} exceeds maximum attainable guaranteed profit {max_guaranteed_profit}"
    )]
    UnattainableFloor {
        /// Requested minimum profit floor.
        floor: f64,
        /// Maximum equalized profit attainable at current hedge odds.
        max_guaranteed_profit: f64,
    },
    /// Probability value is non-finite or outside `[0.0, 1.0]`.
    #[error("Invalid probability: {value} (must be finite and in [0.0, 1.0])")]
    InvalidProbability {
        /// Invalid probability scalar.
        value: f64,
    },
    /// Numerical root-finding failure during `Power` or `Shin` calibration.
    #[error("Root-finding solver failed during devigging: {0}")]
    Solver(#[from] AnalysisError),
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum MarkovError {
    /// Invalid probability value (must be in [0, 1] and finite).
    InvalidProbability {
        /// The invalid probability value.
        value: f64,
    },

    /// Matrix is not stochastic (rows don't sum to 1).
    NotStochastic {
        /// Description of the issue.
        reason: String,
    },

    /// Matrix is not a valid generator (rows don't sum to 0 or invalid rates).
    InvalidGenerator {
        /// Description of the issue.
        reason: String,
    },

    /// Invalid state specification.
    InvalidState {
        /// Description of the issue.
        reason: String,
    },

    /// Time index out of bounds.
    TimeIndexOutOfBounds {
        /// The requested time.
        time: f64,
        /// Valid time range.
        valid_range: (f64, f64),
    },

    /// Observation sequence incompatible with model.
    InvalidObservation {
        /// Description of the issue.
        reason: String,
    },

    /// Matrix not invertible or singular.
    SingularMatrix {
        /// Context about which matrix.
        context: String,
    },

    #[allow(missing_docs)]
    Math(math_commons::error::MathError),
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

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum ZipError {
    /// Invalid probability parameter (must be in [0, 1]).
    #[allow(missing_docs)]
    InvalidProbability { value: f64, parameter: String },
    /// Invalid rate parameter (must be positive).
    #[allow(missing_docs)]
    InvalidRate { value: f64 },
    /// Invalid count value (must be non-negative integer).
    #[allow(missing_docs)]
    InvalidCount { value: f64 },
    /// Insufficient data for regression.
    #[allow(missing_docs)]
    InsufficientData { required: usize, actual: usize },
    /// Regression convergence failed.
    #[allow(missing_docs)]
    ConvergenceFailed { iterations: usize, reason: String },
    /// Invalid predictor matrix dimensions.
    #[allow(missing_docs)]
    InvalidDimensions { expected: String, actual: String },
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum TdaError {
    /// Empty point cloud provided.
    EmptyPointCloud,
    /// Invalid radius (must be non-negative).
    #[allow(missing_docs)]
    InvalidRadius { value: f64 },
    /// Invalid dimension for Betti number computation.
    #[allow(missing_docs)]
    InvalidDimension { dimension: usize },
    /// Insufficient points for the requested operation.
    #[allow(missing_docs)]
    InsufficientPoints { required: usize, actual: usize },
    /// Invalid simplex (e.g., duplicate vertices).
    #[allow(missing_docs)]
    InvalidSimplex { reason: String },
    /// Matrix computation error.
    #[allow(missing_docs)]
    MatrixError { reason: String },
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum CopulaError {
    /// Invalid probability (must be in [0, 1]).
    InvalidProbability {
        #[allow(missing_docs)]
        value: f64,
    },
    /// Invalid correlation (must be in [-1, 1]).
    InvalidCorrelation {
        #[allow(missing_docs)]
        value: f64,
    },

    /// Matrix is not positive definite.
    NotPositiveDefinite,
    /// Matrix is not symmetric.
    NotSymmetric,

    #[allow(missing_docs)]
    Math(math_commons::error::MathError),
}

#[derive(Debug, Clone, PartialEq)]
#[allow(missing_docs)]
pub enum Glicko2Error {
    /// Invalid rating value.
    #[allow(missing_docs)]
    InvalidRating { value: f64 },
    /// Invalid rating deviation (must be positive).
    #[allow(missing_docs)]
    InvalidRatingDeviation { value: f64 },
    /// Invalid volatility (must be positive).
    #[allow(missing_docs)]
    InvalidVolatility { value: f64 },
    /// Invalid system constant tau (must be between 0.3 and 1.2).
    #[allow(missing_docs)]
    InvalidSystemConstant { value: f64 },
    /// Invalid opponent count (must be at least 1).
    #[allow(missing_docs)]
    InvalidOpponentCount { count: usize },
    /// Volatility convergence failed after maximum iterations.
    #[allow(missing_docs)]
    VolatilityConvergenceFailed { iterations: usize },
    /// Invalid score (must be between 0 and 1).
    #[allow(missing_docs)]
    InvalidScore { value: f64 },
    /// Empty match results provided.
    EmptyMatchResults,
    /// Roster index referenced by a tournament match or pairing is out of bounds.
    #[allow(missing_docs)]
    PlayerIndexOutOfBounds { index: usize, roster_len: usize },
    /// A tournament match or pairing pitted a player against themselves.
    #[allow(missing_docs)]
    SelfMatch { player_index: usize },
    /// Tournament schedule or pairing requires at least two players.
    #[allow(missing_docs)]
    InsufficientPlayers { count: usize },
    /// Slice dimensions passed to a batch tournament routine did not match.
    #[allow(missing_docs)]
    DimensionMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for MarkovError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for MarkovError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for ZipError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for ZipError {}

impl std::fmt::Display for TdaError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyPointCloud => write!(f, "point cloud cannot be empty"),
            Self::InvalidRadius { value } => {
                write!(
                    f,
                    "filtration radius must be non-negative and finite (got {value})"
                )
            }
            Self::InvalidDimension { dimension } => {
                write!(f, "invalid homology dimension {dimension}")
            }
            Self::InsufficientPoints { required, actual } => {
                write!(
                    f,
                    "insufficient points (required at least {required}, got {actual})"
                )
            }
            Self::InvalidSimplex { reason } => write!(f, "invalid simplex: {reason}"),
            Self::MatrixError { reason } => write!(f, "matrix computation failed: {reason}"),
        }
    }
}

impl std::error::Error for TdaError {}

impl std::fmt::Display for CopulaError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for CopulaError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for Glicko2Error {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRating { value } => write!(f, "rating must be finite (got {value})"),
            Self::InvalidRatingDeviation { value } => {
                write!(
                    f,
                    "rating deviation must be positive and finite (got {value})"
                )
            }
            Self::InvalidVolatility { value } => {
                write!(f, "volatility must be positive and finite (got {value})")
            }
            Self::InvalidSystemConstant { value } => {
                write!(
                    f,
                    "system constant tau must be in [0.3, 1.2] and finite (got {value})"
                )
            }
            Self::InvalidOpponentCount { count } => write!(f, "invalid opponent count ({count})"),
            Self::VolatilityConvergenceFailed { iterations } => {
                write!(
                    f,
                    "volatility iteration failed to converge after {iterations} steps"
                )
            }
            Self::InvalidScore { value } => {
                write!(
                    f,
                    "match score must be in [0.0, 1.0] and finite (got {value})"
                )
            }
            Self::EmptyMatchResults => write!(f, "match results cannot be empty"),
            Self::PlayerIndexOutOfBounds { index, roster_len } => {
                write!(
                    f,
                    "player index {index} is out of bounds for roster of length {roster_len}"
                )
            }
            Self::SelfMatch { player_index } => {
                write!(
                    f,
                    "self-match is not permitted for player index {player_index}"
                )
            }
            Self::InsufficientPlayers { count } => {
                write!(
                    f,
                    "at least 2 players are required for tournament pairings (got {count})"
                )
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "buffer length mismatch (expected {expected}, got {actual})"
                )
            }
        }
    }
}

impl std::error::Error for Glicko2Error {}

impl From<math_commons::error::MathError> for MarkovError {
    fn from(err: math_commons::error::MathError) -> Self {
        MarkovError::Math(err)
    }
}

impl From<math_commons::error::MathError> for CopulaError {
    fn from(err: math_commons::error::MathError) -> Self {
        CopulaError::Math(err)
    }
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
