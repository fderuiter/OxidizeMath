use thiserror::Error;

#[allow(missing_docs)]
pub type Result<T> = math_commons::error::MathResult<T>;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[allow(missing_docs)]
pub enum NumberTheoryError {
    #[error("Failed to parse integer from string: {0}")]
    #[allow(missing_docs)]
    ParseError(String),

    /// Radix supplied to `Integer::from_str_radix` was outside `2..=36`.
    #[error("Invalid integer radix {radix} (must be between 2 and 36)")]
    InvalidRadix {
        /// Invalid numeric base.
        radix: u32,
    },

    /// String literal could not be parsed as an integer in the given radix.
    #[error("Failed to parse integer literal in base {radix}")]
    InvalidIntegerLiteral {
        /// Numeric base used during parsing.
        radix: u32,
    },

    /// Modular arithmetic operation received a zero modulus.
    #[error("Modulus must be non-zero")]
    ZeroModulus,

    /// Integer element is not coprime to the modulus and has no multiplicative inverse.
    #[error("Element has no multiplicative inverse for the given modulus")]
    NoModularInverse,

    /// Attempted to invert the additive identity (`0`) in a finite field.
    #[error("Cannot invert zero element in finite field")]
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

    /// Divisor `QSeries` had a zero constant term `b_0 == 0`.
    #[error("Division by a QSeries with zero constant term")]
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
    /// Invalid net profit multiplier (must be finite and strictly positive).
    #[error("Invalid net odds: {value} (must be finite and > 0.0)")]
    InvalidNetOdds {
        /// Rejected net profit multiplier `b`.
        value: f64,
    },
    /// Invalid American odds (must be finite with magnitude at least 100).
    #[error("Invalid American odds: {value} (must be finite with |odds| >= 100.0)")]
    InvalidAmericanOdds {
        /// Rejected American odds input.
        value: f64,
    },
    /// Invalid fractional odds (numerator and denominator must both be finite and strictly positive).
    #[error(
        "Invalid fractional odds: {numerator}/{denominator} (numerator and denominator must be finite and > 0.0)"
    )]
    InvalidFractionalOdds {
        /// Rejected fractional profit numerator.
        numerator: f64,
        /// Rejected fractional stake denominator.
        denominator: f64,
    },
    /// Invalid implied probability (must be finite and strictly inside `(0.0, 1.0)`).
    #[error(
        "Invalid implied probability: {value} (must be finite and strictly between 0.0 and 1.0)"
    )]
    InvalidImpliedProbability {
        /// Rejected implied probability input.
        value: f64,
    },
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
    statistics::{poisson::PoissonError, queueing::QueueingError},
    tensor::types::TensorError,
};
pub use umbrella::PureMathError;

#[cfg(test)]
mod tests;
