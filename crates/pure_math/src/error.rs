use thiserror::Error;

/// Type alias `Result` used across this module.
pub type Result<T> = math_commons::error::MathResult<T>;

/// Enumeration `NumberTheoryError` classifying NumberTheoryError variants.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NumberTheoryError {
    /// Variant `ParseError` (ParseError).
    #[error("Failed to parse integer from string: {0}")]
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
    /// Variant `ModuloError` (ModuloError).
    #[error("Modulo operation failed")]
    ModuloError,
    /// Variant `ConversionError` (ConversionError).
    #[error("Conversion to usize failed")]
    ConversionError,
    /// Variant `DivisionByZeroQSeries` (DivisionByZeroQSeries).
    #[error("Division by zero QSeries")]
    DivisionByZeroQSeries,
    /// Divisor `QSeries` had a zero constant term `b_0 == 0`.
    #[error("Division by a QSeries with zero constant term")]
    DivisionByZeroConstantTerm,
}

/// Enumeration `KellyError` classifying KellyError variants.
#[derive(Debug, Clone, PartialEq)]
pub enum KellyError {
    /// Invalid probability (must be between 0 and 1).
    InvalidProbability {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid odds (must be greater than 1.0 for decimal odds).
    InvalidOdds {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid net profit multiplier (must be finite and strictly positive).
    InvalidNetOdds {
        /// Rejected net profit multiplier `b`.
        value: f64,
    },
    /// Invalid American odds (must be finite with magnitude at least 100).
    InvalidAmericanOdds {
        /// Rejected American odds input.
        value: f64,
    },
    /// Invalid fractional odds (numerator and denominator must both be finite and strictly positive).
    InvalidFractionalOdds {
        /// Rejected fractional profit numerator.
        numerator: f64,
        /// Rejected fractional stake denominator.
        denominator: f64,
    },
    /// Invalid implied probability (must be finite and strictly inside `(0.0, 1.0)`).
    InvalidImpliedProbability {
        /// Rejected implied probability input.
        value: f64,
    },
    /// Invalid fraction (must be between 0 and 1).
    InvalidFraction {
        /// Value for `value` (value).
        value: f64,
    },
    /// No edge (negative expected value - should not bet).
    NoEdge {
        /// Value for `probability` (probability).
        probability: f64,
        /// Value for `odds` (odds).
        odds: f64,
    },
    /// Invalid bankroll amount (must be positive).
    InvalidBankroll {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid stake or portfolio weight (must be finite and strictly positive).
    InvalidStake {
        /// Rejected stake or portfolio weight value.
        value: f64,
    },
    /// Invalid drawdown floor, target barrier, or confidence level.
    InvalidBarrier {
        /// Configured wealth floor or barrier threshold.
        floor: f64,
        /// Initial bankroll wealth.
        initial_wealth: f64,
    },
    /// Empty or insufficient bet ledger for statistical estimation.
    InsufficientBets {
        /// Minimum required number of bets.
        required: usize,
        /// Actual number of bets provided.
        actual: usize,
    },
    /// Mismatched slice or matrix dimensions across probabilities, odds, stakes, or covariances.
    DimensionMismatch {
        /// Expected slice or matrix dimension.
        expected: usize,
        /// Actual slice or matrix dimension provided.
        actual: usize,
    },
    /// Covariance matrix is not symmetric positive-definite.
    NotPositiveDefinite,
    /// Insufficient observations or assets for portfolio estimation.
    InsufficientData {
        /// Minimum required observation or asset count.
        required: usize,
        /// Actual observation or asset count provided.
        actual: usize,
    },
    /// Confidence level for VaR/CVaR must lie strictly inside `(0, 1)`.
    InvalidConfidenceLevel {
        /// Rejected confidence level input.
        value: f64,
    },
    /// Non-finite or out-of-range portfolio / horizon parameter.
    InvalidParameter {
        /// Static diagnostic describing the invalid parameter.
        reason: &'static str,
    },
    /// Root-finding or adjustment-coefficient solver failed to bracket or converge.
    ConvergenceFailed {
        /// Static diagnostic describing why convergence failed.
        reason: &'static str,
    },
    /// Outcome probabilities for a complete multi-state wager do not sum to 1.0.
    InvalidProbabilityDistribution {
        /// Observed sum of state probabilities.
        sum: f64,
    },
    /// Mutually exclusive market outcome probabilities exceed 1.0.
    InvalidMarketProbabilitySum {
        /// Observed sum of market outcome probabilities.
        sum: f64,
    },
    /// Empty market or simultaneous slate provided.
    EmptyPortfolio,
    /// Market or simultaneous slate exceeds preallocated scratch capacity.
    PortfolioTooLarge {
        /// Requested portfolio wager or outcome count.
        count: usize,
        /// Maximum supported preallocated capacity.
        max: usize,
    },
    /// Invalid Busseti-Ryu-Boyd drawdown floor `alpha` or risk tolerance `beta`.
    InvalidDrawdownBounds {
        /// Drawdown fraction floor `alpha`.
        alpha: f64,
        /// Maximum tolerated drawdown probability `beta`.
        beta: f64,
    },
    /// Numerical root or convex Kelly solver failed to reach target residual within iteration cap.
    SolverConvergenceFailed {
        /// Iterations executed before stopping.
        iterations: usize,
        /// Final KKT or root residual magnitude.
        residual: f64,
    },
    /// Underlying `#![no_std]` linear algebra engine failure preserving causal `source()`.
    Linalg(engine_linalg::LinalgError),
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

/// Enumeration `Glicko2Error` classifying Glicko2Error variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Glicko2Error {
    /// Invalid rating value.
    #[error("rating must be finite (got {value})")]
    InvalidRating {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid rating deviation (must be positive).
    #[error("rating deviation must be positive and finite (got {value})")]
    InvalidRatingDeviation {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid volatility (must be positive).
    #[error("volatility must be positive and finite (got {value})")]
    InvalidVolatility {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid system constant tau (must be between 0.3 and 1.2).
    #[error("system constant tau must be in [0.3, 1.2] and finite (got {value})")]
    InvalidSystemConstant {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid opponent count (must be at least 1).
    #[error("invalid opponent count ({count})")]
    InvalidOpponentCount {
        /// Value for `count` (count).
        count: usize,
    },
    /// Volatility convergence failed after maximum iterations.
    #[error("volatility iteration failed to converge after {iterations} steps")]
    VolatilityConvergenceFailed {
        /// Value for `iterations` (iterations).
        iterations: usize,
    },
    /// Invalid score (must be between 0 and 1).
    #[error("match score must be in [0.0, 1.0] and finite (got {value})")]
    InvalidScore {
        /// Value for `value` (value).
        value: f64,
    },
    /// Empty match results provided.
    #[error("match results cannot be empty")]
    EmptyMatchResults,
    /// Roster index referenced by a tournament match or pairing is out of bounds.
    #[error("player index {index} is out of bounds for roster of length {roster_len}")]
    PlayerIndexOutOfBounds {
        /// Value for `index` (index).
        index: usize,
        /// Value for `roster_len` (roster len).
        roster_len: usize,
    },
    /// A tournament match or pairing pitted a player against themselves.
    #[error("self-match is not permitted for player index {player_index}")]
    SelfMatch {
        /// Value for `player_index` (player index).
        player_index: usize,
    },
    /// Tournament schedule or pairing requires at least two players.
    #[error("at least 2 players are required for tournament pairings (got {count})")]
    InsufficientPlayers {
        /// Value for `count` (count).
        count: usize,
    },
    /// Slice dimensions passed to a batch tournament routine did not match.
    #[error("buffer length mismatch (expected {expected}, got {actual})")]
    DimensionMismatch {
        /// Value for `expected` (expected).
        expected: usize,
        /// Value for `actual` (actual).
        actual: usize,
    },
}

mod helpers;
mod kelly;
mod sports;
mod stochastic;
mod submodule_diagnostics;
mod umbrella;
pub(crate) use helpers::{IntoMathResult, ensure_math};

pub use crate::pure_math::{
    algebra::{BraidError, linear_algebra::numerical::LinearAlgebraError},
    algorithmic_information::FractranError,
    analysis::{
        evolution::EvolutionError, optimization::OptimizationError, roots::AnalysisError,
        stochastic::StochasticError, transforms::TransformError,
    },
    differential_geometry::hyperbolic::Su11HyperbolicError,
    elliptic_curves::EllipticCurveError,
    graph_theory::GraphTheoryError,
    hyperbolic::HyperbolicError,
    knot_theory::KnotError,
    number_theory::ambs::AmbsError,
    statistics::{
        black_scholes::OptionPricingError, monte_carlo::MonteCarloError, poisson::PoissonError,
        queueing::QueueingError, risk::RiskError,
    },
    tensor::types::TensorError,
};
pub use sports::{EloError, MatchModelError};
pub use stochastic::{CopulaError, MarkovError, OuError, TdaError, ZipError};
pub use umbrella::PureMathError;

#[cfg(test)]
mod tests;
