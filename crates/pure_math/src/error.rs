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

/// Errors returned by sports Elo rating updates, batch schedules, and line converters.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum EloError {
    /// Elo rating is non-finite (`NaN` or infinite).
    #[error("Elo rating must be finite (got {value})")]
    InvalidRating {
        /// Offending rating value.
        value: f64,
    },
    /// Logistic scale parameter is non-positive or non-finite.
    #[error("Elo logistic scale must be positive and finite (got {value})")]
    InvalidScale {
        /// Offending scale parameter.
        value: f64,
    },
    /// Elo update factor `K` is non-positive or non-finite.
    #[error("Elo K-factor must be positive and finite (got {value})")]
    InvalidKFactor {
        /// Offending K-factor value.
        value: f64,
    },
    /// Home-advantage rating offset is non-finite.
    #[error("Elo home-advantage offset must be finite (got {value})")]
    InvalidHomeAdvantage {
        /// Offending home-advantage offset.
        value: f64,
    },
    /// Match outcome score lies outside `[0.0, 1.0]` or is non-finite.
    #[error("match score must be in [0.0, 1.0] and finite (got {value})")]
    InvalidScore {
        /// Offending match score.
        value: f64,
    },
    /// Seasonal mean-regression weight `theta` lies outside `[0.0, 1.0]` or is non-finite.
    #[error("regression factor theta must be in [0.0, 1.0] and finite (got {value})")]
    InvalidRegressionFactor {
        /// Offending regression factor.
        value: f64,
    },
    /// Elo-points-per-goal spread conversion factor is non-positive or non-finite.
    #[error("Elo points-per-goal scale must be positive and finite (got {value})")]
    InvalidPointsPerElo {
        /// Offending points-per-goal factor.
        value: f64,
    },
    /// Win probability is `0.0` or `1.0` and cannot produce finite two-way moneyline odds.
    #[error(
        "win probability must lie strictly in (0.0, 1.0) to quote moneyline odds (got {value})"
    )]
    DegenerateProbability {
        /// Offending win probability.
        value: f64,
    },
    /// Team roster index is out of bounds for the supplied roster slice.
    #[error("team index {index} is out of bounds for roster of length {roster_len}")]
    PlayerIndexOutOfBounds {
        /// Requested team index.
        index: usize,
        /// Length of the supplied roster slice.
        roster_len: usize,
    },
    /// Match schedule attempted to pair a team against itself.
    #[error("self-match is not permitted for team index {team_index}")]
    SelfMatch {
        /// Index of the team paired with itself.
        team_index: usize,
    },
    /// Output or scratch slice length did not match the roster length.
    #[error("buffer length mismatch (expected {expected}, got {actual})")]
    DimensionMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length provided.
        actual: usize,
    },
}

/// Errors returned by 2D score grids, Poisson/Dixon-Coles models, market pricers, and MLE calibration.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum MatchModelError {
    /// Requested `max_goals` cutoff exceeds the stack grid ceiling.
    #[error("max_goals {max_goals} exceeds supported grid limit {limit}")]
    InvalidMaxGoals {
        /// Requested maximum goals per team.
        max_goals: usize,
        /// Maximum supported goal cutoff.
        limit: usize,
    },
    /// Expected goal rate `lambda` or `mu` is non-positive or non-finite.
    #[error("expected goal rate must be strictly positive and finite (got {value})")]
    InvalidGoalRate {
        /// Offending goal rate.
        value: f64,
    },
    /// Bivariate Poisson component rate is negative or non-finite.
    #[error("bivariate Poisson component rate must be non-negative and finite (got {value})")]
    InvalidSharedRate {
        /// Offending component rate.
        value: f64,
    },
    /// Shared goal covariance lies outside `[0.0, min(lambda, mu)]` or is non-finite.
    #[error("shared covariance {covariance} must lie in [0.0, {max_allowed}]")]
    InvalidCovariance {
        /// Requested shared covariance.
        covariance: f64,
        /// Upper bound `min(lambda_home, mu_away)`.
        max_allowed: f64,
    },
    /// Dixon-Coles low-score correlation `rho` violates non-negativity bounds on `tau`.
    #[error("Dixon-Coles rho {rho} is outside feasible bounds [{min}, {max}]")]
    InvalidCorrelation {
        /// Offending `rho` value.
        rho: f64,
        /// Lower feasible bound `max(-1.0 / (lambda * mu), -1.0)`.
        min: f64,
        /// Upper feasible bound `min(1.0 / lambda, 1.0 / mu)`.
        max: f64,
    },
    /// Score grid total probability mass is non-positive or non-finite.
    #[error("score grid total probability mass must be positive and finite (got {mass})")]
    InvalidProbabilityMass {
        /// Computed grid probability mass.
        mass: f64,
    },
    /// Score grid cell contains a negative or non-finite probability.
    #[error("negative cell probability {value} at score ({home}, {away})")]
    NegativeCellProbability {
        /// Home goal count for the invalid cell.
        home: usize,
        /// Away goal count for the invalid cell.
        away: usize,
        /// Offending cell probability.
        value: f64,
    },
    /// Queried `(home, away)` score exceeds the grid's `max_goals` bound.
    #[error("score ({home}, {away}) exceeds grid max_goals {max_goals}")]
    ScoreOutOfBounds {
        /// Queried home goal count.
        home: usize,
        /// Queried away goal count.
        away: usize,
        /// Grid `max_goals` cutoff.
        max_goals: usize,
    },
    /// Handicap or total line is non-finite, negative (for totals), or not a `0.25`-goal multiple.
    #[error("market line {line} must be finite, valid for market, and aligned to 0.25 increments")]
    InvalidLine {
        /// Offending handicap or total line.
        line: f64,
    },
    /// Market has zero or unit active probability mass after excluding pushes.
    #[error("market quote is degenerate with active probability {probability}")]
    DegenerateMarket {
        /// Active win probability causing degenerate odds.
        probability: f64,
    },
    /// Exponential time-decay rate `xi` is negative or non-finite.
    #[error("time-decay parameter xi must be non-negative and finite (got {value})")]
    InvalidDecayRate {
        /// Offending decay rate `xi`.
        value: f64,
    },
    /// Historical match age is negative or non-finite.
    #[error("historical match age must be non-negative and finite (got {value})")]
    InvalidMatchAge {
        /// Offending match age.
        value: f64,
    },
    /// Calibration dataset contains fewer matches than required.
    #[error("insufficient historical matches (required at least {required}, got {actual})")]
    InsufficientMatches {
        /// Minimum required match count.
        required: usize,
        /// Provided match count.
        actual: usize,
    },
    /// Calibration league contains fewer than two teams.
    #[error("at least 2 teams are required for calibration (got {count})")]
    InsufficientTeams {
        /// Provided team count.
        count: usize,
    },
    /// Historical match references a team index outside `0..num_teams`.
    #[error("team index {index} is out of bounds for {num_teams} teams")]
    TeamIndexOutOfBounds {
        /// Offending team index.
        index: usize,
        /// Number of teams in the league.
        num_teams: usize,
    },
    /// Historical match pairs a team against itself.
    #[error("self-match is not permitted for team index {team_index}")]
    SelfMatch {
        /// Team index paired with itself.
        team_index: usize,
    },
    /// Maximum likelihood optimization failed to produce finite parameters.
    #[error("Dixon-Coles MLE failed to produce finite parameters after {iterations} iterations")]
    CalibrationFailed {
        /// Iteration count executed before failure.
        iterations: usize,
    },
}

mod display_impls;
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
