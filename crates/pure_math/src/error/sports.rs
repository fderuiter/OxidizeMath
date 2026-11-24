use thiserror::Error;

/// Errors returned by sports Elo rating updates, batch schedules, and line converters.
#[derive(Debug, Clone, PartialEq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EloError {
    /// Elo rating is non-finite (`NaN` or infinite).
    #[error("Elo rating must be finite (got {value})")]
    InvalidRating { value: f64 },
    /// Logistic scale parameter is non-positive or non-finite.
    #[error("Elo logistic scale must be positive and finite (got {value})")]
    InvalidScale { value: f64 },
    /// Elo update factor `K` is non-positive or non-finite.
    #[error("Elo K-factor must be positive and finite (got {value})")]
    InvalidKFactor { value: f64 },
    /// Home-advantage rating offset is non-finite.
    #[error("Elo home-advantage offset must be finite (got {value})")]
    InvalidHomeAdvantage { value: f64 },
    /// Match outcome score lies outside `[0.0, 1.0]` or is non-finite.
    #[error("match score must be in [0.0, 1.0] and finite (got {value})")]
    InvalidScore { value: f64 },
    /// Seasonal mean-regression weight `theta` lies outside `[0.0, 1.0]` or is non-finite.
    #[error("regression factor theta must be in [0.0, 1.0] and finite (got {value})")]
    InvalidRegressionFactor { value: f64 },
    /// Elo-points-per-goal spread conversion factor is non-positive or non-finite.
    #[error("Elo points-per-goal scale must be positive and finite (got {value})")]
    InvalidPointsPerElo { value: f64 },
    /// Win probability is `0.0` or `1.0` and cannot produce finite two-way moneyline odds.
    #[error(
        "win probability must lie strictly in (0.0, 1.0) to quote moneyline odds (got {value})"
    )]
    DegenerateProbability { value: f64 },
    /// Team roster index is out of bounds for the supplied roster slice.
    #[error("team index {index} is out of bounds for roster of length {roster_len}")]
    PlayerIndexOutOfBounds { index: usize, roster_len: usize },
    /// Match schedule attempted to pair a team against itself.
    #[error("self-match is not permitted for team index {team_index}")]
    SelfMatch { team_index: usize },
    /// Output or scratch slice length did not match the roster length.
    #[error("buffer length mismatch (expected {expected}, got {actual})")]
    DimensionMismatch { expected: usize, actual: usize },
}

/// Errors returned by 2D score grids, Poisson/Dixon-Coles models, market pricers, and MLE calibration.
#[derive(Debug, Clone, PartialEq, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MatchModelError {
    /// Requested `max_goals` cutoff exceeds the stack grid ceiling.
    #[error("max_goals {max_goals} exceeds supported grid limit {limit}")]
    InvalidMaxGoals { max_goals: usize, limit: usize },
    /// Expected goal rate `lambda` or `mu` is non-positive or non-finite.
    #[error("expected goal rate must be strictly positive and finite (got {value})")]
    InvalidGoalRate { value: f64 },
    /// Bivariate Poisson component rate is negative or non-finite.
    #[error("bivariate Poisson component rate must be non-negative and finite (got {value})")]
    InvalidSharedRate { value: f64 },
    /// Shared goal covariance lies outside `[0.0, min(lambda, mu)]` or is non-finite.
    #[error("shared covariance {covariance} must lie in [0.0, {max_allowed}]")]
    InvalidCovariance { covariance: f64, max_allowed: f64 },
    /// Dixon-Coles low-score correlation `rho` violates non-negativity bounds on `tau`.
    #[error("Dixon-Coles rho {rho} is outside feasible bounds [{min}, {max}]")]
    InvalidCorrelation { rho: f64, min: f64, max: f64 },
    /// Score grid total probability mass is non-positive or non-finite.
    #[error("score grid total probability mass must be positive and finite (got {mass})")]
    InvalidProbabilityMass { mass: f64 },
    /// Score grid cell contains a negative or non-finite probability.
    #[error("negative cell probability {value} at score ({home}, {away})")]
    NegativeCellProbability {
        home: usize,
        away: usize,
        value: f64,
    },
    /// Queried `(home, away)` score exceeds the grid's `max_goals` bound.
    #[error("score ({home}, {away}) exceeds grid max_goals {max_goals}")]
    ScoreOutOfBounds {
        home: usize,
        away: usize,
        max_goals: usize,
    },
    /// Handicap or total line is non-finite, negative (for totals), or not a `0.25`-goal multiple.
    #[error("market line {line} must be finite, valid for market, and aligned to 0.25 increments")]
    InvalidLine { line: f64 },
    /// Market has zero or unit active probability mass after excluding pushes.
    #[error("market quote is degenerate with active probability {probability}")]
    DegenerateMarket { probability: f64 },
    /// Exponential time-decay rate `xi` is negative or non-finite.
    #[error("time-decay parameter xi must be non-negative and finite (got {value})")]
    InvalidDecayRate { value: f64 },
    /// Historical match age is negative or non-finite.
    #[error("historical match age must be non-negative and finite (got {value})")]
    InvalidMatchAge { value: f64 },
    /// Calibration dataset contains fewer matches than required.
    #[error("insufficient historical matches (required at least {required}, got {actual})")]
    InsufficientMatches { required: usize, actual: usize },
    /// Calibration league contains fewer than two teams.
    #[error("at least 2 teams are required for calibration (got {count})")]
    InsufficientTeams { count: usize },
    /// Historical match references a team index outside `0..num_teams`.
    #[error("team index {index} is out of bounds for {num_teams} teams")]
    TeamIndexOutOfBounds { index: usize, num_teams: usize },
    /// Historical match pairs a team against itself.
    #[error("self-match is not permitted for team index {team_index}")]
    SelfMatch { team_index: usize },
    /// Maximum likelihood optimization failed to produce finite parameters.
    #[error("Dixon-Coles MLE failed to produce finite parameters after {iterations} iterations")]
    CalibrationFailed { iterations: usize },
}
