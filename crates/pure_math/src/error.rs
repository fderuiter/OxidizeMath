use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

#[allow(missing_docs)]
pub type Result<T> = math_commons::error::MathResult<T>;

#[derive(Debug, Error, Clone, PartialEq, Eq)]
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

impl Diagnostic for NumberTheoryError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "NumberTheoryError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for KellyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "KellyError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for MarkovError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "MarkovError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for OuError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "OuError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for ZipError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "ZipError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for TdaError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "TdaError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for CopulaError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "CopulaError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl Diagnostic for Glicko2Error {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }
    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "Glicko2Error".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

impl std::fmt::Display for MarkovError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for MarkovError {}

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

impl std::error::Error for CopulaError {}

impl std::fmt::Display for Glicko2Error {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRating { value } => {
                write!(f, "rating must be finite (got {value})")
            }
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
            Self::InvalidOpponentCount { count } => {
                write!(f, "invalid opponent count ({count})")
            }
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
