use thiserror::Error;

/// Enumeration `MarkovError` classifying MarkovError variants.
#[derive(Debug, Clone, PartialEq, Error)]
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
    /// Variant `Math` (Math).
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
}

/// Enumeration `OuError` classifying OuError variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum OuError {
    /// Invalid mean reversion rate (must be positive).
    #[error("Invalid mean reversion rate: {value} (must be positive)")]
    InvalidMeanReversionRate {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid volatility (must be non-negative).
    #[error("Invalid volatility: {value} (must be non-negative)")]
    InvalidVolatility {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid time step (must be positive).
    #[error("Invalid time step: {value} (must be positive)")]
    InvalidTimeStep {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid simulation parameters.
    #[error("Invalid simulation parameters: {reason}")]
    InvalidSimulationParams {
        /// Value for `reason` (reason).
        reason: String,
    },
    /// Insufficient data for parameter estimation.
    #[error("Insufficient data: required at least {required}, got {actual}")]
    InsufficientData {
        /// Value for `required` (required).
        required: usize,
        /// Value for `actual` (actual).
        actual: usize,
    },
}

/// Enumeration `ZipError` classifying ZipError variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ZipError {
    /// Invalid probability parameter (must be in [0, 1]).
    #[error("Invalid probability {value} for parameter '{parameter}' (must be in [0, 1])")]
    InvalidProbability {
        /// Value for `value` (value).
        value: f64,
        /// Value for `parameter` (parameter).
        parameter: String,
    },
    /// Invalid rate parameter (must be positive).
    #[error("Invalid Poisson rate {value} (must be strictly positive)")]
    InvalidRate {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid count value (must be non-negative integer).
    #[error("Invalid count {value} (must be a finite non-negative integer)")]
    InvalidCount {
        /// Value for `value` (value).
        value: f64,
    },
    /// Insufficient data for regression.
    #[error(
        "Insufficient observations for ZIP regression: required at least {required}, got {actual}"
    )]
    InsufficientData {
        /// Value for `required` (required).
        required: usize,
        /// Value for `actual` (actual).
        actual: usize,
    },
    /// Regression convergence failed.
    #[error("ZIP regression failed to converge after {iterations} iterations: {reason}")]
    ConvergenceFailed {
        /// Value for `iterations` (iterations).
        iterations: usize,
        /// Value for `reason` (reason).
        reason: String,
    },
    /// Invalid predictor matrix dimensions.
    #[error("Invalid predictor matrix dimensions: expected {expected}, got {actual}")]
    InvalidDimensions {
        /// Value for `expected` (expected).
        expected: String,
        /// Value for `actual` (actual).
        actual: String,
    },
}

/// Enumeration `TdaError` classifying TdaError variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TdaError {
    /// Empty point cloud provided.
    #[error("point cloud cannot be empty")]
    EmptyPointCloud,
    /// Invalid radius (must be non-negative).
    #[error("filtration radius must be non-negative and finite (got {value})")]
    InvalidRadius {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid dimension for Betti number computation.
    #[error("invalid homology dimension {dimension}")]
    InvalidDimension {
        /// Value for `dimension` (dimension).
        dimension: usize,
    },
    /// Insufficient points for the requested operation.
    #[error("insufficient points (required at least {required}, got {actual})")]
    InsufficientPoints {
        /// Value for `required` (required).
        required: usize,
        /// Value for `actual` (actual).
        actual: usize,
    },
    /// Invalid simplex (e.g., duplicate vertices).
    #[error("invalid simplex: {reason}")]
    InvalidSimplex {
        /// Value for `reason` (reason).
        reason: String,
    },
    /// Matrix computation error.
    #[error("matrix computation failed: {reason}")]
    MatrixError {
        /// Value for `reason` (reason).
        reason: String,
    },
}

/// Enumeration `CopulaError` classifying CopulaError variants.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CopulaError {
    /// Invalid probability (must be in [0, 1]).
    #[error("Invalid copula probability {value} (must be in [0, 1])")]
    InvalidProbability {
        /// Value for `value` (value).
        value: f64,
    },
    /// Invalid correlation (must be in [-1, 1]).
    #[error("Invalid correlation coefficient {value} (must be in [-1, 1])")]
    InvalidCorrelation {
        /// Value for `value` (value).
        value: f64,
    },
    /// Matrix is not positive definite.
    #[error("Correlation matrix is not positive definite")]
    NotPositiveDefinite,
    /// Matrix is not symmetric.
    #[error("Correlation matrix is not symmetric")]
    NotSymmetric,
    /// Variant `Math` (Math).
    #[error("Mathematical error: {0}")]
    Math(#[from] math_commons::error::MathError),
}
