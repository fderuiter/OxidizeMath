/// Error variants for univariate probability distributions and parameter estimators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistributionError {
    /// Distribution shape, scale, rate, or location parameter violated its mathematical domain.
    InvalidParameter {
        /// Parameter name.
        name: &'static str,
        /// Rejected numeric value.
        value: f64,
    },
    /// Support interval bounds were non-finite or inverted (`lower >= upper`).
    InvalidBounds {
        /// Lower bound value.
        lower: f64,
        /// Upper bound value.
        upper: f64,
    },
    /// Quantile or probability argument fell outside `[0.0, 1.0]` or open `(0.0, 1.0)` where required.
    InvalidProbability {
        /// Rejected probability value.
        value: f64,
    },
    /// Sample slice contained fewer observations than required by the estimator.
    InsufficientData {
        /// Minimum sample count required.
        required: usize,
        /// Actual sample count supplied.
        actual: usize,
    },
    /// Maximum likelihood or method-of-moments estimation failed on degenerate or out-of-support data.
    FittingFailed {
        /// Concise diagnostic reason.
        reason: &'static str,
    },
}

impl std::fmt::Display for DistributionError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidParameter { name, value } => {
                write!(f, "invalid distribution parameter {name} (got {value})")
            }
            Self::InvalidBounds { lower, upper } => {
                write!(
                    f,
                    "invalid support bounds: lower ({lower}) must be finite and less than upper ({upper})"
                )
            }
            Self::InvalidProbability { value } => {
                write!(
                    f,
                    "probability argument must lie in [0.0, 1.0] and be finite (got {value})"
                )
            }
            Self::InsufficientData { required, actual } => {
                write!(
                    f,
                    "insufficient observations for fitting (required at least {required}, got {actual})"
                )
            }
            Self::FittingFailed { reason } => {
                write!(f, "distribution parameter estimation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for DistributionError {}

/// Error variants for conjugate Bayesian updating, credible intervals, Bayes factors, and MCMC.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BayesianError {
    /// Prior hyperparameter violated positivity or finiteness requirements.
    InvalidHyperparameter {
        /// Hyperparameter name.
        name: &'static str,
        /// Rejected numeric value.
        value: f64,
    },
    /// Observed count, trial count, or sample value violated model likelihood domain.
    InvalidObservation {
        /// Concise diagnostic reason.
        reason: &'static str,
    },
    /// Requested credible interval probability mass fell outside `(0.0, 1.0)`.
    InvalidCredibleLevel {
        /// Rejected credible level.
        level: f64,
    },
    /// Category index exceeded simplex dimension `K`.
    CategoryOutOfBounds {
        /// Requested category index.
        index: usize,
        /// Simplex dimension.
        dimension: usize,
    },
    /// Metropolis-Hastings proposal scale was non-positive or non-finite.
    InvalidProposalScale {
        /// Rejected proposal standard deviation.
        scale: f64,
    },
    /// Pre-allocated MCMC chain or scratch buffer was empty or mismatched in length.
    InvalidBufferLength {
        /// Minimum or expected buffer length.
        expected: usize,
        /// Supplied buffer length.
        actual: usize,
    },
    /// Log-target or marginal likelihood evaluated to a non-finite quantity.
    NumericalInstability {
        /// Concise diagnostic reason.
        reason: &'static str,
    },
}

impl std::fmt::Display for BayesianError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidHyperparameter { name, value } => {
                write!(f, "invalid prior hyperparameter {name} (got {value})")
            }
            Self::InvalidObservation { reason } => {
                write!(f, "invalid Bayesian observation: {reason}")
            }
            Self::InvalidCredibleLevel { level } => {
                write!(
                    f,
                    "credible interval level must lie strictly in (0.0, 1.0) (got {level})"
                )
            }
            Self::CategoryOutOfBounds { index, dimension } => {
                write!(
                    f,
                    "category index {index} is out of bounds for {dimension}-simplex"
                )
            }
            Self::InvalidProposalScale { scale } => {
                write!(
                    f,
                    "MCMC proposal scale must be positive and finite (got {scale})"
                )
            }
            Self::InvalidBufferLength { expected, actual } => {
                write!(
                    f,
                    "invalid pre-allocated buffer length (expected at least {expected}, got {actual})"
                )
            }
            Self::NumericalInstability { reason } => {
                write!(f, "Bayesian numerical instability: {reason}")
            }
        }
    }
}

impl std::error::Error for BayesianError {}
