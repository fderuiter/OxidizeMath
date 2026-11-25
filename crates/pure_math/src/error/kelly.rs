use super::KellyError;
use core::fmt;

impl KellyError {
    #[verified_engine::verified]
    fn fmt_wager_inputs(&self, f: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        match self {
            Self::InvalidProbability { value } => Some(write!(
                f,
                "Invalid probability: {value} (must be between 0 and 1)"
            )),
            Self::InvalidOdds { value } => Some(write!(f, "Invalid odds: {value} (must be > 1.0)")),
            Self::InvalidNetOdds { value } => Some(write!(
                f,
                "Invalid net odds: {value} (must be finite and > 0.0)"
            )),
            Self::InvalidAmericanOdds { value } => Some(write!(
                f,
                "Invalid American odds: {value} (must be finite with |odds| >= 100.0)"
            )),
            Self::InvalidFractionalOdds {
                numerator,
                denominator,
            } => Some(write!(
                f,
                "Invalid fractional odds: {numerator}/{denominator} (numerator and denominator must be finite and > 0.0)"
            )),
            Self::InvalidImpliedProbability { value } => Some(write!(
                f,
                "Invalid implied probability: {value} (must be finite and strictly between 0.0 and 1.0)"
            )),
            Self::InvalidFraction { value } => Some(write!(
                f,
                "Invalid fraction: {value} (must be between 0 and 1)"
            )),
            Self::NoEdge { probability, odds } => Some(write!(
                f,
                "No edge: p={probability}, odds={odds} results in negative expectation"
            )),
            Self::InvalidBankroll { value } => {
                Some(write!(f, "Invalid bankroll: {value} (must be positive)"))
            }
            Self::InvalidStake { value } => Some(write!(
                f,
                "Invalid stake or weight: {value} (must be finite and > 0)"
            )),
            Self::InvalidBarrier {
                floor,
                initial_wealth,
            } => Some(write!(
                f,
                "Invalid barrier configuration: floor={floor}, initial_wealth={initial_wealth}"
            )),
            Self::InsufficientBets { required, actual } => Some(write!(
                f,
                "Insufficient bets: required at least {required}, got {actual}"
            )),
            _ => None,
        }
    }
}

impl fmt::Display for KellyError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(res) = self.fmt_wager_inputs(f) {
            return res;
        }
        match self {
            Self::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Slice length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::NotPositiveDefinite => f.write_str("Covariance matrix is not positive definite"),
            Self::InsufficientData { required, actual } => {
                write!(
                    f,
                    "Insufficient data: required at least {required}, got {actual}"
                )
            }
            Self::InvalidConfidenceLevel { value } => {
                write!(f, "Invalid confidence level: {value} (must be in (0, 1))")
            }
            Self::InvalidParameter { reason } => {
                write!(f, "Invalid portfolio parameter: {reason}")
            }
            Self::ConvergenceFailed { reason } => {
                write!(f, "Solvency root solver failed to converge: {reason}")
            }
            Self::InvalidProbabilityDistribution { sum } => {
                write!(
                    f,
                    "Invalid probability distribution: sum={sum} (must equal 1.0)"
                )
            }
            Self::InvalidMarketProbabilitySum { sum } => write!(
                f,
                "Invalid mutually exclusive market probabilities: sum={sum} (must be <= 1.0)"
            ),
            Self::EmptyPortfolio => {
                f.write_str("Portfolio or market must contain at least one wager")
            }
            Self::PortfolioTooLarge { count, max } => {
                write!(
                    f,
                    "Portfolio size {count} exceeds maximum supported capacity {max}"
                )
            }
            Self::InvalidDrawdownBounds { alpha, beta } => write!(
                f,
                "Invalid drawdown bounds: alpha={alpha}, beta={beta} (both must lie strictly in (0, 1))"
            ),
            Self::SolverConvergenceFailed {
                iterations,
                residual,
            } => write!(
                f,
                "Kelly solver failed to converge after {iterations} iterations (residual={residual})"
            ),
            Self::Linalg(err) => fmt::Display::fmt(err, f),
            _ => Ok(()),
        }
    }
}

impl std::error::Error for KellyError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::NotPositiveDefinite => Some(&engine_linalg::LinalgError::NotPositiveDefinite),
            Self::Linalg(err) => Some(err),
            _ => None,
        }
    }
}

impl From<engine_linalg::LinalgError> for KellyError {
    #[verified_engine::verified]
    fn from(err: engine_linalg::LinalgError) -> Self {
        Self::Linalg(err)
    }
}
