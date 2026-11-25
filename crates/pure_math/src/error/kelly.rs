use super::KellyError;
use std::fmt;

impl KellyError {
    #[verified_engine::verified]
    fn fmt_portfolio(&self, f: &mut fmt::Formatter<'_>) -> Option<fmt::Result> {
        match self {
            Self::InsufficientBets { required, actual } => Some(write!(
                f,
                "Insufficient bets: required at least {required}, got {actual}"
            )),
            Self::DimensionMismatch { expected, actual } => Some(write!(
                f,
                "Slice length mismatch: expected {expected}, got {actual}"
            )),
            Self::NotPositiveDefinite => {
                Some(f.write_str("Covariance matrix is not positive definite"))
            }
            Self::InsufficientData { required, actual } => Some(write!(
                f,
                "Insufficient data: required at least {required}, got {actual}"
            )),
            Self::InvalidConfidenceLevel { value } => Some(write!(
                f,
                "Invalid confidence level: {value} (must be in (0, 1))"
            )),
            Self::InvalidParameter { reason } => {
                Some(write!(f, "Invalid portfolio parameter: {reason}"))
            }
            Self::ConvergenceFailed { reason } => Some(write!(
                f,
                "Solvency root solver failed to converge: {reason}"
            )),
            Self::InvalidProbabilityDistribution { sum } => Some(write!(
                f,
                "Invalid probability distribution: sum={sum} (must equal 1.0)"
            )),
            Self::InvalidMarketProbabilitySum { sum } => Some(write!(
                f,
                "Invalid mutually exclusive market probabilities: sum={sum} (must be <= 1.0)"
            )),
            Self::EmptyPortfolio => {
                Some(f.write_str("Portfolio or market must contain at least one wager"))
            }
            Self::PortfolioTooLarge { count, max } => Some(write!(
                f,
                "Portfolio size {count} exceeds maximum supported capacity {max}"
            )),
            Self::InvalidDrawdownBounds { alpha, beta } => Some(write!(
                f,
                "Invalid drawdown bounds: alpha={alpha}, beta={beta} (both must lie strictly in (0, 1))"
            )),
            Self::SolverConvergenceFailed {
                iterations,
                residual,
            } => Some(write!(
                f,
                "Kelly solver failed to converge after {iterations} iterations (residual={residual})"
            )),
            Self::Linalg(err) => Some(fmt::Display::fmt(err, f)),
            _ => None,
        }
    }
}

impl fmt::Display for KellyError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(res) = self.fmt_portfolio(f) {
            return res;
        }
        match self {
            Self::InvalidProbability { value } => {
                write!(f, "Invalid probability: {value} (must be between 0 and 1)")
            }
            Self::InvalidOdds { value } => write!(f, "Invalid odds: {value} (must be > 1.0)"),
            Self::InvalidNetOdds { value } => {
                write!(f, "Invalid net odds: {value} (must be finite and > 0.0)")
            }
            Self::InvalidAmericanOdds { value } => write!(
                f,
                "Invalid American odds: {value} (must be finite with |odds| >= 100.0)"
            ),
            Self::InvalidFractionalOdds {
                numerator,
                denominator,
            } => write!(
                f,
                "Invalid fractional odds: {numerator}/{denominator} (numerator and denominator must be finite and > 0.0)"
            ),
            Self::InvalidImpliedProbability { value } => write!(
                f,
                "Invalid implied probability: {value} (must be finite and strictly between 0.0 and 1.0)"
            ),
            Self::InvalidFraction { value } => {
                write!(f, "Invalid fraction: {value} (must be between 0 and 1)")
            }
            Self::NoEdge { probability, odds } => write!(
                f,
                "No edge: p={probability}, odds={odds} results in negative expectation"
            ),
            Self::InvalidBankroll { value } => {
                write!(f, "Invalid bankroll: {value} (must be positive)")
            }
            Self::InvalidStake { value } => {
                write!(
                    f,
                    "Invalid stake or weight: {value} (must be finite and > 0)"
                )
            }
            Self::InvalidBarrier {
                floor,
                initial_wealth,
            } => write!(
                f,
                "Invalid barrier configuration: floor={floor}, initial_wealth={initial_wealth}"
            ),
            _ => unreachable!("portfolio KellyError variants handled in fmt_portfolio"),
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
