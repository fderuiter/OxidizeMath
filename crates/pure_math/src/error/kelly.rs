use super::KellyError;

impl From<engine_linalg::LinalgError> for KellyError {
    #[verified_engine::verified]
    fn from(err: engine_linalg::LinalgError) -> Self {
        Self::Linalg(err)
    }
}

#[verified_engine::verified]
fn fmt_odds(err: &KellyError, f: &mut std::fmt::Formatter<'_>) -> Option<std::fmt::Result> {
    match err {
        KellyError::InvalidProbability { value } => Some(write!(
            f,
            "Invalid probability: {value} (must be between 0 and 1)"
        )),
        KellyError::InvalidOdds { value } => {
            Some(write!(f, "Invalid odds: {value} (must be > 1.0)"))
        }
        KellyError::InvalidNetOdds { value } => Some(write!(
            f,
            "Invalid net odds: {value} (must be finite and > 0.0)"
        )),
        KellyError::InvalidAmericanOdds { value } => Some(write!(
            f,
            "Invalid American odds: {value} (must be finite with |odds| >= 100.0)"
        )),
        KellyError::InvalidFractionalOdds {
            numerator,
            denominator,
        } => Some(write!(
            f,
            "Invalid fractional odds: {numerator}/{denominator} (numerator and denominator must be finite and > 0.0)"
        )),
        KellyError::InvalidImpliedProbability { value } => Some(write!(
            f,
            "Invalid implied probability: {value} (must be finite and strictly between 0.0 and 1.0)"
        )),
        KellyError::InvalidFraction { value } => Some(write!(
            f,
            "Invalid fraction: {value} (must be between 0 and 1)"
        )),
        KellyError::NoEdge { probability, odds } => Some(write!(
            f,
            "No edge: p={probability}, odds={odds} results in negative expectation"
        )),
        _ => None,
    }
}

#[verified_engine::verified]
fn fmt_risk(err: &KellyError, f: &mut std::fmt::Formatter<'_>) -> Option<std::fmt::Result> {
    match err {
        KellyError::InvalidBankroll { value } => {
            Some(write!(f, "Invalid bankroll: {value} (must be positive)"))
        }
        KellyError::InvalidStake { value } => Some(write!(
            f,
            "Invalid stake or weight: {value} (must be finite and > 0)"
        )),
        KellyError::InvalidBarrier {
            floor,
            initial_wealth,
        } => Some(write!(
            f,
            "Invalid barrier configuration: floor={floor}, initial_wealth={initial_wealth}"
        )),
        KellyError::InsufficientBets { required, actual } => Some(write!(
            f,
            "Insufficient bets: required at least {required}, got {actual}"
        )),
        KellyError::DimensionMismatch { expected, actual } => Some(write!(
            f,
            "Slice length mismatch: expected {expected}, got {actual}"
        )),
        KellyError::NotPositiveDefinite => {
            Some(write!(f, "Covariance matrix is not positive definite"))
        }
        KellyError::InsufficientData { required, actual } => Some(write!(
            f,
            "Insufficient data: required at least {required}, got {actual}"
        )),
        KellyError::InvalidConfidenceLevel { value } => Some(write!(
            f,
            "Invalid confidence level: {value} (must be in (0, 1))"
        )),
        _ => None,
    }
}

#[verified_engine::verified]
fn fmt_portfolio(err: &KellyError, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match err {
        KellyError::InvalidParameter { reason } => {
            write!(f, "Invalid portfolio parameter: {reason}")
        }
        KellyError::ConvergenceFailed { reason } => {
            write!(f, "Solvency root solver failed to converge: {reason}")
        }
        KellyError::InvalidProbabilityDistribution { sum } => {
            write!(
                f,
                "Invalid probability distribution: sum={sum} (must equal 1.0)"
            )
        }
        KellyError::InvalidMarketProbabilitySum { sum } => write!(
            f,
            "Invalid mutually exclusive market probabilities: sum={sum} (must be <= 1.0)"
        ),
        KellyError::EmptyPortfolio => {
            write!(f, "Portfolio or market must contain at least one wager")
        }
        KellyError::PortfolioTooLarge { count, max } => {
            write!(
                f,
                "Portfolio size {count} exceeds maximum supported capacity {max}"
            )
        }
        KellyError::InvalidDrawdownBounds { alpha, beta } => write!(
            f,
            "Invalid drawdown bounds: alpha={alpha}, beta={beta} (both must lie strictly in (0, 1))"
        ),
        KellyError::SolverConvergenceFailed {
            iterations,
            residual,
        } => write!(
            f,
            "Kelly solver failed to converge after {iterations} iterations (residual={residual})"
        ),
        KellyError::Linalg(inner) => std::fmt::Display::fmt(inner, f),
        _ => Ok(()),
    }
}

impl std::fmt::Display for KellyError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(res) = fmt_odds(self, f) {
            return res;
        }
        if let Some(res) = fmt_risk(self, f) {
            return res;
        }
        fmt_portfolio(self, f)
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
