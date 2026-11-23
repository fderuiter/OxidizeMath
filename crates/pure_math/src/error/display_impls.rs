use super::{CopulaError, Glicko2Error, MarkovError, TdaError, ZipError};

impl std::fmt::Display for MarkovError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for MarkovError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            _ => None,
        }
    }
}

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

impl std::error::Error for CopulaError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Math(err) => Some(err),
            _ => None,
        }
    }
}

impl std::fmt::Display for Glicko2Error {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRating { value } => write!(f, "rating must be finite (got {value})"),
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
            Self::InvalidOpponentCount { count } => write!(f, "invalid opponent count ({count})"),
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
