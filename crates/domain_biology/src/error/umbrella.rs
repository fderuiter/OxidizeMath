use super::{HodgkinHuxleyError, KineticsError, ReactionDiffusionError};
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;
use thiserror::Error;

/// Extensible crate-level umbrella error for `domain_biology` models and upstream math/applied errors.
#[non_exhaustive]
#[derive(Debug, Error)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum BiologyError {
    /// Hodgkin-Huxley cellular neuroscience error.
    #[error("{0}")]
    HodgkinHuxley(#[from] HodgkinHuxleyError),
    /// Enzyme kinetics error.
    #[error("{0}")]
    Kinetics(#[from] KineticsError),
    /// Generic reaction-diffusion system error.
    #[error("{0}")]
    ReactionDiffusion(#[from] ReactionDiffusionError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] math_commons::error::MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] math_commons::error::MathCommonsError),
    /// Upstream `pure_math` umbrella error.
    #[error("{0}")]
    PureMath(#[from] pure_math::error::PureMathError),
    /// Upstream `domain_applied` umbrella error.
    #[error("{0}")]
    Applied(#[from] domain_applied::error::AppliedError),
}

impl From<domain_applied::error::GameTheoryError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: domain_applied::error::GameTheoryError) -> Self {
        Self::Applied(domain_applied::error::AppliedError::from(err))
    }
}

impl From<pure_math::error::EvolutionError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::EvolutionError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<pure_math::error::AnalysisError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::AnalysisError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<pure_math::error::StochasticError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::StochasticError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<String> for BiologyError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(math_commons::error::MathError::from(reason))
    }
}

impl BiologyError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::HodgkinHuxley(err) => err,
            Self::Kinetics(err) => err,
            Self::ReactionDiffusion(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::PureMath(err) => err,
            Self::Applied(err) => err,
        }
    }
}

impl Diagnostic for BiologyError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = self.as_diagnostic().metadata();
        map.insert("umbrella_type".to_string(), "BiologyError".to_string());
        map
    }
}
