use super::{EpidemiologyError, NetworkModelError};
use diagnostics::{Diagnostic, Severity};
use std::collections::HashMap;

/// Extensible crate-level umbrella error for `domain_epidemiology` models and upstream math errors.
#[non_exhaustive]
#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum EpidemiologyDomainError {
    /// Deterministic compartmental or matrix dynamics epidemiology error.
    Epidemiology(EpidemiologyError),
    /// Network epidemic model error.
    NetworkModel(NetworkModelError),
    /// Foundational mathematical kernel error.
    Math(math_commons::error::MathError),
    /// Foundational `math_commons` umbrella error.
    MathCommons(math_commons::error::MathCommonsError),
    /// Upstream `pure_math` umbrella error (including stochastic Gillespie solvers).
    PureMath(pure_math::error::PureMathError),
}

impl std::fmt::Display for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Epidemiology(err) => std::fmt::Display::fmt(err, f),
            Self::NetworkModel(err) => std::fmt::Display::fmt(err, f),
            Self::Math(err) => std::fmt::Display::fmt(err, f),
            Self::MathCommons(err) => std::fmt::Display::fmt(err, f),
            Self::PureMath(err) => std::fmt::Display::fmt(err, f),
        }
    }
}

impl std::error::Error for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Epidemiology(err) => Some(err),
            Self::NetworkModel(err) => Some(err),
            Self::Math(err) => Some(err),
            Self::MathCommons(err) => Some(err),
            Self::PureMath(err) => Some(err),
        }
    }
}

impl From<EpidemiologyError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: EpidemiologyError) -> Self {
        Self::Epidemiology(err)
    }
}

impl From<NetworkModelError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: NetworkModelError) -> Self {
        Self::NetworkModel(err)
    }
}

impl From<math_commons::error::MathError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: math_commons::error::MathError) -> Self {
        Self::Math(err)
    }
}

impl From<math_commons::error::MathCommonsError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: math_commons::error::MathCommonsError) -> Self {
        Self::MathCommons(err)
    }
}

impl From<pure_math::error::PureMathError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::PureMathError) -> Self {
        Self::PureMath(err)
    }
}

impl From<pure_math::error::StochasticError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::StochasticError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<pure_math::error::AnalysisError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::AnalysisError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<pure_math::error::EvolutionError> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(err: pure_math::error::EvolutionError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<String> for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(math_commons::error::MathError::from(reason))
    }
}

impl EpidemiologyDomainError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Epidemiology(err) => err,
            Self::NetworkModel(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::PureMath(err) => err,
        }
    }
}

impl Diagnostic for EpidemiologyDomainError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = self.as_diagnostic().metadata();
        map.insert(
            "umbrella_type".to_string(),
            "EpidemiologyDomainError".to_string(),
        );
        map
    }
}
