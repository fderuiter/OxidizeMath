use super::{
    ChaosError, DoseFluenceError, FluidError, GatingError, HighEnergyError, NuclearError,
    RadarError, SolidStateError, StandardModelError, StatMechError,
};
use diagnostics::{Diagnostic, Severity};
use domain_applied::error::{AppliedError, KalmanError};
use math_commons::error::{EvalError, LexError, MathCommonsError, MathError, ParseError};
use pure_math::error::{
    AnalysisError, EvolutionError, LinearAlgebraError, PureMathError, TransformError,
};
use std::collections::HashMap;
use thiserror::Error;

impl Diagnostic for NuclearError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("error_type".to_string(), "NuclearError".to_string());
        map.insert("description".to_string(), self.to_string());
        map
    }
}

/// Extensible crate-level umbrella error for `domain_physics` models and upstream math/applied errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PhysicsError {
    /// Nonlinear dynamics and chaos simulation error.
    #[error("{0}")]
    Chaos(#[from] ChaosError),
    /// Radiation dose-fluence modeling error.
    #[error("{0}")]
    DoseFluence(#[from] DoseFluenceError),
    /// Fluid dynamics modeling error.
    #[error("{0}")]
    Fluid(#[from] FluidError),
    /// Respiratory beam gating state-machine error.
    #[error("{0}")]
    Gating(#[from] GatingError),
    /// High-energy astrophysics and relativistic error.
    #[error("{0}")]
    HighEnergy(#[from] HighEnergyError),
    /// Nuclear structure and reaction error.
    #[error("{0}")]
    Nuclear(#[from] NuclearError),
    /// Radar array and MUSIC estimation error.
    #[error("{0}")]
    Radar(#[from] RadarError),
    /// Solid-state physics solver error.
    #[error("{0}")]
    SolidState(#[from] SolidStateError),
    /// Particle physics Standard Model error.
    #[error("{0}")]
    StandardModel(#[from] StandardModelError),
    /// Statistical mechanics error.
    #[error("{0}")]
    StatMech(#[from] StatMechError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),
    /// Upstream `pure_math` umbrella error.
    #[error("{0}")]
    PureMath(#[from] PureMathError),
    /// Upstream `domain_applied` umbrella error.
    #[error("{0}")]
    Applied(#[from] AppliedError),
}

impl From<KalmanError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: KalmanError) -> Self {
        Self::Applied(AppliedError::from(err))
    }
}

impl From<EvolutionError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: EvolutionError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<AnalysisError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: AnalysisError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<LinearAlgebraError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: LinearAlgebraError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<TransformError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: TransformError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<EvalError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<String> for PhysicsError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl PhysicsError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Chaos(err) => err,
            Self::DoseFluence(err) => err,
            Self::Fluid(err) => err,
            Self::Gating(err) => err,
            Self::HighEnergy(err) => err,
            Self::Nuclear(err) => err,
            Self::Radar(err) => err,
            Self::SolidState(err) => err,
            Self::StandardModel(err) => err,
            Self::StatMech(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::PureMath(err) => err,
            Self::Applied(err) => err,
        }
    }
}

impl Diagnostic for PhysicsError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = self.as_diagnostic().metadata();
        map.insert("umbrella_type".to_string(), "PhysicsError".to_string());
        map
    }
}
