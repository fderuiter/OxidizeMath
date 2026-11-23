use super::{
    BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError, IsosurfaceError,
    KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
};
use diagnostics::{Diagnostic, Severity};
use math_commons::error::{
    EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError,
};
use pure_math::error::{
    AnalysisError, EvolutionError, LinearAlgebraError, OptimizationError, PureMathError,
    StochasticError,
};
use std::collections::HashMap;
use thiserror::Error;

/// Extensible crate-level umbrella error for `domain_applied` modules and upstream math errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum AppliedError {
    /// Battery degradation modeling error.
    #[error("{0}")]
    Battery(#[from] BatteryError),
    /// Clinical trial design or analysis error.
    #[error("{0}")]
    ClinicalTrial(#[from] ClinicalTrialError),
    /// Engineering calculation error.
    #[error("{0}")]
    Engineering(#[from] EngineeringError),
    /// Game theory modeling error.
    #[error("{0}")]
    GameTheory(#[from] GameTheoryError),
    /// Isosurface extraction error.
    #[error("{0}")]
    Isosurface(#[from] IsosurfaceError),
    /// Kalman filter tracking error.
    #[error("{0}")]
    Kalman(#[from] KalmanError),
    /// LoRA module ensemble error.
    #[error("{0}")]
    Lora(#[from] LoraError),
    /// Pharmacokinetics modeling error.
    #[error("{0}")]
    Pharmacokinetics(#[from] PharmacokineticsError),
    /// Volumetric voxel grid construction error.
    #[error("{0}")]
    VoxelGrid(#[from] VoxelGridError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),
    /// Upstream `pure_math` umbrella error.
    #[error("{0}")]
    PureMath(#[from] PureMathError),
}

impl From<AnalysisError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: AnalysisError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<EvolutionError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: EvolutionError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<OptimizationError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: OptimizationError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<LinearAlgebraError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: LinearAlgebraError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<StochasticError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: StochasticError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<EvalError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<String> for AppliedError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl AppliedError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Battery(err) => err,
            Self::ClinicalTrial(err) => err,
            Self::Engineering(err) => err,
            Self::GameTheory(err) => err,
            Self::Isosurface(err) => err,
            Self::Kalman(err) => err,
            Self::Lora(err) => err,
            Self::Pharmacokinetics(err) => err,
            Self::VoxelGrid(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::PureMath(err) => err,
        }
    }
}

impl Diagnostic for AppliedError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = self.as_diagnostic().metadata();
        map.insert("umbrella_type".to_string(), "AppliedError".to_string());
        map
    }
}
