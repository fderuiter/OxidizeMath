use super::{
    BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError, IsosurfaceError,
    KalmanError, LoraError, PharmacokineticsError, PortfolioError, VoxelGridError,
};
use engine_linalg::LinalgError;
use engine_opt::error::FitError;
use math_commons::error::{
    EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError, ShaderEmitError,
};
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use pure_math::error::{
    AnalysisError, EvolutionError, LinearAlgebraError, MonteCarloError, OptimizationError,
    PureMathError, RiskError, StochasticError, TensorError, WardropError,
};
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
    /// Markowitz mean-variance portfolio optimization error.
    #[error("{0}")]
    Portfolio(#[from] PortfolioError),
    /// Volumetric voxel grid construction error.
    #[error("{0}")]
    VoxelGrid(#[from] VoxelGridError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),
    /// Foundational `#![no_std]` curve-fitting solver error.
    #[error("{0}")]
    Fit(#[from] FitError),
    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
    /// Foundational `#![no_std]` partial-differential-equation solver error.
    #[error("{0}")]
    Pde(#[from] engine_pde::PdeError),
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

impl From<RiskError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: RiskError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<MonteCarloError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: MonteCarloError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<TensorError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: TensorError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<WardropError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: WardropError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<ShaderEmitError> for AppliedError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
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

impl From<&'static str> for AppliedError {
    #[verified_engine::verified]
    fn from(reason: &'static str) -> Self {
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
            Self::Portfolio(err) => err,
            Self::VoxelGrid(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::Fit(err) => err,
            Self::Linalg(err) => err,
            Self::Pde(err) => err,
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
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        self.as_diagnostic().visit_fields(visitor);
        visitor.record_str("umbrella_type", "AppliedError");
    }
}
