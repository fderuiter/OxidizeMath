use super::{
    ChaosError, DoseFluenceError, FluidError, GatingError, HighEnergyError, NuclearError,
    QuantumError, RadarError, SandpileError, SolidStateError, StandardModelError, StatMechError,
};
use domain_applied::error::{AppliedError, KalmanError};
use engine_linalg::LinalgError;
use engine_opt::error::{FitError, OptError};
use math_commons::error::{
    EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError, ShaderEmitError,
};
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use pure_math::error::{
    AnalysisError, EvolutionError, LinearAlgebraError, MonteCarloError, OptimizationError,
    PureMathError, TensorError, TransformError,
};
use thiserror::Error;

impl Diagnostic for NuclearError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        Severity::Error
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        visitor.record_str("error_type", "NuclearError");
        visitor.record_display("description", self);
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
    /// Hydrogenic orbital and quantum mechanics error.
    #[error("{0}")]
    Quantum(#[from] QuantumError),
    /// Radar array and MUSIC estimation error.
    #[error("{0}")]
    Radar(#[from] RadarError),
    /// Abelian sandpile lattice and Kirchhoff Bareiss error.
    #[error("{0}")]
    Sandpile(#[from] SandpileError),
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
    /// Foundational `#![no_std]` curve-fitting solver error.
    #[error("{0}")]
    Fit(#[from] FitError),
    /// Foundational `#![no_std]` optimization, control, DP, MFG, and Nash engine error.
    #[error("{0}")]
    Opt(#[from] OptError),
    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
    /// Foundational `#![no_std]` differential-equation solver error.
    #[error("{0}")]
    Diffeq(#[from] engine_diffeq::DiffeqError),
    /// Foundational `#![no_std]` partial-differential-equation solver error.
    #[error("{0}")]
    Pde(#[from] engine_pde::PdeError),
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

impl From<TensorError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: TensorError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<MonteCarloError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: MonteCarloError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<OptimizationError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: OptimizationError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<ShaderEmitError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<EvalError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for PhysicsError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
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

impl From<&'static str> for PhysicsError {
    #[verified_engine::verified]
    fn from(reason: &'static str) -> Self {
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
            Self::Quantum(err) => err,
            Self::Radar(err) => err,
            Self::Sandpile(err) => err,
            Self::SolidState(err) => err,
            Self::StandardModel(err) => err,
            Self::StatMech(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::Fit(err) => err,
            Self::Opt(err) => err,
            Self::Linalg(err) => err,
            Self::Diffeq(err) => err,
            Self::Pde(err) => err,
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
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        self.as_diagnostic().visit_fields(visitor);
        visitor.record_str("umbrella_type", "PhysicsError");
    }
}
