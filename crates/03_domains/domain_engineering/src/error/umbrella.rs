use super::{BatteryError, HardwareError, RadarError};
use engine_diffeq::{DiffeqError, KalmanError};
use engine_linalg::LinalgError;
use engine_opt::error::FitError;
use math_commons::error::{
    ArenaError, EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError,
    ShaderEmitError,
};
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use pure_math::error::{
    AnalysisError, LinearAlgebraError, OptimizationError, PureMathError, TransformError,
};
use thiserror::Error;

/// Extensible crate-level umbrella error for `domain_engineering` solvers and upstream math engines.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum EngineeringDomainError {
    /// Battery state-of-health and aging calibration error.
    #[error("{0}")]
    Battery(#[from] BatteryError),
    /// CNC machining and UART hardware calculation error.
    #[error("{0}")]
    Hardware(#[from] HardwareError),
    /// Radar array, MUSIC estimation, and tracking error.
    #[error("{0}")]
    Radar(#[from] RadarError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),
    /// Foundational `#![no_std]` curve-fitting solver error.
    #[error("{0}")]
    Fit(#[from] FitError),
    /// Foundational `#![no_std]` differential-equation solver error.
    #[error("{0}")]
    Diffeq(#[from] DiffeqError),
    /// Foundational `#![no_std]` Kalman filter tracking error.
    #[error("{0}")]
    Kalman(#[from] KalmanError),
    /// Foundational `#![no_std]` linear algebra engine error.
    #[error("{0}")]
    Linalg(#[from] LinalgError),
    /// Upstream `pure_math` umbrella error.
    #[error("{0}")]
    PureMath(#[from] PureMathError),
}

impl From<AnalysisError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: AnalysisError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<LinearAlgebraError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: LinearAlgebraError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<OptimizationError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: OptimizationError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<TransformError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: TransformError) -> Self {
        Self::PureMath(PureMathError::from(err))
    }
}

impl From<ArenaError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: ArenaError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ShaderEmitError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<EvalError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<&'static str> for EngineeringDomainError {
    #[verified_engine::verified]
    fn from(reason: &'static str) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl EngineeringDomainError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Battery(err) => err,
            Self::Hardware(err) => err,
            Self::Radar(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::Fit(err) => err,
            Self::Diffeq(err) => err,
            Self::Kalman(err) => err,
            Self::Linalg(err) => err,
            Self::PureMath(err) => err,
        }
    }
}

impl Diagnostic for EngineeringDomainError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        self.as_diagnostic().visit_fields(visitor);
        visitor.record_str("umbrella_type", "EngineeringDomainError");
    }
}
