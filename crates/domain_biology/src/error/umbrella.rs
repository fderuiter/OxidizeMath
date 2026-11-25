use super::{BotanicalError, HodgkinHuxleyError, KineticsError, ReactionDiffusionError};
use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_commons::error::{
    ArenaError, EvalError, FitError, GlslError, LexError, MathCommonsError, ParseError,
    ShaderEmitError,
};
use pure_math::error::{
    LinalgError, LinearAlgebraError, MonteCarloError, OptimizationError, TensorError,
};
use thiserror::Error;

/// Extensible crate-level umbrella error for `domain_biology` models and upstream math/applied errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum BiologyError {
    /// Botanical morphogenesis (L-system or phyllotaxis) error.
    #[error("{0}")]
    Botanical(#[from] BotanicalError),
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

impl From<LinearAlgebraError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: LinearAlgebraError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<OptimizationError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: OptimizationError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<MonteCarloError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: MonteCarloError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<TensorError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: TensorError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<ArenaError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: ArenaError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LinalgError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: LinalgError) -> Self {
        Self::PureMath(pure_math::error::PureMathError::from(err))
    }
}

impl From<FitError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: FitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<EvalError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ShaderEmitError> for BiologyError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
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
            Self::Botanical(err) => err,
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
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        self.as_diagnostic().visit_fields(visitor);
        visitor.record_str("umbrella_type", "BiologyError");
    }
}
