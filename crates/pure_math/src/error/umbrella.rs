use super::{
    AmbsError, AnalysisError, CopulaError, EllipticCurveError, EvolutionError, FractranError,
    Glicko2Error, KellyError, LinearAlgebraError, MarkovError, NumberTheoryError,
    OptimizationError, OuError, StochasticError, TdaError, TensorError, TransformError, ZipError,
};
use diagnostics::{Diagnostic, Severity};
use math_commons::error::{
    EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError,
};
use std::collections::HashMap;
use thiserror::Error;

/// Extensible crate-level umbrella error for `pure_math` algorithms and foundational math errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PureMathError {
    /// Copula modeling error.
    #[error("{0}")]
    Copula(#[from] CopulaError),
    /// Glicko-2 rating error.
    #[error("{0}")]
    Glicko2(#[from] Glicko2Error),
    /// Kelly criterion error.
    #[error("{0}")]
    Kelly(#[from] KellyError),
    /// Markov process error.
    #[error("{0}")]
    Markov(#[from] MarkovError),
    /// Number theory error.
    #[error("{0}")]
    NumberTheory(#[from] NumberTheoryError),
    /// Ornstein-Uhlenbeck process error.
    #[error("{0}")]
    Ou(#[from] OuError),
    /// Topological data analysis error.
    #[error("{0}")]
    Tda(#[from] TdaError),
    /// Zero-inflated Poisson regression error.
    #[error("{0}")]
    Zip(#[from] ZipError),
    /// Amicable/sociable number theory error.
    #[error("{0}")]
    Ambs(#[from] AmbsError),
    /// Numerical root-finding and analysis error.
    #[error("{0}")]
    Analysis(#[from] AnalysisError),
    /// Elliptic curve error.
    #[error("{0}")]
    EllipticCurve(#[from] EllipticCurveError),
    /// Unified numerical evolution engine error.
    #[error("{0}")]
    Evolution(#[from] EvolutionError),
    /// Prime-exponent FRACTRAN register machine error.
    #[error("{0}")]
    Fractran(#[from] FractranError),
    /// Numerical linear algebra error.
    #[error("{0}")]
    LinearAlgebra(#[from] LinearAlgebraError),
    /// Mathematical optimization error.
    #[error("{0}")]
    Optimization(#[from] OptimizationError),
    /// Stochastic simulation error.
    #[error("{0}")]
    Stochastic(#[from] StochasticError),
    /// Tensor calculus error.
    #[error("{0}")]
    Tensor(#[from] TensorError),
    /// Spectral transform error.
    #[error("{0}")]
    Transform(#[from] TransformError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),
}

impl From<EvalError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: EvalError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<GlslError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: GlslError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LexError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: LexError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<ParseError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: ParseError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<String> for PureMathError {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl PureMathError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Copula(err) => err,
            Self::Glicko2(err) => err,
            Self::Kelly(err) => err,
            Self::Markov(err) => err,
            Self::NumberTheory(err) => err,
            Self::Ou(err) => err,
            Self::Tda(err) => err,
            Self::Zip(err) => err,
            Self::Ambs(err) => err,
            Self::Analysis(err) => err,
            Self::EllipticCurve(err) => err,
            Self::Evolution(err) => err,
            Self::Fractran(err) => err,
            Self::LinearAlgebra(err) => err,
            Self::Optimization(err) => err,
            Self::Stochastic(err) => err,
            Self::Tensor(err) => err,
            Self::Transform(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
        }
    }
}

impl Diagnostic for PureMathError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = self.as_diagnostic().metadata();
        map.insert("umbrella_type".to_string(), "PureMathError".to_string());
        map
    }
}
