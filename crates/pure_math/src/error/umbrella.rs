use super::{
    AmbsError, AnalysisError, CopulaError, DiffeqError, EllipticCurveError, EloError,
    EvolutionError, FractranError, Glicko2Error, KellyError, LinearAlgebraError,
    MarketPricingError, MarkovError, MatchModelError, NumberTheoryError, OptimizationError,
    OuError, StochasticError, TdaError, TensorError, TransformError, ZipError,
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
    /// Differential equation and quadrature engine error.
    #[error("{0}")]
    Diffeq(#[from] DiffeqError),
    /// Sports Elo rating error.
    #[error("{0}")]
    Elo(#[from] EloError),
    /// Glicko-2 rating error.
    #[error("{0}")]
    Glicko2(#[from] Glicko2Error),
    /// Kelly criterion error.
    #[error("{0}")]
    Kelly(#[from] KellyError),
    /// Market pricing, devigging, and cross-book execution error.
    #[error("{0}")]
    MarketPricing(#[from] MarketPricingError),
    /// Markov process error.
    #[error("{0}")]
    Markov(#[from] MarkovError),
    /// Calibrated 2D score-grid match model error.
    #[error("{0}")]
    MatchModel(#[from] MatchModelError),
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
    fn as_diagnostic(&self) -> Option<&dyn Diagnostic> {
        match self {
            Self::Copula(err) => Some(err),
            Self::Diffeq(_) => None,
            Self::Elo(err) => Some(err),
            Self::Glicko2(err) => Some(err),
            Self::Kelly(err) => Some(err),
            Self::MarketPricing(err) => Some(err),
            Self::Markov(err) => Some(err),
            Self::MatchModel(err) => Some(err),
            Self::NumberTheory(err) => Some(err),
            Self::Ou(err) => Some(err),
            Self::Tda(err) => Some(err),
            Self::Zip(err) => Some(err),
            Self::Ambs(err) => Some(err),
            Self::Analysis(err) => Some(err),
            Self::EllipticCurve(err) => Some(err),
            Self::Evolution(err) => Some(err),
            Self::Fractran(err) => Some(err),
            Self::LinearAlgebra(err) => Some(err),
            Self::Optimization(err) => Some(err),
            Self::Stochastic(err) => Some(err),
            Self::Tensor(err) => Some(err),
            Self::Transform(err) => Some(err),
            Self::Math(err) => Some(err),
            Self::MathCommons(err) => Some(err),
        }
    }
}

impl Diagnostic for PureMathError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        match self.as_diagnostic() {
            Some(diag) => diag.severity(),
            None => Severity::Error,
        }
    }

    #[verified_engine::verified]
    fn metadata(&self) -> HashMap<String, String> {
        let mut map = match (self, self.as_diagnostic()) {
            (_, Some(diag)) => diag.metadata(),
            (Self::Diffeq(err), None) => {
                let mut m = HashMap::new();
                m.insert("error_type".to_string(), "DiffeqError".to_string());
                m.insert("description".to_string(), err.to_string());
                m
            }
            (_, None) => HashMap::new(),
        };
        map.insert("umbrella_type".to_string(), "PureMathError".to_string());
        map
    }
}
