use super::{
    AmbsError, AnalysisError, BraidError, CopulaError, EllipticCurveError, EloError,
    EvolutionError, FractranError, Glicko2Error, GraphTheoryError, HyperbolicError, KellyError,
    KnotError, LaurentError, LinearAlgebraError, MarketPricingError, MarkovError, MatchModelError,
    MonteCarloError, NumberTheoryError, OptimizationError, OptionPricingError, OuError,
    PoissonError, QueueingError, RiskError, StochasticError, Su11HyperbolicError, TdaError,
    TensorError, TransformError, ZipError,
};
use engine_linalg::LinalgError;
use math_commons::error::{
    ArenaError, EvalError, GlslError, LexError, MathCommonsError, MathError, ParseError,
    ShaderEmitError,
};
use math_diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
use math_scalars::ScalarError;
use thiserror::Error;

/// Extensible crate-level umbrella error for `pure_math` algorithms and foundational math errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PureMathError {
    /// Copula modeling error.
    #[error("{0}")]
    Copula(#[from] CopulaError),
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
    /// Monte Carlo, quasi-Monte Carlo, and bootstrap estimation error.
    #[error("{0}")]
    MonteCarlo(#[from] MonteCarloError),
    /// Number theory error.
    #[error("{0}")]
    NumberTheory(#[from] NumberTheoryError),
    /// Black-Scholes-Merton option pricing error.
    #[error("{0}")]
    OptionPricing(#[from] OptionPricingError),
    /// Ornstein-Uhlenbeck process error.
    #[error("{0}")]
    Ou(#[from] OuError),
    /// Homogeneous or inhomogeneous Poisson process error.
    #[error("{0}")]
    Poisson(#[from] PoissonError),
    /// Birth-death queueing model error.
    #[error("{0}")]
    Queueing(#[from] QueueingError),
    /// Statistical tail-risk, drawdown, and return-ratio error.
    #[error("{0}")]
    Risk(#[from] RiskError),
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
    /// Artin braid group error.
    #[error("{0}")]
    Braid(#[from] BraidError),
    /// Elliptic curve error.
    #[error("{0}")]
    EllipticCurve(#[from] EllipticCurveError),
    /// Knot theory and skein invariant error.
    #[error("{0}")]
    Knot(#[from] KnotError),
    /// Laurent polynomial exponent, span, or integer coefficient error.
    #[error("{0}")]
    Laurent(#[from] LaurentError),
    /// Unified numerical evolution engine error.
    #[error("{0}")]
    Evolution(#[from] EvolutionError),
    /// Prime-exponent FRACTRAN register machine error.
    #[error("{0}")]
    Fractran(#[from] FractranError),
    /// Graph theory solver or interchange error.
    #[error("{0}")]
    GraphTheory(#[from] GraphTheoryError),
    /// Hyperbolic geometry and geodesic billiard error.
    #[error("{0}")]
    Hyperbolic(#[from] HyperbolicError),
    /// Intrinsic `SU(1,1)` hyperbolic geometry error.
    #[error("{0}")]
    Su11Hyperbolic(#[from] Su11HyperbolicError),
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
    /// Foundational `#![no_std]` differential-equation engine error.
    #[error("{0}")]
    Diffeq(#[from] engine_diffeq::DiffeqError),
    /// Foundational `#![no_std]` PDE and SPDE engine error.
    #[error("{0}")]
    Pde(#[from] engine_pde::PdeError),
    /// Foundational mathematical kernel error.
    #[error("{0}")]
    Math(#[from] MathError),
    /// Foundational `math_commons` umbrella error.
    #[error("{0}")]
    MathCommons(#[from] MathCommonsError),
    /// Foundational `math_scalars` error.
    #[error("{0}")]
    Scalar(#[from] ScalarError),
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

impl From<ArenaError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: ArenaError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<LinalgError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: LinalgError) -> Self {
        Self::LinearAlgebra(LinearAlgebraError::from(err))
    }
}

impl From<ShaderEmitError> for PureMathError {
    #[verified_engine::verified]
    fn from(err: ShaderEmitError) -> Self {
        Self::MathCommons(MathCommonsError::from(err))
    }
}

impl From<&'static str> for PureMathError {
    #[verified_engine::verified]
    fn from(reason: &'static str) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl PureMathError {
    #[verified_engine::verified]
    fn as_diagnostic(&self) -> &dyn Diagnostic {
        match self {
            Self::Copula(err) => err,
            Self::Elo(err) => err,
            Self::Glicko2(err) => err,
            Self::Kelly(err) => err,
            Self::MarketPricing(err) => err,
            Self::Markov(err) => err,
            Self::MatchModel(err) => err,
            Self::NumberTheory(err) => err,
            Self::Ou(err) => err,
            Self::Poisson(err) => err,
            Self::Queueing(err) => err,
            Self::Risk(err) => err,
            Self::Tda(err) => err,
            Self::Zip(err) => err,
            Self::Ambs(err) => err,
            Self::Analysis(err) => err,
            Self::Braid(err) => err,
            Self::EllipticCurve(err) => err,
            Self::Knot(err) => err,
            Self::Laurent(err) => err,
            Self::Evolution(err) => err,
            Self::Fractran(err) => err,
            Self::GraphTheory(err) => err,
            Self::Hyperbolic(err) => err,
            Self::Su11Hyperbolic(err) => err,
            Self::LinearAlgebra(err) => err,
            Self::MonteCarlo(err) => err,
            Self::Optimization(err) => err,
            Self::OptionPricing(err) => err,
            Self::Stochastic(err) => err,
            Self::Tensor(err) => err,
            Self::Transform(err) => err,
            Self::Diffeq(err) => err,
            Self::Pde(err) => err,
            Self::Math(err) => err,
            Self::MathCommons(err) => err,
            Self::Scalar(err) => err,
        }
    }
}

impl Diagnostic for PureMathError {
    #[verified_engine::verified]
    fn severity(&self) -> Severity {
        self.as_diagnostic().severity()
    }

    #[verified_engine::verified]
    fn visit_fields(&self, visitor: &mut dyn DiagnosticVisitor) {
        self.as_diagnostic().visit_fields(visitor);
        visitor.record_str("umbrella_type", "PureMathError");
    }
}
