//! Unified feature-gated error re-exports and facade umbrella error across all OxidizeMath domains.

pub use math_commons::error::{
    EvalError, FitError, GlslError, LexError, MathCommonsError, MathError, MathResult, ParseError,
};

#[cfg(feature = "ai")]
pub use domain_ai::error::AIError;

#[cfg(feature = "applied")]
pub use domain_applied::error::{
    AppliedError, BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError,
    IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
};

#[cfg(feature = "biology")]
pub use domain_biology::error::{
    BiologyError, HodgkinHuxleyError, KineticsError, ReactionDiffusionError,
};

#[cfg(feature = "epidemiology")]
pub use domain_epidemiology::error::{
    EpidemiologyDomainError, EpidemiologyError, NetworkModelError,
};

#[cfg(feature = "physics")]
pub use domain_physics::error::{
    ChaosError, DoseFluenceError, FluidError, GatingError, HighEnergyError, NuclearError,
    PhysicsError, RadarError, SolidStateError, StandardModelError, StatMechError,
};

#[cfg(feature = "pure_math")]
pub use pure_math::error::{
    AmbsError, AnalysisError, BayesianError, CopulaError, DistributionError, EllipticCurveError,
    EvolutionError, FractranError, Glicko2Error, KellyError, LinearAlgebraError, MarkovError,
    NumberTheoryError, OptimizationError, OuError, PureMathError, StochasticError, TdaError,
    TensorError, TransformError, ZipError,
};

/// Unified result alias defaulting to [`Error`] across active `math_explorer` features.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Feature-gated facade umbrella error converting any active leaf or crate umbrella error in one hop.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// AST evaluation error.
    Eval(EvalError),
    /// Zero-allocation curve-fitting solver error.
    Fit(FitError),
    /// AST-to-GLSL transpilation error.
    Glsl(GlslError),
    /// AST lexer error.
    Lex(LexError),
    /// Foundational mathematical kernel error.
    Math(MathError),
    /// Foundational `math_commons` umbrella error.
    MathCommons(MathCommonsError),
    /// AST parser error.
    Parse(ParseError),
    /// Standard I/O error (e.g., mesh export or file serialization).
    Io(std::io::Error),

    /// Artificial intelligence and machine learning error.
    #[cfg(feature = "ai")]
    AI(AIError),

    /// Applied mathematics crate umbrella error.
    #[cfg(feature = "applied")]
    Applied(AppliedError),
    /// Battery degradation error.
    #[cfg(feature = "applied")]
    Battery(BatteryError),
    /// Clinical trial design or analysis error.
    #[cfg(feature = "applied")]
    ClinicalTrial(ClinicalTrialError),
    /// Engineering calculation error.
    #[cfg(feature = "applied")]
    Engineering(EngineeringError),
    /// Game theory modeling error.
    #[cfg(feature = "applied")]
    GameTheory(GameTheoryError),
    /// Isosurface extraction error.
    #[cfg(feature = "applied")]
    Isosurface(IsosurfaceError),
    /// Kalman filter tracking error.
    #[cfg(feature = "applied")]
    Kalman(KalmanError),
    /// LoRA module ensemble error.
    #[cfg(feature = "applied")]
    Lora(LoraError),
    /// Pharmacokinetics modeling error.
    #[cfg(feature = "applied")]
    Pharmacokinetics(PharmacokineticsError),
    /// Volumetric voxel grid construction error.
    #[cfg(feature = "applied")]
    VoxelGrid(VoxelGridError),

    /// Computational biology crate umbrella error.
    #[cfg(feature = "biology")]
    Biology(BiologyError),
    /// Hodgkin-Huxley cellular neuroscience error.
    #[cfg(feature = "biology")]
    HodgkinHuxley(HodgkinHuxleyError),
    /// Enzyme kinetics error.
    #[cfg(feature = "biology")]
    Kinetics(KineticsError),
    /// Generic reaction-diffusion system error.
    #[cfg(feature = "biology")]
    ReactionDiffusion(ReactionDiffusionError),

    /// Epidemiology crate umbrella error.
    #[cfg(feature = "epidemiology")]
    EpidemiologyDomain(EpidemiologyDomainError),
    /// Deterministic compartmental epidemiology error.
    #[cfg(feature = "epidemiology")]
    Epidemiology(EpidemiologyError),
    /// Network epidemic model error.
    #[cfg(feature = "epidemiology")]
    NetworkModel(NetworkModelError),

    /// Physics crate umbrella error.
    #[cfg(feature = "physics")]
    Physics(PhysicsError),
    /// Chaos and nonlinear dynamics error.
    #[cfg(feature = "physics")]
    Chaos(ChaosError),
    /// Radiation dose-fluence modeling error.
    #[cfg(feature = "physics")]
    DoseFluence(DoseFluenceError),
    /// Fluid dynamics modeling error.
    #[cfg(feature = "physics")]
    Fluid(FluidError),
    /// Respiratory beam gating error.
    #[cfg(feature = "physics")]
    Gating(GatingError),
    /// High-energy astrophysics error.
    #[cfg(feature = "physics")]
    HighEnergy(HighEnergyError),
    /// Nuclear physics error.
    #[cfg(feature = "physics")]
    Nuclear(NuclearError),
    /// Radar array estimation error.
    #[cfg(feature = "physics")]
    Radar(RadarError),
    /// Solid-state physics solver error.
    #[cfg(feature = "physics")]
    SolidState(SolidStateError),
    /// Particle physics Standard Model error.
    #[cfg(feature = "physics")]
    StandardModel(StandardModelError),
    /// Statistical mechanics error.
    #[cfg(feature = "physics")]
    StatMech(StatMechError),

    /// Pure mathematics crate umbrella error.
    #[cfg(feature = "pure_math")]
    PureMath(PureMathError),
    /// Amicable/sociable number theory error.
    #[cfg(feature = "pure_math")]
    Ambs(AmbsError),
    /// Numerical root-finding and analysis error.
    #[cfg(feature = "pure_math")]
    Analysis(AnalysisError),
    /// Bayesian inference and MCMC error.
    #[cfg(feature = "pure_math")]
    Bayesian(BayesianError),
    /// Copula modeling error.
    #[cfg(feature = "pure_math")]
    Copula(CopulaError),
    /// Probability distribution and fitting error.
    #[cfg(feature = "pure_math")]
    Distribution(DistributionError),
    /// Elliptic curve error.
    #[cfg(feature = "pure_math")]
    EllipticCurve(EllipticCurveError),
    /// Unified numerical evolution engine error.
    #[cfg(feature = "pure_math")]
    Evolution(EvolutionError),
    /// Prime-exponent FRACTRAN register machine error.
    #[cfg(feature = "pure_math")]
    Fractran(FractranError),
    /// Glicko-2 rating error.
    #[cfg(feature = "pure_math")]
    Glicko2(Glicko2Error),
    /// Kelly criterion error.
    #[cfg(feature = "pure_math")]
    Kelly(KellyError),
    /// Numerical linear algebra error.
    #[cfg(feature = "pure_math")]
    LinearAlgebra(LinearAlgebraError),
    /// Markov process error.
    #[cfg(feature = "pure_math")]
    Markov(MarkovError),
    /// Number theory error.
    #[cfg(feature = "pure_math")]
    NumberTheory(NumberTheoryError),
    /// Mathematical optimization error.
    #[cfg(feature = "pure_math")]
    Optimization(OptimizationError),
    /// Ornstein-Uhlenbeck process error.
    #[cfg(feature = "pure_math")]
    Ou(OuError),
    /// Stochastic simulation error.
    #[cfg(feature = "pure_math")]
    Stochastic(StochasticError),
    /// Topological data analysis error.
    #[cfg(feature = "pure_math")]
    Tda(TdaError),
    /// Tensor calculus error.
    #[cfg(feature = "pure_math")]
    Tensor(TensorError),
    /// Spectral transform error.
    #[cfg(feature = "pure_math")]
    Transform(TransformError),
    /// Zero-inflated Poisson regression error.
    #[cfg(feature = "pure_math")]
    Zip(ZipError),
}

mod conversions;
mod impls;
