use super::*;

macro_rules! impl_error_from {
    (@cfg $feat:literal: $($err_ty:ty => $variant:ident),+ $(,)?) => {
        $(
            #[cfg(feature = $feat)]
            impl From<$err_ty> for Error {
                #[verified_engine::verified]
                fn from(err: $err_ty) -> Self {
                    Self::$variant(err)
                }
            }
        )+
    };
    ($err_ty:ty => $variant:ident) => {
        impl From<$err_ty> for Error {
            #[verified_engine::verified]
            fn from(err: $err_ty) -> Self {
                Self::$variant(err)
            }
        }
    };
}

impl_error_from!(ArenaError => Arena);
impl_error_from!(DiagnosticError => Diagnostic);
impl_error_from!(DiffeqError => Diffeq);
impl_error_from!(PdeError => Pde);
impl_error_from!(EvalError => Eval);
impl_error_from!(FitError => Fit);
impl_error_from!(GlslError => Glsl);
impl_error_from!(LexError => Lex);
impl_error_from!(LinalgError => Linalg);
impl_error_from!(MathError => Math);
impl_error_from!(MathCommonsError => MathCommons);
impl_error_from!(ScalarError => Scalar);
impl_error_from!(ParseError => Parse);
impl_error_from!(QmcError => Qmc);
impl_error_from!(RngError => Rng);
impl_error_from!(ShaderEmitError => ShaderEmit);
impl_error_from!(std::io::Error => Io);

impl From<&'static str> for Error {
    #[verified_engine::verified]
    fn from(reason: &'static str) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl_error_from!(@cfg "ai":
    AIError => AI,
);

impl_error_from!(@cfg "applied":
    AppliedError => Applied,
    ClinicalTrialError => ClinicalTrial,
    EngineeringError => Engineering,
    IsosurfaceError => Isosurface,
    LoraError => Lora,
    PharmacokineticsError => Pharmacokinetics,
    PortfolioError => Portfolio,
    VoxelGridError => VoxelGrid,
);

impl_error_from!(@cfg "biology":
    BiologyError => Biology,
    BotanicalError => Botanical,
    HodgkinHuxleyError => HodgkinHuxley,
    KineticsError => Kinetics,
    ReactionDiffusionError => ReactionDiffusion,
);

impl_error_from!(@cfg "climate":
    ClimateError => Climate,
);

impl_error_from!(@cfg "engineering":
    EngineeringDomainError => EngineeringDomain,
    BatteryError => Battery,
    HardwareError => Hardware,
    RadarError => Radar,
);

impl_error_from!(@cfg "epidemiology":
    EpidemiologyDomainError => EpidemiologyDomain,
    EpidemiologyError => Epidemiology,
    NetworkModelError => NetworkModel,
);

impl_error_from!(@cfg "physics":
    PhysicsError => Physics,
    ChaosError => Chaos,
    DoseFluenceError => DoseFluence,
    FluidError => Fluid,
    GatingError => Gating,
    HighEnergyError => HighEnergy,
    NuclearError => Nuclear,
    QuantumError => Quantum,
    SandpileError => Sandpile,
    SolidStateError => SolidState,
    StandardModelError => StandardModel,
    StatMechError => StatMech,
);

impl_error_from!(@cfg "pure_math":
    PureMathError => PureMath,
    AmbsError => Ambs,
    AnalysisError => Analysis,
    BraidError => Braid,
    CopulaError => Copula,
    EllipticCurveError => EllipticCurve,
    EloError => Elo,
    EvolutionError => Evolution,
    FractranError => Fractran,
    GameTheoryError => GameTheory,
    Glicko2Error => Glicko2,
    GraphTheoryError => GraphTheory,
    HyperbolicError => Hyperbolic,
    KalmanError => Kalman,
    KellyError => Kelly,
    KnotError => Knot,
    LinearAlgebraError => LinearAlgebra,
    MarketPricingError => MarketPricing,
    MarkovError => Markov,
    MatchModelError => MatchModel,
    MonteCarloError => MonteCarlo,
    NumberTheoryError => NumberTheory,
    OptimizationError => Optimization,
    OptionPricingError => OptionPricing,
    OuError => Ou,
    PoissonError => Poisson,
    QueueingError => Queueing,
    RiskError => Risk,
    StochasticError => Stochastic,
    Su11HyperbolicError => Su11Hyperbolic,
    TdaError => Tda,
    TensorError => Tensor,
    TransformError => Transform,
    ZipError => Zip,
);
