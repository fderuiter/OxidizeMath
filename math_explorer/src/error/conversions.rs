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

impl_error_from!(EvalError => Eval);
impl_error_from!(FitError => Fit);
impl_error_from!(GlslError => Glsl);
impl_error_from!(LexError => Lex);
impl_error_from!(MathError => Math);
impl_error_from!(MathCommonsError => MathCommons);
impl_error_from!(ParseError => Parse);
impl_error_from!(std::io::Error => Io);

impl From<String> for Error {
    #[verified_engine::verified]
    fn from(reason: String) -> Self {
        Self::Math(MathError::from(reason))
    }
}

impl_error_from!(@cfg "ai":
    AIError => AI,
);

impl_error_from!(@cfg "applied":
    AppliedError => Applied,
    BatteryError => Battery,
    ClinicalTrialError => ClinicalTrial,
    EngineeringError => Engineering,
    GameTheoryError => GameTheory,
    IsosurfaceError => Isosurface,
    KalmanError => Kalman,
    LoraError => Lora,
    PharmacokineticsError => Pharmacokinetics,
    VoxelGridError => VoxelGrid,
);

impl_error_from!(@cfg "biology":
    BiologyError => Biology,
    HodgkinHuxleyError => HodgkinHuxley,
    KineticsError => Kinetics,
    ReactionDiffusionError => ReactionDiffusion,
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
    RadarError => Radar,
    SandpileError => Sandpile,
    SolidStateError => SolidState,
    StandardModelError => StandardModel,
    StatMechError => StatMech,
);

impl_error_from!(@cfg "pure_math":
    PureMathError => PureMath,
    AmbsError => Ambs,
    AnalysisError => Analysis,
    CopulaError => Copula,
    EllipticCurveError => EllipticCurve,
    EvolutionError => Evolution,
    FractranError => Fractran,
    Glicko2Error => Glicko2,
    KellyError => Kelly,
    LinearAlgebraError => LinearAlgebra,
    MarkovError => Markov,
    NumberTheoryError => NumberTheory,
    OptimizationError => Optimization,
    OuError => Ou,
    StochasticError => Stochastic,
    TdaError => Tda,
    TensorError => Tensor,
    TransformError => Transform,
    ZipError => Zip,
);
