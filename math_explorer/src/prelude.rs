//! Ergonomic prelude of shared mathematical primitives, numerical & domain solvers, and errors.

pub use crate::error::*;
pub use diagnostics::{Diagnostic, Severity};
pub use math_commons::fit::{
    CurveFit, FitModel, MAX_FIT_PARAMS, fit_model, levenberg_marquardt_numeric,
    sum_squared_errors_numeric,
};
pub use math_commons::poi::{closest_threshold_crossing, intersect_polylines};
pub use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
pub use math_commons::sensitivity::{
    MonteCarloSummary, ParameterUncertainty, SensitivityEngine, SensitivityMetric, SplitMix64,
    TornadoBounds, TornadoInterval,
};
pub use math_commons::series::{PointSeries, Sample2D};

#[cfg(feature = "pure_math")]
pub use nalgebra::{self, DMatrix, DVector};
#[cfg(feature = "pure_math")]
pub use oxidize_core::mesh::{Mesh, Point3D};
#[cfg(feature = "pure_math")]
pub use oxidize_core::rng::OxidizeRng;
#[cfg(feature = "pure_math")]
pub use pure_math::math_types::{Float, Integer, Rational};
#[cfg(feature = "pure_math")]
pub use pure_math::pure_math::analysis::ode::{
    ArrayState, Euler, Heun, IntegrationMethod, OdeModel, OdeSystem, RungeKutta4, Solver,
    SolverExt, TimeStepper, VecState,
};
#[cfg(feature = "pure_math")]
pub use pure_math::pure_math::statistics::glicko2::{
    GlickoPlayer, MatchResult, Rating, RatingDeviation, SystemConstant, Volatility, update_rating,
};
#[cfg(feature = "pure_math")]
pub use pure_math::pure_math::statistics::kelly::{
    BankrollFraction, Odds, expected_growth_rate, kelly_fraction,
};
#[cfg(feature = "pure_math")]
pub use pure_math::pure_math::statistics::tda::{
    PersistenceInterval, RipsPersistenceResult, SimplicialComplex, TdaFiltrationWorkspace,
    betti_numbers, compute_rips_persistence,
};

#[cfg(feature = "applied")]
pub use domain_applied::applied::battery_degradation::model::DegradationModel;
#[cfg(feature = "applied")]
pub use domain_applied::applied::battery_degradation::{
    Capacity, Cycles, DepthOfDischarge, PowerLawModel,
};
#[cfg(feature = "applied")]
pub use domain_applied::applied::clinical_trials::sample_size::{
    calculate_sample_size_means, calculate_sample_size_proportions,
};
#[cfg(feature = "applied")]
pub use domain_applied::applied::clinical_trials::survival_analysis::{
    Observation, TimePoint, kaplan_meier,
};
#[cfg(feature = "applied")]
pub use domain_applied::applied::game_theory::evolutionary::ReplicatorDynamics;
#[cfg(feature = "applied")]
pub use domain_applied::applied::pharmacokinetics::{
    BatemanModel, PKParameters, PharmacokineticModel, SuperpositionModel, concentration_bateman,
};

#[cfg(feature = "ai")]
pub use domain_ai::ai::transformer::attention::scaled_dot_product_attention;
#[cfg(feature = "ai")]
pub use domain_ai::ai::transformer::{Encoder, FeedForward, LayerNorm, MultiHeadAttention};

#[cfg(feature = "biology")]
pub use domain_biology::biology::neuroscience::HodgkinHuxleyNeuron;

#[cfg(feature = "physics")]
pub use domain_physics::physics::chaos::lorenz::{LorenzState, LorenzSystem};
