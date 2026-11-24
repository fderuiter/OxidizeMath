#![doc = include_str!("../README.md")]
pub mod applied;
#[allow(missing_docs)]
pub mod error;

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::{
            AppliedError, BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError,
            IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::error::*;
    }
    pub mod prelude {
        pub use crate::applied::battery_degradation::model::DegradationModel;
        pub use crate::applied::battery_degradation::{
            Capacity, Cycles, DepthOfDischarge, PowerLawModel,
        };
        pub use crate::applied::clinical_trials::sample_size::{
            calculate_sample_size_means, calculate_sample_size_proportions,
        };
        pub use crate::applied::clinical_trials::survival_analysis::{
            Observation, TimePoint, kaplan_meier,
        };
        pub use crate::applied::game_theory::evolutionary::ReplicatorDynamics;
        pub use crate::applied::pharmacokinetics::{
            BatemanModel, PKParameters, PharmacokineticModel, SuperpositionModel,
            concentration_bateman,
        };
        pub use crate::error::{
            AppliedError, BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError,
            IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::prelude::*;
    }
    pub use crate::applied;
    pub use ::pure_math::__doctest_facade::{
        ast, bootstrap, constants, diagnostics, fit, math_kernel, math_types, mesh, monte_carlo,
        nalgebra, num_complex, petgraph, poi, primitives, pure_math, qmc, rand, registry, rng,
        sensitivity, series, stochastic, wigner,
    };
}
