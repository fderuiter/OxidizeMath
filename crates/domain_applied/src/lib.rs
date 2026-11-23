#![doc = include_str!("../README.md")]
pub mod applied;
/// Domain-specific error types and diagnostic implementations for applied models.
pub mod error;

/// Hidden facade re-exporting workspace modules under `math_explorer` for doctests.
#[doc(hidden)]
pub mod __doctest_facade {
    /// Doctest error re-exports.
    pub mod error {
        pub use crate::error::{
            AppliedError, BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError,
            IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::error::*;
    }
    /// Doctest prelude re-exports.
    pub mod prelude {
        pub use crate::error::{
            AppliedError, BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError,
            IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::prelude::*;
    }
    pub use crate::applied;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, fit, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, poi, primitives, pure_math, rand, registry, rng, series, wigner,
    };
}
