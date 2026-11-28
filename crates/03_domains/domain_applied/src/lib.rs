#![doc = include_str!("../README.md")]
pub mod applied;
/// Domain-specific error types and diagnostic implementations for applied models.
pub mod error;

#[cfg(test)]
pub(crate) static VERIFIED_ENGINE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Hidden facade re-exporting workspace modules under `math_explorer` for doctests.
#[doc(hidden)]
pub mod __doctest_facade {
    /// Doctest error re-exports.
    pub mod error {
        pub use crate::error::{
            AppliedError, ClinicalTrialError, EngineeringError, IsosurfaceError, KalmanError,
            LoraError, PharmacokineticsError, PortfolioError, VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::error::*;
        pub use engine_opt::error::FitError;
    }
    /// Doctest prelude re-exports.
    pub mod prelude {
        pub use crate::error::{
            AppliedError, ClinicalTrialError, EngineeringError, IsosurfaceError, KalmanError,
            LoraError, PharmacokineticsError, PortfolioError, VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::prelude::*;
        pub use engine_opt::fit::FitModel;
    }
    pub use crate::applied;
    pub use ::pure_math::__doctest_facade::{
        ast, bootstrap, constants, diagnostics, math_kernel, math_types, mesh, monte_carlo,
        nalgebra, num_complex, petgraph, poi, primitives, pure_math, qmc, rand, registry, rng,
        series, wigner,
    };
    pub use engine_opt::fit;
}
