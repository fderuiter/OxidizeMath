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
            IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, PortfolioError,
            VoxelGridError,
        };
        pub use ::pure_math::__doctest_facade::error::*;
        pub use engine_opt::error::{FitError, OptError};
    }
    /// Doctest prelude re-exports.
    pub mod prelude {
        pub use crate::error::{
            AppliedError, BatteryError, ClinicalTrialError, EngineeringError, GameTheoryError,
            IsosurfaceError, KalmanError, LoraError, PharmacokineticsError, PortfolioError,
            VoxelGridError,
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
    pub use engine_opt;
    pub use engine_opt::fit;
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
pub(crate) static VERIFIED_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
