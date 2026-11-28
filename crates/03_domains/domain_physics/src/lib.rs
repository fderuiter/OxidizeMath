#![doc = include_str!("../README.md")]
/// Typed physical domain errors across chaos, relativity, fluids, nuclear, medical, and quantum modules.
pub mod error;
pub mod physics;

/// Convenience re-exports of core `domain_physics` modules and error types.
pub mod prelude {
    pub use crate::error::*;
    pub use crate::physics::*;
}

#[doc(hidden)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::{
            ChaosError, DoseFluenceError, FluidError, GatingError, HighEnergyError, NuclearError,
            PhysicsError, QuantumError, SandpileError, SolidStateError, StandardModelError,
            StatMechError,
        };
        pub use ::pure_math::__doctest_facade::error::*;
        pub use engine_diffeq::KalmanError;
        pub use engine_opt::error::FitError;
    }
    pub mod prelude {
        pub use crate::error::{
            ChaosError, DoseFluenceError, FluidError, GatingError, HighEnergyError, NuclearError,
            PhysicsError, QuantumError, SandpileError, SolidStateError, StandardModelError,
            StatMechError,
        };
        pub use ::pure_math::__doctest_facade::prelude::*;
        pub use engine_diffeq::KalmanError;
        pub use engine_opt::fit::FitModel;
    }
    pub use crate::physics;
    pub use ::pure_math::__doctest_facade::{
        ast, bootstrap, constants, diagnostics, math_kernel, math_types, mesh, monte_carlo,
        nalgebra, num_complex, petgraph, poi, primitives, pure_math, qmc, rand, registry, rng,
        series, wigner,
    };
    pub use engine_opt::fit;
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
pub(crate) static VERIFIED_ALLOC_TEST_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());
