#![doc = include_str!("../README.md")]
/// RF/mmWave radar signal processing, electrochemical battery degradation, CNC tool life, and UART throughput models.
pub mod engineering;
/// Typed domain errors and umbrella diagnostics for `domain_engineering`.
pub mod error;

/// Convenience re-exports of core `domain_engineering` modules and error types.
pub mod prelude {
    pub use crate::engineering::*;
    pub use crate::error::*;
}

/// Hidden facade re-exporting workspace modules under `math_explorer` for doctests.
#[doc(hidden)]
pub mod __doctest_facade {
    /// Doctest error re-exports.
    pub mod error {
        pub use crate::error::{BatteryError, EngineeringDomainError, HardwareError, RadarError};
        pub use ::pure_math::__doctest_facade::error::*;
        pub use engine_opt::error::FitError;
    }
    /// Doctest prelude re-exports.
    pub mod prelude {
        pub use crate::error::{BatteryError, EngineeringDomainError, HardwareError, RadarError};
        pub use ::pure_math::__doctest_facade::prelude::*;
        pub use engine_opt::fit::FitModel;
    }
    pub use crate::engineering;
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
