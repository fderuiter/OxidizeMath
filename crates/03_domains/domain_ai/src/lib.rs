#![doc = include_str!("../README.md")]
pub mod ai;
/// Domain-specific error types and diagnostic metadata for AI models.
pub mod error;

/// Convenience re-exports of core AI modules and error types.
pub mod prelude {
    pub use crate::ai::*;
    pub use crate::error::*;
}

/// Hidden facade re-exporting workspace modules under `math_explorer` for doctests.
#[doc(hidden)]
pub mod __doctest_facade {
    /// Doctest error re-exports.
    pub mod error {
        pub use crate::error::AIError;
        pub use ::pure_math::__doctest_facade::error::*;
    }
    /// Doctest prelude re-exports.
    pub mod prelude {
        pub use crate::error::AIError;
        pub use ::pure_math::__doctest_facade::prelude::*;
    }
    pub use crate::ai;
    pub use ::engine_opt;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, poi, primitives, pure_math, rand, registry, rng, series, wigner,
    };
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[cfg(test)]
pub(crate) static VERIFIED_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
