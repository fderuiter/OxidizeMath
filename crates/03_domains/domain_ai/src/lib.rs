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
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, fit, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, poi, primitives, pure_math, rand, registry, rng, series, wigner,
    };
}
