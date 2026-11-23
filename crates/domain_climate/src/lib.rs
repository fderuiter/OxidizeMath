#![doc = include_str!("../README.md")]
pub mod climate;
pub mod error;

/// Prelude exports for the climate modeling domain.
pub mod prelude {
    pub use crate::climate::*;
    pub use crate::error::ClimateError;
}

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::ClimateError;
        pub use ::pure_math::__doctest_facade::error::*;
    }
    pub mod prelude {
        pub use crate::error::ClimateError;
        pub use ::pure_math::__doctest_facade::prelude::*;
    }
    pub use crate::climate;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, fit, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, poi, primitives, pure_math, rand, registry, rng, series, wigner,
    };
}
