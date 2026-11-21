#![doc = include_str!("../README.md")]
pub mod epidemiology;
#[allow(missing_docs)]
pub mod error;

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::EpidemiologyError;
        pub use ::pure_math::__doctest_facade::error::*;
    }
    pub mod prelude {
        pub use crate::error::EpidemiologyError;
        pub use ::pure_math::__doctest_facade::prelude::*;
    }
    pub use crate::epidemiology;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, primitives, pure_math, rand, registry, rng, wigner,
    };
}
