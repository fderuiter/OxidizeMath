#![doc = include_str!("../README.md")]
pub mod epidemiology;
/// Typed epidemiological, compartmental, and network contagion errors.
pub mod error;

#[doc(hidden)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::{EpidemiologyDomainError, EpidemiologyError, NetworkModelError};
        pub use ::pure_math::__doctest_facade::error::*;
    }
    pub mod prelude {
        pub use crate::error::{EpidemiologyDomainError, EpidemiologyError, NetworkModelError};
        pub use ::pure_math::__doctest_facade::prelude::*;
    }
    pub use crate::epidemiology;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, poi, primitives, pure_math, rand, registry, rng, series, wigner,
    };
}
