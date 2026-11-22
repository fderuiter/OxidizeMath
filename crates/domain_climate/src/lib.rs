#![doc = include_str!("../README.md")]
pub mod climate;
#[allow(missing_docs)]
pub mod error;

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __doctest_facade {
    pub use crate::climate;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, error, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, prelude, primitives, pure_math, rand, registry, rng, wigner,
    };
}
