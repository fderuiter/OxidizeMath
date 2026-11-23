#![doc = include_str!("../README.md")]
pub mod biology;
#[allow(missing_docs)]
pub mod error;

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::{
            BiologyError, HodgkinHuxleyError, KineticsError, ReactionDiffusionError,
        };
        pub use ::domain_applied::__doctest_facade::error::*;
    }
    pub mod prelude {
        pub use crate::error::{
            BiologyError, HodgkinHuxleyError, KineticsError, ReactionDiffusionError,
        };
        pub use ::domain_applied::__doctest_facade::prelude::*;
    }
    pub use crate::biology;
    pub use ::domain_applied::applied;
    pub use ::pure_math::__doctest_facade::{
        ast, constants, diagnostics, fit, math_kernel, math_types, mesh, nalgebra, num_complex,
        petgraph, poi, primitives, pure_math, rand, registry, rng, series, wigner,
    };
}
