#![doc = include_str!("../README.md")]
#[allow(missing_docs)]
pub mod error;
#[allow(missing_docs)]
pub mod math_types;
pub mod pure_math;
#[allow(missing_docs)]
pub mod theory_macro;

#[doc(hidden)]
pub mod __macro_deps {
    pub use oxidize_core;
    pub use verified_engine;
}

#[doc(hidden)]
#[allow(missing_docs)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::{
            CopulaError, Glicko2Error, KellyError, MarkovError, NumberTheoryError, OuError,
            TdaError, ZipError,
        };
        pub use math_commons::error::{MathError, MathResult};
    }
    pub mod prelude {
        pub use super::error::*;
        pub use crate::math_types::{Float, Integer, Rational};
        pub use diagnostics::{Diagnostic, Severity};
        pub use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
        pub use nalgebra::{self, DMatrix, DVector};
        pub use oxidize_core::mesh::{Mesh, Point3D};
        pub use oxidize_core::rng::OxidizeRng;
    }
    pub use crate::{math_types, pure_math};
    pub use diagnostics;
    pub use math_commons::{ast, constants, math_kernel, primitives, registry, wigner};
    pub use nalgebra;
    pub use num_complex;
    pub use oxidize_core::{mesh, rng};
    pub use petgraph;
    pub use rand;
}
