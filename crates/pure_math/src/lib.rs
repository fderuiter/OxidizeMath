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
            AmbsError, AnalysisError, CopulaError, EllipticCurveError, EvolutionError,
            FractranError, Glicko2Error, KellyError, LinearAlgebraError, MarketPricingError,
            MarkovError, NumberTheoryError, OptimizationError, OuError, PureMathError,
            PureMathError as Error, StochasticError, TdaError, TensorError, TransformError,
            ZipError,
        };
        pub use math_commons::error::{
            EvalError, FitError, GlslError, LexError, MathCommonsError, MathError, MathResult,
            ParseError,
        };
        pub type Result<T, E = Error> = std::result::Result<T, E>;
    }
    pub mod prelude {
        pub use super::error::*;
        pub use crate::math_types::{Float, Integer, Rational};
        pub use diagnostics::{Diagnostic, Severity};
        pub use math_commons::fit::FitModel;
        pub use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
        pub use math_commons::series::{PointSeries, Sample2D};
        pub use nalgebra::{self, DMatrix, DVector};
        pub use oxidize_core::mesh::{Mesh, Point3D};
        pub use oxidize_core::rng::OxidizeRng;
    }
    pub use crate::{math_types, pure_math};
    pub use diagnostics;
    pub use math_commons::{
        ast, constants, fit, math_kernel, poi, primitives, registry, series, wigner,
    };
    pub use nalgebra;
    pub use num_complex;
    pub use oxidize_core::{mesh, rng};
    pub use petgraph;
    pub use rand;
}

#[cfg(test)]
#[global_allocator]
static TEST_ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;
