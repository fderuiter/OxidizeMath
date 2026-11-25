#![doc = include_str!("../README.md")]
/// Submodule providing `error` types, traits, and algorithms.
pub mod error;
/// Submodule providing `math_types` types, traits, and algorithms.
pub mod math_types;
pub mod pure_math;
/// Submodule providing `theory_macro` types, traits, and algorithms.
pub mod theory_macro;

#[doc(hidden)]
pub mod __macro_deps {
    pub use oxidize_core;
    pub use verified_engine;
}

#[doc(hidden)]
pub mod __doctest_facade {
    pub mod error {
        pub use crate::error::{
            AmbsError, AnalysisError, BraidError, CopulaError, EllipticCurveError, EloError,
            EvolutionError, FractranError, Glicko2Error, GraphTheoryError, HyperbolicError,
            KellyError, KnotError, LinearAlgebraError, MarketPricingError, MarkovError,
            MatchModelError, MonteCarloError, NumberTheoryError, OptimizationError,
            OptionPricingError, OuError, PoissonError, PureMathError, PureMathError as Error,
            QueueingError, StochasticError, Su11HyperbolicError, TdaError, TensorError,
            TransformError, ZipError,
        };
        pub use math_commons::error::{
            EvalError, FitError, GlslError, LexError, LinalgError, MathCommonsError, MathError,
            MathResult, ParseError, ShaderEmitError,
        };
        pub type Result<T, E = Error> = std::result::Result<T, E>;
    }
    pub mod prelude {
        pub use super::error::*;
        pub use crate::math_types::{Float, Integer, Rational};
        pub use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
        pub use math_commons::fit::FitModel;
        pub use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
        pub use math_commons::series::{PointSeries, Sample2D};
        pub use nalgebra::{self, DMatrix, DVector};
        pub use oxidize_core::mesh::{Mesh, Point3D};
        pub use oxidize_core::rng::OxidizeRng;
        pub use scientific_metadata::theory::{
            MaturityTier, ParameterConstraint, TheoryDescribable,
        };
    }
    pub use crate::pure_math::statistics::{bootstrap, monte_carlo, qmc};
    pub use crate::{math_types, pure_math};
    pub use diagnostics;
    pub use math_commons::{
        ast, constants, fit, math_kernel, poi, primitives, registry, series, stochastic, wigner,
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
