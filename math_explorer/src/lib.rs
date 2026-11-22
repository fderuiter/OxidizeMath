#![doc = include_str!("../README.md")]

pub mod error;
pub mod prelude;

pub use diagnostics;
pub use math_commons::{ast, constants, contours, math_kernel, primitives, registry, wigner};

#[cfg(feature = "ai")]
pub use domain_ai::ai;

#[cfg(feature = "applied")]
pub use domain_applied::applied;

#[cfg(feature = "biology")]
pub use domain_biology::biology;

#[cfg(feature = "climate")]
pub use domain_climate::climate;

#[cfg(feature = "epidemiology")]
pub use domain_epidemiology::epidemiology;

#[cfg(feature = "physics")]
pub use domain_physics::physics;

#[cfg(feature = "pure_math")]
pub use ::pure_math::{math_types, pure_math};
#[cfg(feature = "pure_math")]
pub use nalgebra;
#[cfg(feature = "pure_math")]
pub use num_complex;
#[cfg(feature = "pure_math")]
pub use oxidize_core::{mesh, rng};
#[cfg(feature = "pure_math")]
pub use petgraph;
#[cfg(feature = "pure_math")]
pub use rand;
