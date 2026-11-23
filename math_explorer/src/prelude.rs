//! Ergonomic prelude of shared mathematical primitives, core containers, and active domain errors.

pub use crate::error::*;
pub use diagnostics::{Diagnostic, Severity};
pub use math_commons::fit::FitModel;
pub use math_commons::primitives::{NonNegativeFloat, PositiveFloat, UnitInterval};
pub use math_commons::series::{PointSeries, Sample2D};

#[cfg(feature = "pure_math")]
pub use nalgebra::{self, DMatrix, DVector};
#[cfg(feature = "pure_math")]
pub use oxidize_core::mesh::{Mesh, Point3D};
#[cfg(feature = "pure_math")]
pub use oxidize_core::rng::OxidizeRng;
#[cfg(feature = "pure_math")]
pub use pure_math::math_types::{Float, Integer, Rational};
