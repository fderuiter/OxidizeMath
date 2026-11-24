//! Ergonomic prelude of shared mathematical primitives, core containers, and active domain errors.

pub use crate::error::*;
pub use diagnostics::{Diagnostic, Severity};
pub use math_commons::fit::FitModel;
pub use math_commons::primitives::{
    Interval, NonNegativeFloat, NonNegativeInterval, PositiveFloat, PositiveInterval, UnitInterval,
    UnitIntervalEnclosure,
};
pub use math_commons::series::{PointSeries, Sample2D};

#[cfg(feature = "pure_math")]
pub use nalgebra::{self, DMatrix, DVector};
#[cfg(feature = "pure_math")]
pub use oxidize_core::mesh::{ColoredMesh, Mesh, Point3D, VertexColor};
#[cfg(feature = "pure_math")]
pub use oxidize_core::rng::OxidizeRng;
#[cfg(feature = "pure_math")]
pub use pure_math::math_types::{Float, Integer, Rational};
#[cfg(feature = "pure_math")]
pub use pure_math::pure_math::differential_geometry::CotangentLaplacian;

#[cfg(feature = "applied")]
pub use domain_applied::applied::freesurfer::cortical_thickness_batch;

#[cfg(feature = "biology")]
pub use domain_biology::biology::diffusion::CotangentMeshDiffusion;
#[cfg(feature = "biology")]
pub use domain_biology::biology::neuroscience::CorticalSheetRunner;
