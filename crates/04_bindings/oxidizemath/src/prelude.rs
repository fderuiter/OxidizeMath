//! Ergonomic prelude of shared mathematical primitives, core containers, and active domain errors.

pub use crate::error::*;
pub use diagnostics::{Diagnostic, DiagnosticVisitor, Severity};
pub use engine_opt::fit::FitModel;
pub use math_commons::primitives::{
    Interval, NonNegativeFloat, NonNegativeInterval, PositiveFloat, PositiveInterval, UnitInterval,
    UnitIntervalEnclosure,
};
pub use math_commons::series::{PointSeries, Sample2D};
pub use math_scalars::{ComplexField, Dual, Field, RealField, RefOps, Ring, Scalar};
pub use scientific_metadata::theory::{MaturityTier, ParameterConstraint, TheoryDescribable};

#[cfg(feature = "pure_math")]
pub use nalgebra::{self, DMatrix, DVector};
#[cfg(feature = "pure_math")]
pub use oxidize_core::mesh::{ColoredMesh, Mesh, Point3D, VertexColor, cortical_thickness_batch};
#[cfg(feature = "pure_math")]
pub use oxidize_core::rng::{
    BASE_SEED, HaltonView, NiederreiterView, OxidizeRng, RngScalar, SobolView,
};
#[cfg(feature = "pure_math")]
pub use pure_math::math_types::{Float, Integer, Rational};
#[cfg(feature = "pure_math")]
pub use pure_math::pure_math::differential_geometry::CotangentLaplacian;

#[cfg(feature = "biology")]
pub use domain_biology::biology::diffusion::CotangentMeshDiffusion;
#[cfg(feature = "biology")]
pub use domain_biology::biology::neuroscience::CorticalSheetRunner;
