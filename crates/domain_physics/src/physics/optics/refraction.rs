//! Snell refraction, polarized Fresnel coefficients, Cauchy dispersion, 2D optical presets, and surface intersections.
//!
//! Combines analytical graded-index and negative-index 2D wavefield media ([`medium`]) with
//! polarized Fresnel reflectance/transmittance, empirical Cauchy glass dispersion, and ray-surface
//! intersection primitives ([`fresnel`]).

// [cite:optics]

pub mod fresnel;
pub mod medium;

pub use fresnel::{
    CauchyDispersion, FresnelCoefficients, GlassMaterial, SnellOutcome, brewster_angle,
    cauchy_refractive_index, critical_angle, fresnel_polarized, intersect_ray_circle_arc,
    intersect_ray_segment, normalize_2d, reflect_vector_2d, snell_refract_angle, snell_vector_2d,
};
pub use medium::{OpticalMediumConfig, OpticsPreset, snell_refract_vector, snell_refracted_angle};
