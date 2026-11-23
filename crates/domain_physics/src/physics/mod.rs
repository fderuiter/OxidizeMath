#![doc = include_str!("README.md")]

/// Galactic rotation curves, dark-matter halo profiles, and scaling relations.
pub mod astrophysics;
pub mod chaos;
pub mod fluid_dynamics;
pub mod high_energy;
pub mod medical;
pub mod mri;
pub mod nuclear;
/// Geometric, wave, and Fourier optics models.
pub mod optics;
pub mod quantum;
pub mod solid_state;
pub mod standard_model;
pub mod stat_mech;

/// Error types for physics simulation modules.
pub use crate::error;

// [cite:favorite_child]
