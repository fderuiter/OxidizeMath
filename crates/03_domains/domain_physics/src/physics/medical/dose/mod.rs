//! Terma and point-spread convolution dose deposition kernels.

/// Total Energy Released per unit Mass (TERMA) photon attenuation routines.
pub mod algorithm;
/// Radial point-spread dose deposition kernels (`DoseKernel`, `ExponentialKernel`).
pub mod kernel;

pub use algorithm::calculate_terma;
pub use kernel::{DoseKernel, ExponentialKernel};

// [cite:mmwave_radiotherapy_setup]
