//! Cross-domain 2D MRI k-Space Scanner and Frequency-Domain Sculpting Tool.
//!
//! Couples 2D spatial spin-echo phantoms—analytic Shepp-Logan tissue ellipses and
//! optional Schnakenberg Turing reaction-diffusion patterns (`#[cfg(feature = "biology")]`)—with
//! physical gradient trajectories (`k_space_coordinate`, `accumulated_phase`), chaotic 3D
//! `LorenzSystem` compressed-sensing paths, tactile 2D k-space brushes, and separable 2D
//! Fourier reconstruction (`simulate_signal_2d`, `inverse_dft_2d`).

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod canvas;
pub mod phantom;
pub mod recon;
pub mod state;
pub mod theory;
pub mod tool;
pub mod trajectory;
pub mod ui;

#[cfg(test)]
mod tests;
#[cfg(all(test, target_arch = "wasm32"))]
mod wasm_tests;

#[cfg(test)]
pub use tool::MriKSpaceScannerTool;

/// Spatial and frequency matrix dimension (`64x64`) for interactive 2D k-space reconstruction.
pub const KSPACE_DIM: usize = 64;

/// Total number of spatial pixels / k-space bins (`64 * 64 = 4096`).
pub const KSPACE_LEN: usize = KSPACE_DIM * KSPACE_DIM;

/// Centered DC frequency index (`N / 2 = 32`) after 2D FFT quadrant shift.
pub const KSPACE_CENTER: usize = KSPACE_DIM / 2;

/// Physical square field of view in meters (`256 mm` isotropic head slice).
pub const DEFAULT_FOV_METERS: f64 = 0.256;

// [cite:favorite_child]
