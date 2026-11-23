//! # Continuous Cellular Automata (Lenia)
//!
//! Implements spectral 2D **Lenia** continuous cellular automata on a periodic torus
//! using pre-allocated, transposed unit-stride `rustfft` transforms, optional
//! [`math_commons::ast::CompiledExpr`] formula sandboxes, and a stack-allocated `32x32`
//! radix-2 2D FFT soliton viability probe.
//!
//! ## Mathematical Formulation
//!
//! Given continuous cell activations $A_t(x, y) \in [0, 1]$ on a periodic $W \times H$ torus,
//! a normalized concentric-ring or custom radial kernel $K$ ($\sum_{x, y} K(x, y) = 1$), and
//! growth mapping $G(u) \in [-1, 1]$ (default Gaussian $2 \exp(-(u - \mu)^2 / (2\sigma^2)) - 1$):
//!
//! $$U_t = \mathcal{F}^{-1}\!\bigl(\mathcal{F}(A_t) \odot \hat{K}\bigr), \qquad
//!   A_{t + \Delta t} = \operatorname{clamp}\!\bigl(A_t + \Delta t \, G(U_t),\, 0,\, 1\bigr)$$
//!
//! ## Quick Start
//!
//! ```rust
//! # use domain_biology::__doctest_facade as math_explorer;
//! use math_explorer::biology::lenia::{LeniaEngine, LeniaSpecies};
//! use math_explorer::rng::OxidizeRng;
//!
//! let mut rng = OxidizeRng::new(42);
//! let mut engine = LeniaEngine::from_species(64, 64, LeniaSpecies::Orbium, 0.0, &mut rng);
//!
//! for _ in 0..10 {
//!     engine.step();
//! }
//!
//! assert!(engine.telemetry().total_mass > 0.0);
//! ```

/// Core engine and pre-allocated 2D FFT stepper.
pub mod engine;
/// Zero-allocation `32x32` stack-buffered radix-2 2D FFT soliton viability probe.
pub mod fft_probe;
mod fft_radix2;
/// Concentric-ring kernel shells, Gaussian growth mappings, and compiled AST rule evaluators.
pub mod kernel;
/// Canonical soliton species genomes and deterministic seeding helpers.
pub mod presets;
mod telemetry_math;

#[cfg(test)]
mod tests;

/// Minimum supported kernel interaction radius $R$ in grid cells.
pub const MIN_KERNEL_RADIUS: f64 = 4.0;
/// Maximum supported kernel interaction radius $R$ in grid cells.
pub const MAX_KERNEL_RADIUS: f64 = 32.0;
/// Maximum number of concentric kernel rings supported by [`KernelShells`].
pub const MAX_KERNEL_RINGS: usize = 4;
/// Minimum supported periodic grid dimension along each axis.
pub const MIN_GRID_SIZE: usize = 32;
/// Maximum supported periodic grid dimension along each axis.
pub const MAX_GRID_SIZE: usize = 256;

pub use engine::{LeniaEngine, LeniaTelemetry};
pub use fft_probe::{
    LeniaProbeOutcome, LeniaProbeSpec, LeniaProbeSummary, PROBE_CELLS, PROBE_DOWNSAMPLE,
    PROBE_SIDE, PROBE_STEPS, evaluate_lenia_probe_32, evaluate_lenia_probe_32_preseeded,
    fft_2d_forward_32, fft_2d_inverse_32, precompute_probe_kernel_hat_32, seed_probe_species_32,
};
pub use kernel::{
    CompiledGrowthRule, CompiledKernelRule, GrowthParams, KernelShells, MAX_LENIA_RULE_PARAMS,
    growth_gaussian, kernel_core_bump, write_normalized_torus_kernel,
};
pub use presets::{LeniaGenome, LeniaSpecies};

// [cite:biology]
