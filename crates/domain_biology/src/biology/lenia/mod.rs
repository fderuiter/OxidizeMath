//! # Continuous Cellular Automata (Lenia)
//!
//! Implements spectral 2D **Lenia** continuous cellular automata on a periodic torus
//! using pre-allocated, transposed unit-stride `rustfft` transforms.
//!
//! ## Mathematical Formulation
//!
//! Given continuous cell activations $A_t(x, y) \in [0, 1]$ on a periodic $W \times H$ torus,
//! a normalized concentric-ring kernel $K$ ($\sum_{x, y} K(x, y) = 1$), and Gaussian growth
//! mapping $G(u; \mu, \sigma) = 2 \exp\!\left(-\frac{(u - \mu)^2}{2\sigma^2}\right) - 1$:
//!
//! $$U_t = \mathcal{F}^{-1}\!\bigl(\mathcal{F}(A_t) \odot \hat{K}\bigr), \qquad
//!   A_{t + \Delta t} = \operatorname{clamp}\!\bigl(A_t + \Delta t \, G(U_t; \mu, \sigma),\, 0,\, 1\bigr)$$
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
/// Concentric-ring kernel shells and Gaussian growth mappings.
pub mod kernel;
/// Canonical soliton species genomes and deterministic seeding helpers.
pub mod presets;

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
pub use kernel::{GrowthParams, KernelShells, growth_gaussian, kernel_core_bump};
pub use presets::{LeniaGenome, LeniaSpecies};

// [cite:biology]
