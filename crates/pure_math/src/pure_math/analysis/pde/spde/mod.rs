//! Zero-allocation stochastic partial differential equation (SPDE) building blocks.
//!
//! Provides pre-allocated Gaussian and staggered conservative-flux ($\nabla \cdot \boldsymbol{\xi}$)
//! spatial noise buffers backed by [`oxidize_core::rng::OxidizeRng`], alongside pre-planned 2D
//! `rustfft` semi-implicit solvers for stiff Laplacian and biharmonic operators.
//!
//! # Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::pure_math::analysis::pde::spde::{
//!     ReactionBiharmonicParams, SemiImplicitSpectral2D, SpatialNoiseBuffer,
//! };
//! use oxidize_core::rng::OxidizeRng;
//!
//! let (w, h) = (16, 16);
//! let mut rng = OxidizeRng::from_seed(42);
//! let mut noise = SpatialNoiseBuffer::new_2d(w, h);
//! let conserved = noise.fill_conserved_flux_2d_periodic(&mut rng, w, h, 1.0, 1.0, 0.05);
//! assert!(conserved.iter().sum::<f64>().abs() < 1e-13);
//!
//! let mut solver = SemiImplicitSpectral2D::new(w, h, 1.0, 1.0);
//! let mut u = vec![0.1_f64; w * h];
//! let rhs = vec![0.0_f64; w * h];
//! let params = ReactionBiharmonicParams {
//!     dt: 0.05,
//!     alpha_0: 0.0,
//!     alpha_2: 1.0,
//!     alpha_4: 0.25,
//! };
//! solver.step_reaction_biharmonic(&mut u, &rhs, Some(conserved), params);
//! assert!(u.iter().all(|v| v.is_finite()));
//! ```

// [cite:spde]

/// Pre-allocated non-conserved Gaussian and conservative staggered-flux noise buffers.
pub mod noise;
/// Pre-planned 2D `rustfft` semi-implicit Laplacian/biharmonic stepper and radial structure factor.
pub mod spectral_2d;
mod spectral_tables;

pub use noise::SpatialNoiseBuffer;
pub use spectral_2d::{
    ConservedBiharmonicParams, RadialSpectrumSummary, ReactionBiharmonicParams,
    SemiImplicitSpectral2D,
};

#[cfg(test)]
mod tests;
