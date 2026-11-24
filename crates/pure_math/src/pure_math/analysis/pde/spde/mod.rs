//! Pre-planned 2D semi-implicit spectral PDE/SPDE solvers and radial structure-factor diagnostics.
//!
//! Pairs arbitrary-dimension 2D `rustfft` semi-implicit Laplacian/biharmonic steppers and circularly
//! averaged radial structure-factor binning (`S(k, t)`) with the `#![no_std]` spatio-temporal noise
//! injectors in [`engine_pde::spde_noise`].
//!
//! # Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use engine_pde::{BoundaryCondition, inject_conserved_flux_noise_2d};
//! use math_commons::math_kernel::types::SolverWorkspace;
//! use math_explorer::pure_math::analysis::pde::spde::{
//!     ReactionBiharmonicParams, SemiImplicitSpectral2D,
//! };
//! use oxidize_core::rng::OxidizeRng;
//!
//! let (w, h) = (16, 16);
//! let mut rng = OxidizeRng::from_seed(42);
//! let mut scratch = vec![0.0_f64; 2 * w * h];
//! let mut conserved = vec![0.0_f64; w * h];
//! let mut ws = SolverWorkspace::new(&mut scratch);
//! inject_conserved_flux_noise_2d(
//!     &mut conserved,
//!     w,
//!     h,
//!     1.0,
//!     1.0,
//!     0.05,
//!     0.025,
//!     BoundaryCondition::Periodic,
//!     &mut ws,
//!     &mut rng,
//! );
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
//! solver.step_reaction_biharmonic(&mut u, &rhs, Some(&conserved), params);
//! assert!(u.iter().all(|v| v.is_finite()));
//! ```

// [cite:spde]

/// Pre-planned 2D `rustfft` semi-implicit Laplacian/biharmonic stepper and radial structure factor.
pub mod spectral_2d;
mod spectral_tables;

pub use spectral_2d::{
    ConservedBiharmonicParams, RadialSpectrumSummary, ReactionBiharmonicParams,
    SemiImplicitSpectral2D,
};

#[cfg(test)]
mod tests;
