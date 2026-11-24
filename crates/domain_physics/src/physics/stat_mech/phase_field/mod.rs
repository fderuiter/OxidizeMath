//! Temperature-quenched 2D stochastic phase-field kinetics: non-conserved Allen-Cahn (Model A)
//! and conserved Cahn-Hilliard (Model B) spinodal decomposition with radial structure factor $S(k, t)$ tracking.
//!
//! Complements [`crate::physics::stat_mech::allen_cahn`] and [`crate::physics::stat_mech::cahn_hilliard`]
//! with Eyre convex-splitting [`pure_math::pure_math::analysis::pde::spde::SemiImplicitSpectral2D`]
//! stepping on arbitrary periodic 2D grid dimensions, [`engine_pde::spde_noise`] conserved
//! staggered-face thermal noise, interactive conserved droplet brushing, and circularly averaged
//! $S(k, t)$ coarsening telemetry.
//!
//! # Example
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::stat_mech::phase_field::{
//!     PhaseFieldModel, PhaseFieldParams, PhaseFieldSolver2D,
//! };
//! use oxidize_core::rng::OxidizeRng;
//!
//! let params = PhaseFieldParams {
//!     model: PhaseFieldModel::CahnHilliard,
//!     temperature: 0.25,
//!     critical_temperature: 1.0,
//!     mobility: 1.0,
//!     kappa: 0.5,
//!     noise_sigma: 0.0,
//!     dx: 1.0,
//!     dt: 0.05,
//! };
//! let mut rng = OxidizeRng::from_seed(2026);
//! let mut solver = PhaseFieldSolver2D::new_spinodal(32, 32, 0.0, 0.08, &mut rng, params);
//! let m0 = solver.mean_concentration();
//! let f0 = solver.free_energy();
//! solver.step_n(&mut rng, 10);
//! assert!((solver.mean_concentration() - m0).abs() < 1e-12);
//! assert!(solver.free_energy() < f0);
//! ```

// [cite:phase_field]

/// Semi-implicit 2D stochastic Allen-Cahn and Cahn-Hilliard solver.
pub mod solver;
/// Pre-allocated radial structure factor $S(k, t)$ and coarsening history recorder.
pub mod telemetry;

pub use solver::{
    LIFSHITZ_SLYOZOV_EXPONENT, MODEL_A_COARSENING_EXPONENT, PhaseFieldModel, PhaseFieldParams,
    PhaseFieldSolver2D,
};
pub use telemetry::PhaseFieldTelemetry;

#[cfg(test)]
mod tests;
