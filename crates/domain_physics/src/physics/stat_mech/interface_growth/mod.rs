//! Stochastic surface roughening and interface growth solvers (Edwards-Wilkinson and KPZ).
//!
//! Implements zero-allocation 1D and 2D periodic lattice solvers using
//! [`pure_math::pure_math::analysis::pde::fused_stepper::StencilOperators`] and Lam-Shin
//! conservative nonlinear gradient-squared discretizations alongside pre-allocated
//! Family-Vicsek dynamic scaling telemetry.
//!
//! # Example
//!
//! ```rust
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::stat_mech::interface_growth::{
//!     InterfaceGrowth1D, InterfaceGrowthParams, InterfaceScalingTelemetry,
//! };
//! use oxidize_core::rng::OxidizeRng;
//!
//! let params = InterfaceGrowthParams {
//!     nu: 0.5,
//!     lambda: 3.0,
//!     noise_sigma: 1.0,
//!     dx: 1.0,
//!     dt: 0.02,
//! };
//! let mut solver = InterfaceGrowth1D::new(128, params);
//! let mut rng = OxidizeRng::from_seed(12345);
//! let mut telemetry = InterfaceScalingTelemetry::new(64);
//!
//! solver.step_n(&mut rng, 20);
//! telemetry.record_kpz_1d(solver.time(), solver.interface_width(), solver.len() as f64);
//! assert!(solver.interface_width() > 0.0);
//! assert_eq!(telemetry.len(), 1);
//! ```

// [cite:interface_growth]

/// Zero-allocation 1D periodic Edwards-Wilkinson and KPZ solver.
pub mod kpz_1d;
/// Zero-allocation 2D periodic Edwards-Wilkinson and KPZ heightfield solver.
pub mod kpz_2d;
/// Pre-allocated interface width $W(L, t)$, Family-Vicsek collapse, and $\beta$-exponent estimator.
pub mod scaling;

pub use kpz_1d::{InterfaceGrowth1D, InterfaceGrowthParams};
pub use kpz_2d::InterfaceGrowth2D;
pub use scaling::{
    EW_ALPHA_1D, EW_BETA_1D, EW_DYNAMIC_Z_1D, InterfaceGrowthKind, InterfaceScalingTelemetry,
    KPZ_ALPHA_1D, KPZ_BETA_1D, KPZ_DYNAMIC_Z_1D,
};

#[cfg(test)]
mod tests;
