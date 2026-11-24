//! Stochastic surface roughening, Lam-Shin conservative KPZ stepping, and Family-Vicsek scaling telemetry.
//!
//! Complements [`crate::physics::stat_mech::edwards_wilkinson`] and [`crate::physics::stat_mech::kpz`]
//! with arbitrary-size periodic 1D and 2D Lam-Shin conservative $(\nabla h)^2$ + tamed Euler-Maruyama
//! solvers built on [`engine_pde::StencilOperators`] and [`engine_pde::spde_noise`], paired with
//! pre-allocated Family-Vicsek dynamic scaling collapse and log-log $\beta$ slope regression.
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

/// Zero-allocation 1D periodic Edwards-Wilkinson and Lam-Shin tamed KPZ solver.
pub mod kpz_1d;
/// Zero-allocation 2D periodic Edwards-Wilkinson and Lam-Shin tamed KPZ heightfield solver.
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
