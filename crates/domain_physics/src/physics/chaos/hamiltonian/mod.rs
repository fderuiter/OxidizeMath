//! Zero-allocation Symplectic Hamiltonian Chaos Kernels (`domain_physics::physics::chaos::hamiltonian`).
//!
//! Consolidates conservative Hamiltonian flows and symplectic maps backed by fixed-size
//! [`ArrayState`] buffers:
//! - [`DoublePendulum`]: Non-separable 2-DOF canonical planar double pendulum integrated via
//!   fixed-iteration [`ImplicitMidpoint`] or [`GaussLegendre4`].
//! - [`HenonHeiles`]: Separable 2-DOF galactic potential integrated via 4th-order [`Yoshida4`].
//! - [`ChirikovStandardMap`]: Exact area-preserving 2D kicked-rotor map exhibiting KAM torus
//!   breakdown above Greene's critical parameter [`CRITICAL_K_GREENE`].
//! - [`FigureEightThreeBody`]: Planar 3-body Chenciner-Montgomery periodic figure-8 choreography.
//!
//! # Example
//!
//! ```
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::chaos::hamiltonian::{
//!     ChirikovStandardMap, DoublePendulum, HamiltonianSystem, ImplicitMidpoint, Solver,
//! };
//!
//! let pendulum = DoublePendulum::default();
//! let mut state = DoublePendulum::default_chaotic_state();
//! let e0 = pendulum.hamiltonian(&state);
//! let mut midpoint = ImplicitMidpoint::new(&state);
//! for step in 0..500 {
//!     midpoint.step(&pendulum, step as f64 * 0.01, &mut state, 0.01);
//! }
//! assert!(pendulum.relative_energy_error(&state, e0) < 5e-3);
//!
//! let map = ChirikovStandardMap::default();
//! let mut torus_pt = ChirikovStandardMap::seed_on_torus(1.0, 2.0).unwrap();
//! map.step_in_place(&mut torus_pt);
//! assert!((map.jacobian_determinant(torus_pt.0[0]) - 1.0).abs() < 1e-15);
//! ```

// [cite:chaos_theory]

pub mod chirikov;
pub mod fan;

pub use super::double_pendulum::DoublePendulum;
pub use super::henon_heiles::HenonHeiles;
pub use super::poincare::{CrossingDirection, PoincareCrossing, PoincareSection};
pub use crate::physics::astrophysics::nbody::{FIGURE_EIGHT_PERIOD, FigureEightThreeBody};
pub use chirikov::{CRITICAL_K_GREENE, ChirikovStandardMap};
pub use fan::{perturb_figure_eight_momentum, seed_double_pendulum_fan};
pub use pure_math::pure_math::analysis::ode::{
    ArrayState, GaussLegendre4, HamiltonianSystem, ImplicitMidpoint, RungeKutta4,
    SeparableHamiltonian, Solver, StormerVerlet, SymplecticMethod, Yoshida4,
};
