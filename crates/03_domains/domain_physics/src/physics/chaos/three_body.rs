//! Planar 3-Body Figure-8 Choreography (`ThreeBodyChoreography`) re-exported in `physics::chaos`.
//!
//! Bridges the canonical 12D planar 3-body gravitational choreography from
//! [`crate::physics::astrophysics::nbody`] into the conservative chaos suite without duplicating
//! force-evaluation kernels.
//!
//! # Example
//!
//! ```
//! # use domain_physics::__doctest_facade as math_explorer;
//! use math_explorer::physics::chaos::ThreeBodyChoreography;
//! use math_explorer::pure_math::analysis::ode::{ArrayState, HamiltonianSystem, Yoshida4};
//!
//! let choreography = ThreeBodyChoreography::default();
//! let mut state = ThreeBodyChoreography::perturbed_figure_eight(0.005, 0.002);
//! let e0 = choreography.hamiltonian(&state);
//! let l0 = choreography.angular_momentum_z(&state);
//! let mut solver = Yoshida4::new(&ArrayState::<6>::zeros());
//! for _ in 0..500 {
//!     solver.step_separable(&choreography, &mut state, 0.01);
//! }
//! assert!(choreography.relative_energy_error(&state, e0) < 1e-7);
//! let p_tot = choreography.total_linear_momentum(&state);
//! assert!(p_tot[0].hypot(p_tot[1]) < 1e-12);
//! assert!((choreography.angular_momentum_z(&state) - l0).abs() < 1e-11);
//! ```

// [cite:chaos_theory]

pub use crate::physics::astrophysics::nbody::{
    FIGURE_EIGHT_PERIOD, FigureEightThreeBody, FigureEightThreeBody as ThreeBodyChoreography,
};
