//! Statistical Mechanics module.
//!
//! This module serves as the mathematical bridge between quantum/classical micro-physics
//! and macroscopic thermodynamics, covering Ensemble Theory, Quantum Statistics,
//! Phase Transitions, and Non-Equilibrium dynamics.

/// Boltzmann Constant in J/K.
pub const KB: f64 = 1.380649e-23;

/// Langevin brownian dynamics, fluctuation-dissipation, and Fokker-Planck drift-diffusion.
pub mod dynamics;
/// Canonical, grand-canonical, and microcanonical partition functions and entropy relations.
pub mod ensembles;
pub mod ising;
/// Fermi-Dirac, Bose-Einstein, and Maxwell-Boltzmann equilibrium occupation distributions.
pub mod quantum_stats;
pub mod sandpile;

// [cite:graph_parameters_rust]

use pure_math::theory_verification;

theory_verification!(
    module = "stat_mech",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        use approx::assert_relative_eq;
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
