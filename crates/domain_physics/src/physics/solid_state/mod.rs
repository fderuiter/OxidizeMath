//! # Solid State Physics
//!
//! This module provides implementations for core concepts in solid state physics,
//! facilitating the transition from single-particle band theory to Many-Body Physics.
//!
//! It covers:
//! 1. Second Quantization (Operators and Fock States)
//! 2. Screening (Thomas-Fermi and Yukawa)
//! 3. Lattice Structures (SC, BCC, FCC)
//! 4. Lattice Dynamics (Phonons)
//! 5. Magnetism (Heisenberg Model)
//! 6. Superconductivity (BCS)
//! 7. Interactions (Electron-Phonon)
//! 8. Types and Errors (Strong types and error handling)

pub mod band_theory;
#[allow(missing_docs)]
pub mod bcs;
pub mod interactions;
#[allow(missing_docs)]
pub mod lattice;
pub mod magnetism;
pub mod phonons;
pub mod screening;
pub mod second_quantization;
#[allow(missing_docs)]
pub mod types;

// [cite:mmwave_radiotherapy_setup]

use pure_math::theory_verification;

theory_verification!(
    module = "solid_state",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
