//! # Solid State Physics
//!
//! This module provides verified implementations for periodic and aperiodic condensed-matter
//! physics, spanning crystallography, reciprocal-space Bragg diffraction, quasiperiodic
//! Hamiltonians, lattice dynamics, and many-body BCS superconductivity:
//! 1. Second Quantization (Fermionic/Bosonic operators and Fock states)
//! 2. Screening (Thomas-Fermi dielectric function and Yukawa potential)
//! 3. Lattice Structures (SC, BCC, and FCC Bravais unit cells)
//! 4. Cut-and-Project Quasicrystals (1D Fibonacci, 5/10-fold Penrose, 8-fold Ammann-Beenker, 12-fold Stampfli, and phason flips)
//! 5. Reciprocal Space & Bragg Diffraction (Reciprocal bases, Brillouin zones, and zero-allocation $S(\mathbf{q})$ solvers)
//! 6. Band Theory & Quasiperiodic Chains (Bloch dispersions, Fibonacci Cantor spectra, and Aubry-André localization)
//! 7. Lattice Dynamics (Debye acoustic, Einstein optical phonons, and low-$T$ $C_V \propto T^3$)
//! 8. Magnetism (Heisenberg exchange) & Interactions (Fröhlich electron-phonon coupling)
//! 9. Superconductivity (Self-consistent BCS gap equations and Bogoliubov coherence factors)

pub mod band_theory;
pub mod bcs;
pub mod interactions;
/// Bravais crystal lattices (SC, BCC, FCC), coordination numbers, and packing fractions.
pub mod lattice;
pub mod magnetism;
pub mod phonons;
pub mod quasicrystal;
pub mod quasiperiodic;
pub mod reciprocal;
pub mod screening;
pub mod second_quantization;
/// Strongly typed energy (`ElectronVolts`) and temperature (`Kelvin`) unit wrappers.
pub mod types;

// [cite:solid_state]

use pure_math::theory_verification;

theory_verification!(
    module = "solid_state",
    constants = {
        TAU_FIBONACCI_RESIDUAL = quasicrystal::GOLDEN_RATIO * quasicrystal::GOLDEN_RATIO
            - quasicrystal::GOLDEN_RATIO
            - 1.0;
        RECIPROCAL_ORTHOGONALITY_TARGET = std::f64::consts::TAU;
        BCC_ALLOWED_REFLECTION_WEIGHT = 4.0;
        BOGOLIUBOV_NORMALIZATION = 1.0;
    },
    test = {
        use crate::physics::solid_state::bcs::coherence_factors_strong;
        use crate::physics::solid_state::lattice::{BodyCenteredCubic, CrystalSystem};
        use crate::physics::solid_state::reciprocal::{
            reciprocal_basis_3d, unit_cell_structure_factor,
        };
        use crate::physics::solid_state::types::ElectronVolts;
        use approx::assert_relative_eq;

        assert_relative_eq!(
            TAU_FIBONACCI_RESIDUAL,
            0.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );

        let bcc = BodyCenteredCubic.generate(2.5);
        let recip = reciprocal_basis_3d(&bcc.lattice_vectors).expect("cubic reciprocal basis");
        assert_relative_eq!(
            bcc.lattice_vectors[0].dot(&recip[0]),
            RECIPROCAL_ORTHOGONALITY_TARGET,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
        assert_relative_eq!(
            bcc.lattice_vectors[0].dot(&recip[1]),
            0.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
        assert_relative_eq!(
            unit_cell_structure_factor(&bcc, [1, 0, 0]),
            0.0,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );
        assert_relative_eq!(
            unit_cell_structure_factor(&bcc, [1, 1, 0]),
            BCC_ALLOWED_REFLECTION_WEIGHT,
            epsilon = math_commons::registry::TOLERANCE_STANDARD
        );

        let (u_k, v_k) =
            coherence_factors_strong(ElectronVolts::new(0.6), ElectronVolts::new(0.25))
                .expect("valid coherence factors");
        assert_relative_eq!(
            u_k * u_k + v_k * v_k,
            BOGOLIUBOV_NORMALIZATION,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
    }
);

#[cfg(test)]
mod tests;
