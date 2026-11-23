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
//! 9. 2D Cut-and-Project Quasicrystals (Penrose, Ammann-Beenker, Stampfli)
//! 10. 2D Bravais Reciprocal Lattices, Brillouin Zones, and Bragg Diffraction

pub mod band_theory;
pub mod bcs;
pub mod interactions;
#[allow(missing_docs)]
pub mod lattice;
pub mod magnetism;
pub mod phonons;
pub mod quasicrystal;
pub mod reciprocal;
pub mod screening;
pub mod second_quantization;
#[allow(missing_docs)]
pub mod types;

// [cite:mmwave_radiotherapy_setup]
// [cite:solid_state]

use pure_math::theory_verification;

theory_verification!(
    module = "solid_state",
    constants = {
        QUASICRYSTAL_GOLDEN_RATIO = quasicrystal::GOLDEN_RATIO;
        QUASICRYSTAL_SILVER_RATIO = quasicrystal::SILVER_RATIO;
        QUASICRYSTAL_STAMPFLI_RATIO = quasicrystal::STAMPFLI_RATIO;
    },
    test = {
        use approx::assert_relative_eq;
        use std::f64::consts::PI;

        let expected_phi = 0.5 * (1.0 + 5.0_f64.sqrt());
        let expected_silver = 1.0 + 2.0_f64.sqrt();
        let expected_stampfli = 2.0 + 3.0_f64.sqrt();

        assert_relative_eq!(
            QUASICRYSTAL_GOLDEN_RATIO,
            expected_phi,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            QUASICRYSTAL_SILVER_RATIO,
            expected_silver,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            QUASICRYSTAL_STAMPFLI_RATIO,
            expected_stampfli,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let hex = reciprocal::Lattice2DParams::hexagonal(2.5).expect("valid 2D hexagonal lattice");
        let [a1, a2] = hex.direct_basis();
        let [b1, b2] = hex.reciprocal_basis();
        assert_relative_eq!(a1.dot(&b1), 2.0 * PI, epsilon = 1e-12);
        assert_relative_eq!(a2.dot(&b2), 2.0 * PI, epsilon = 1e-12);
        assert_relative_eq!(a1.dot(&b2), 0.0, epsilon = 1e-12);
        assert_relative_eq!(a2.dot(&b1), 0.0, epsilon = 1e-12);

        let bz_area = hex.first_brillouin_zone().area();
        let expected_bz_area = 4.0 * PI * PI / hex.unit_cell_area();
        assert_relative_eq!(bz_area, expected_bz_area, epsilon = 1e-10);
    }
);
