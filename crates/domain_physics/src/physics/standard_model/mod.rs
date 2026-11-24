//! # Standard Model of Particle Physics
//!
//! This module implements the mathematical formalism of the Standard Model,
//! covering the Gauge Principle, Spontaneous Symmetry Breaking, Flavor Physics,
//! Quantum Chromodynamics (QCD), and Neutrino Oscillations.
//!
//! The Standard Model is a quantum field theory based on the gauge group
//! SU(3)_C x SU(2)_L x U(1)_Y, describing the strong, weak, and electromagnetic
//! interactions.

pub mod flavor;
pub mod gauge;
pub mod higgs;
pub mod neutrinos;
pub mod qcd;

pub use gauge::GaugeCouplings;
pub use higgs::{HiggsPotential, boson_masses};
pub use neutrinos::{
    NeutrinoFlavor, PmnsParameters, construct_pmns, matter_potential_ev2,
    msw_oscillation_probabilities, pmns_oscillation_probabilities,
};

// [cite:standard_model]

use pure_math::theory_verification;

theory_verification!(
    module = "standard_model",
    constants = {
        HIGGS_VEV_GEV = 246.22;
        WEAK_G1 = 0.357;
        WEAK_G2 = 0.652;
    },
    test = {
        use approx::assert_relative_eq;
        let couplings = gauge::GaugeCouplings::new(WEAK_G1, WEAK_G2, 1.22);
        let (cos_theta_w, sin_theta_w) = couplings.weak_mixing_angle();
        let (m_w, m_z) = higgs::boson_masses(HIGGS_VEV_GEV, &couplings);
        assert_relative_eq!(
            cos_theta_w * cos_theta_w + sin_theta_w * sin_theta_w,
            1.0,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
        assert_relative_eq!(
            m_w / m_z,
            cos_theta_w,
            epsilon = math_commons::registry::TOLERANCE_HIGH
        );
    }
);
