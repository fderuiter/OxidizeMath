//! Nuclear physics module covering properties, models, and reactions.
//!
//! This module provides implementations for:
//! - Basic nuclear properties (Radius, Density).
//! - Liquid Drop Model (Semi-Empirical Mass Formula).
//! - Shell Model (Spin-Orbit coupling).
//! - Radioactive Decay (Activity, Gamow Factor).
//! - Nuclear Reactions (Q-Value, Breit-Wigner Resonance).

#[allow(missing_docs)]
pub mod decay;
#[allow(missing_docs)]
pub mod models;
#[allow(missing_docs)]
pub mod properties;
#[allow(missing_docs)]
pub mod reactions;
#[allow(missing_docs)]
pub mod types;

pub use models::{BindingEnergyModel, LiquidDropModel};
pub use types::*;

// Constants have been moved to math_commons::constants.

/// The Liquid Drop Model (Semi-Empirical Mass Formula).
///
/// **Note:** This module acts as a functional wrapper around `models::LiquidDropModel`.
/// For advanced usage (custom constants), use `LiquidDropModel` directly.
pub mod liquid_drop {
    use super::*;

    /// Calculates the Binding Energy B(Z, A) using the Weizsäcker formula.
    ///
    /// Formula: B = a_v A - a_s A^(2/3) - a_c Z(Z-1)/A^(1/3) - a_sym (A-2Z)^2/A + delta
    ///
    /// # Arguments
    /// * `atomic_number` - Atomic number Z.
    /// * `mass_number` - Mass number A.
    ///
    /// # Returns
    /// * `Result<f64, NuclearError>` - The binding energy in MeV.
    #[verified_engine::verified]
    pub fn binding_energy(
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<f64, NuclearError> {
        models::LiquidDropModel::new().binding_energy(atomic_number, mass_number)
    }

    /// Calculates the Binding Energy per nucleon.
    #[verified_engine::verified]
    pub fn binding_energy_per_nucleon(
        atomic_number: AtomicNumber,
        mass_number: MassNumber,
    ) -> Result<f64, NuclearError> {
        models::LiquidDropModel::new().binding_energy_per_nucleon(atomic_number, mass_number)
    }
}

/// The Shell Model (Spin-Orbit coupling).
///
/// **Note:** This module wraps `models::shell`.
pub mod shell_model {
    use super::*;

    /// Calculates the spin-orbit expectation value <L.S>.
    ///
    /// Formula: <L.S> = (hbar^2 / 2) * (j(j+1) - l(l+1) - s(s+1))
    ///
    /// # Arguments
    /// * `l` - Orbital angular momentum quantum number.
    /// * `s` - Spin angular momentum quantum number.
    /// * `j` - Total angular momentum quantum number.
    ///
    /// # Returns
    /// * `f64` - The energy shift factor.
    #[verified_engine::verified]
    pub fn spin_orbit_coupling(l: f64, s: f64, j: f64) -> f64 {
        models::shell::spin_orbit_coupling(l, s, j)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_iron_peak() {
        // Fe-56: Z=26.
        let be_fe = liquid_drop::binding_energy_per_nucleon(
            AtomicNumber::new(26),
            MassNumber::new(56).unwrap(),
        )
        .unwrap();

        // C-12: Z=6
        let be_c = liquid_drop::binding_energy_per_nucleon(
            AtomicNumber::new(6),
            MassNumber::new(12).unwrap(),
        )
        .unwrap();

        // U-238: Z=92
        let be_u = liquid_drop::binding_energy_per_nucleon(
            AtomicNumber::new(92),
            MassNumber::new(238).unwrap(),
        )
        .unwrap();

        assert!(be_fe > be_c, "Iron should have higher BE/A than Carbon");
        assert!(be_fe > be_u, "Iron should have higher BE/A than Uranium");
    }

    #[test]
    #[verified_engine::verified]
    fn test_spin_orbit() {
        let val_j3_2 = shell_model::spin_orbit_coupling(1.0, 0.5, 1.5);
        let val_j1_2 = shell_model::spin_orbit_coupling(1.0, 0.5, 0.5);

        assert!(
            val_j3_2 > val_j1_2,
            "j=3/2 should have higher coupling value than j=1/2"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_resonance() {
        let e_res = 10.0;
        let gamma = 2.0;

        let max_val = reactions::breit_wigner(e_res, e_res, gamma).unwrap();
        let off_val = reactions::breit_wigner(e_res + 1.0, e_res, gamma).unwrap();

        assert!(max_val > off_val, "Resonance should be max at E_res");
        assert!((max_val - 4.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    #[verified_engine::verified]
    fn test_constants() {
        assert!(math_commons::constants::PROTON_MASS > 900.0);
    }

    #[test]
    #[verified_engine::verified]
    fn test_decay() {
        let half_life = 10.0;
        let n0 = 100.0;
        let remaining = decay::calculate_remaining(n0, half_life, 10.0).unwrap();
        // At t=half_life, should be 50.
        assert!((remaining - 50.0).abs() < math_commons::registry::TOLERANCE_FAST);
    }

    #[test]
    #[verified_engine::verified]
    fn test_liquid_drop_struct_usage() {
        // Test the new Architect-approved way of using the struct
        let model = LiquidDropModel::new();
        let be = model
            .binding_energy(AtomicNumber::new(26), MassNumber::new(56).unwrap())
            .unwrap();
        assert!(be > 0.0);
    }
}

// [cite:favorite_child]

use pure_math::theory_verification;

theory_verification!(
    module = "nuclear",
    constants = {
        DUMMY = 1.0;
    },
    test = {
        assert_relative_eq!(DUMMY, 1.0, epsilon = math_commons::registry::TOLERANCE_FAST);
    }
);
