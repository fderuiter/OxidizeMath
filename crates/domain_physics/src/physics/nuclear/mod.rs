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

pub use decay::{BatemanChain, MAX_BATEMAN_CHAIN_LEN, bateman_two_step};
pub use models::{BindingEnergyModel, LiquidDropBreakdown, LiquidDropModel};
pub use types::*;

// Constants have been moved to math_commons::constants.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_iron_peak() {
        let model = LiquidDropModel::new();

        // Fe-56: Z=26.
        let be_fe = model
            .binding_energy_per_nucleon(AtomicNumber::new(26), MassNumber::new(56).unwrap())
            .unwrap();

        // C-12: Z=6
        let be_c = model
            .binding_energy_per_nucleon(AtomicNumber::new(6), MassNumber::new(12).unwrap())
            .unwrap();

        // U-238: Z=92
        let be_u = model
            .binding_energy_per_nucleon(AtomicNumber::new(92), MassNumber::new(238).unwrap())
            .unwrap();

        assert!(be_fe > be_c, "Iron should have higher BE/A than Carbon");
        assert!(be_fe > be_u, "Iron should have higher BE/A than Uranium");
    }

    #[test]
    #[verified_engine::verified]
    fn test_spin_orbit() {
        let val_j3_2 = models::shell::spin_orbit_coupling(1.0, 0.5, 1.5);
        let val_j1_2 = models::shell::spin_orbit_coupling(1.0, 0.5, 0.5);

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
        let model = LiquidDropModel::new();
        let breakdown = model
            .binding_energy_breakdown(AtomicNumber::new(26), MassNumber::new(56).unwrap())
            .unwrap();
        let be = model
            .binding_energy(AtomicNumber::new(26), MassNumber::new(56).unwrap())
            .unwrap();
        assert!(be > 0.0);
        assert!(breakdown.volume > 0.0);
        assert!(breakdown.surface < 0.0);
        assert!(breakdown.coulomb < 0.0);
        assert!(breakdown.asymmetry < 0.0);
        assert!(breakdown.pairing > 0.0);
        let sum = breakdown.volume
            + breakdown.surface
            + breakdown.coulomb
            + breakdown.asymmetry
            + breakdown.pairing;
        assert!((sum - be).abs() < math_commons::registry::TOLERANCE_HIGH);
        assert!((breakdown.per_nucleon - be / 56.0).abs() < math_commons::registry::TOLERANCE_HIGH);
    }

    #[test]
    #[verified_engine::verified]
    fn test_bateman_degenerate_and_secular_limits() {
        let t_half = 5.0;
        let lam = std::f64::consts::LN_2 / t_half;
        let t = 7.5;
        let (n1, n2) = decay::bateman_two_step(100.0, t_half, Some(t_half), t).unwrap();
        let expected_n2 = 100.0 * lam * t * (-lam * t).exp();
        assert!((n1 - 100.0 * (-lam * t).exp()).abs() < 1e-10);
        assert!((n2 - expected_n2).abs() < 1e-10);

        // 3-step equal-rate chain leading to a stable endpoint conserves total nuclei = 100.
        let chain =
            BatemanChain::from_half_lives([Some(t_half), Some(t_half), None], [100.0, 0.0, 0.0])
                .unwrap();
        let pops_short = chain.populations_at(0.5).unwrap();
        let pops_long = chain.populations_at(25.0).unwrap();
        let sum_short = pops_short[0] + pops_short[1] + pops_short[2];
        let sum_long = pops_long[0] + pops_long[1] + pops_long[2];
        assert!((sum_short - 100.0).abs() < 1e-10);
        assert!((sum_long - 100.0).abs() < 1e-10);
        assert!((pops_long[1] - 100.0 * lam * 25.0 * (-lam * 25.0).exp()).abs() < 1e-9);

        // Secular equilibrium: T_parent >> T_daughter => A_daughter -> A_parent.
        let secular =
            BatemanChain::from_half_lives([Some(1.0e9), Some(10.0), None], [1.0e12, 0.0, 0.0])
                .unwrap();
        let acts = secular.activities_at(500.0).unwrap();
        assert!(((acts[1] - acts[0]) / acts[0]).abs() < 1e-6);
    }
}

// [cite:nuclear]

use pure_math::theory_verification;

theory_verification!(
    module = "nuclear",
    constants = {
        R0_FM = 1.2;
        CARBON_12_A = 12.0;
        NUCLEAR_SATURATION_DENSITY = 0.13815533254504805;
    },
    test = {
        use approx::assert_relative_eq;
        let a12 = MassNumber::new(12).unwrap();
        let radius = properties::calculate_radius(a12);
        let density = properties::calculate_nucleon_density(a12).unwrap();
        assert_relative_eq!(
            radius,
            R0_FM * CARBON_12_A.cbrt(),
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            density,
            NUCLEAR_SATURATION_DENSITY,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
