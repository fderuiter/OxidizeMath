//! Statistical Mechanics module.
//!
//! This module serves as the mathematical bridge between quantum/classical micro-physics
//! and macroscopic thermodynamics, covering Ensemble Theory, Quantum Statistics,
//! Phase Transitions, and Non-Equilibrium dynamics.

/// Boltzmann Constant in J/K.
pub const KB: f64 = 1.380649e-23;

#[allow(missing_docs)]
pub mod dynamics;
#[allow(missing_docs)]
pub mod ensembles;
/// Zero-allocation 1D and 2D Edwards-Wilkinson and KPZ stochastic interface growth solvers.
pub mod interface_growth;
pub mod ising;
/// Zero-allocation 2D stochastic Allen-Cahn and Cahn-Hilliard phase-field solvers.
pub mod phase_field;
#[allow(missing_docs)]
pub mod quantum_stats;

// [cite:stat_mech]
// [cite:stochastic_pde]

use pure_math::theory_verification;

theory_verification!(
    module = "stat_mech",
    constants = {
        EW_GROWTH_BETA_1D = 0.25;
        KPZ_GROWTH_BETA_1D = 1.0 / 3.0;
        KPZ_ROUGHNESS_ALPHA_1D = 0.5;
        KPZ_DYNAMIC_Z_1D = 1.5;
        LSW_COARSENING_EXPONENT = 1.0 / 3.0;
    },
    test = {
        use crate::physics::stat_mech::interface_growth::{
            EW_BETA_1D, InterfaceGrowthKind, KPZ_ALPHA_1D, KPZ_BETA_1D, KPZ_DYNAMIC_Z_1D as KPZ_Z,
        };
        use crate::physics::stat_mech::phase_field::{
            LIFSHITZ_SLYOZOV_EXPONENT, PhaseFieldModel, PhaseFieldParams, PhaseFieldSolver2D,
        };
        use approx::assert_relative_eq;
        use oxidize_core::rng::OxidizeRng;

        // Family-Vicsek and KPZ Galilean invariance scaling laws in 1+1D:
        assert_relative_eq!(
            EW_GROWTH_BETA_1D,
            EW_BETA_1D,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            KPZ_GROWTH_BETA_1D,
            KPZ_BETA_1D,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            KPZ_ROUGHNESS_ALPHA_1D + KPZ_DYNAMIC_Z_1D,
            2.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            KPZ_ALPHA_1D / KPZ_Z,
            KPZ_BETA_1D,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        let (alpha_ew, beta_ew, z_ew) = InterfaceGrowthKind::EdwardsWilkinson.scaling_exponents(1);
        assert_relative_eq!(
            beta_ew,
            alpha_ew / z_ew,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            LSW_COARSENING_EXPONENT,
            LIFSHITZ_SLYOZOV_EXPONENT,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        // Model B Cahn-Hilliard conserved concentration and Lyapunov free-energy decay:
        let mut rng = OxidizeRng::from_seed(0x57A7_4EC0);
        let params = PhaseFieldParams {
            model: PhaseFieldModel::CahnHilliard,
            noise_sigma: 0.0,
            ..PhaseFieldParams::default()
        };
        let mut ch = PhaseFieldSolver2D::new_spinodal(16, 16, 0.05, 0.1, &mut rng, params);
        let m0 = ch.mean_concentration();
        let f0 = ch.free_energy();
        ch.step_n(&mut rng, 5);
        assert!((ch.mean_concentration() - m0).abs() < 1e-12);
        assert!(ch.free_energy() < f0);
    }
);
