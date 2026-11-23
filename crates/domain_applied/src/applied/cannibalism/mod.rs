/// Zero-allocation characteristic-shift McKendrick-von Foerster PDE solver.
pub mod age_structured;
/// Diekmann et al. natural and cannibalistic per-capita mortality rate operator.
pub mod death_rate;
/// Coupled van den Bosch juvenile age-transport and adult maturation dynamics.
pub mod juvenile_adult_dynamics;
/// McKendrick-von Foerster characteristic, upwind Eulerian, and renewal operators.
pub mod mckendrick_von_foerster;
/// Stage-structured *Tribolium* Larvae-Pupae-Adults (LPA) cannibalism map.
pub mod tribolium;
/// Guarded Holling Type I/II two-dimensional normal-cannibal ODE system.
pub mod two_dimensional_ode;

pub use age_structured::*;
pub use death_rate::*;
pub use juvenile_adult_dynamics::*;
pub use mckendrick_von_foerster::*;
pub use tribolium::*;
pub use two_dimensional_ode::*;

use pure_math::theory_verification;

theory_verification!(
    module = "cannibalism",
    constants = {
        BETA_N = 1.0;
        BETA_C = 0.5;
        K_N = 0.1;
        PHI = 0.05;
        MU_N = 0.2;
        MU_C = 0.3;
    },
    test = {
        use approx::assert_relative_eq;
        let model = CannibalismModel::new(BETA_N, BETA_C, K_N, PHI, MU_N, MU_C);
        assert_relative_eq!(
            model.beta_n,
            BETA_N,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            model.beta_c,
            BETA_C,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            model.k_n,
            K_N,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            model.phi_n_c,
            PHI,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            model.mu_n,
            MU_N,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        assert_relative_eq!(
            model.mu_c,
            MU_C,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
    }
);
// [cite:cannibalism]
