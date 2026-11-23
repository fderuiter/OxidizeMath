#[allow(missing_docs)]
pub mod death_rate;
#[allow(missing_docs)]
pub mod juvenile_adult_dynamics;
#[allow(missing_docs)]
pub mod mckendrick_von_foerster;
#[allow(missing_docs)]
pub mod two_dimensional_ode;

pub use death_rate::*;
pub use juvenile_adult_dynamics::*;
pub use mckendrick_von_foerster::*;
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
