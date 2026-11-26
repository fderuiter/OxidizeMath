//! Age-structured McKendrick-von Foerster PDE, Diekmann mortality kernels,
//! van den Bosch juvenile-adult delay dynamics, and 2D ODE cannibalism models.

pub mod death_rate;
pub mod juvenile_adult_dynamics;
pub mod mckendrick_von_foerster;
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

        let kernel = CannibalismKernel::new(0.0, 0.0, 0.0, 2.0, 0.1, 4.0, 0.0, 50.0);
        assert_relative_eq!(
            kernel.ricker_saturation(0.0),
            1.0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );
        let mut mvf = McKendrickSolver::new(33, 8.0, kernel);
        mvf.set_initial_density(|a| if (1.0..=3.0).contains(&a) { 10.0 } else { 0.0 });
        let m0 = mvf.total_population();
        mvf.step_upwind_with_boundary(mvf.da(), 0.0);
        assert_relative_eq!(
            mvf.total_population(),
            m0,
            epsilon = math_commons::registry::TOLERANCE_FAST
        );

        let mut ja = JuvenileAdultSolver::new(33, 12.0, JuvenileAdultParams::default());
        ja.set_initial_state(12.0, |a| (-0.5 * a).exp() * 8.0);
        ja.step_upwind(0.25);
        assert!(ja.adults().is_finite() && ja.adults() >= 0.0);
        assert!(ja.total_juveniles().is_finite() && ja.total_juveniles() >= 0.0);
    }
);
// [cite:cannibalism]
