#![allow(missing_docs)]
#[cfg(test)]
mod tests {
    use domain_applied::applied::cannibalism::two_dimensional_ode::CannibalismModel;
    use nalgebra::Vector2;
    use pure_math::pure_math::analysis::ode::OdeSystem;

    #[test]
    #[verified_engine::verified]
    fn test_struct_implementation() {
        let n = 100.0;
        let c = 10.0;
        let beta_n = 0.5;
        let beta_c = 0.2;
        let k_n = 0.1;
        let phi_n_c = 5.0;
        let mu_n = 0.3;
        let mu_c = 0.4;

        let model = CannibalismModel::new(beta_n, beta_c, k_n, phi_n_c, mu_n, mu_c);
        let state = Vector2::new(n, c);

        // Time t is unused in this autonomous system
        let derivative = model.derivative(0.0, &state);

        let expected_dn = beta_n * n + beta_c * c - k_n * n - phi_n_c - mu_n * n;
        let expected_dc = k_n * n - mu_c * c;

        assert!(
            (derivative[0] - expected_dn).abs() < math_commons::registry::TOLERANCE_HIGH,
            "Model derivative dN/dt should match expected"
        );
        assert!(
            (derivative[1] - expected_dc).abs() < math_commons::registry::TOLERANCE_HIGH,
            "Model derivative dC/dt should match expected"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_juvenile_adult_hopf_cohort_oscillations_vs_subcritical_damping() {
        use domain_applied::applied::cannibalism::{JuvenileAdultParams, JuvenileAdultSolver};

        // Supercritical adult-on-juvenile cannibalism triggers sustained dwarf-and-giant cohort waves
        let hopf_params = JuvenileAdultParams::default();
        let mut hopf = JuvenileAdultSolver::new(81, 16.0, hopf_params);
        hopf.set_initial_state(16.0, |a| if a < 1.0 { 15.0 } else { 1.0 });

        // Burn-in past initial transient (t = 0..60), then measure oscillation amplitude on t = 60..120
        for _ in 0..600 {
            hopf.step_upwind(0.1);
        }
        let mut min_a = f64::INFINITY;
        let mut max_a = 0.0_f64;
        for _ in 0..600 {
            hopf.step_upwind(0.1);
            min_a = min_a.min(hopf.adults());
            max_a = max_a.max(hopf.adults());
        }

        // Subcritical zero-cannibalism regime converges to a stationary equilibrium
        let calm_params = JuvenileAdultParams {
            attack_rate: 0.0,
            cannibal_fecundity_gain: 0.0,
            adult_fecundity: 4.5,
            adult_mortality: 0.6,
            ..hopf_params
        };
        let mut calm = JuvenileAdultSolver::new(81, 20.0, calm_params);
        calm.set_initial_state(20.0, |a| if a < 1.0 { 15.0 } else { 1.0 });
        for _ in 0..600 {
            calm.step_upwind(0.1);
        }
        let mut calm_min = f64::INFINITY;
        let mut calm_max = 0.0_f64;
        for _ in 0..600 {
            calm.step_upwind(0.1);
            calm_min = calm_min.min(calm.adults());
            calm_max = calm_max.max(calm.adults());
        }

        assert!(
            max_a - min_a > 5.0 * (calm_max - calm_min) && max_a - min_a > 2.0,
            "Expected supercritical Hopf oscillations (amp={:.3}) >> calm regime (amp={:.3})",
            max_a - min_a,
            calm_max - calm_min
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_cannibalism_model_rk4_stepping_clamps_non_negative() {
        let model = CannibalismModel::new(0.4, 0.15, 0.08, 2.0, 0.25, 0.3);
        let mut state = Vector2::new(40.0, 8.0);
        for step in 0..100 {
            state = model.step_rk4(step as f64 * 0.1, &state, 0.1);
            assert!(state[0].is_finite() && state[0] >= 0.0);
            assert!(state[1].is_finite() && state[1] >= 0.0);
        }
        let depleted = model.step_rk4(0.0, &Vector2::new(0.0, 0.0), 0.5);
        assert_eq!(depleted, Vector2::new(0.0, 0.0));
    }
}
