//! Integration tests for guarded Holling ODE positivity, delay oscillations, and zero-allocation PDE stepping.
use domain_applied::applied::cannibalism::{
    AgeStructuredConfig, AgeStructuredSolver, CannibalismModel, JuvenileAdultParams,
    JuvenileAdultSolver, TriboliumLpaModel,
};
use nalgebra::Vector2;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};
use std::sync::Mutex;
use verified_engine::allocator::VerifiedAllocator;
use verified_engine::engine::VerifiedEngine;

#[global_allocator]
static ALLOCATOR: VerifiedAllocator = VerifiedAllocator;

static VERIFIED_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn test_struct_implementation_and_holling_predation() {
    let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n = 100.0;
    let c = 10.0;
    let beta_n = 0.5;
    let beta_c = 0.2;
    let k_n = 0.1;
    let phi_n_c = 0.05;
    let mu_n = 0.3;
    let mu_c = 0.4;

    let model = CannibalismModel::new(beta_n, beta_c, k_n, phi_n_c, mu_n, mu_c);
    let state = Vector2::new(n, c);
    let derivative = model.derivative(0.0, &state);

    let expected_dn = beta_n * n + beta_c * c - k_n * n - phi_n_c * n * c - mu_n * n;
    let expected_dc = k_n * n - mu_c * c;

    assert!(
        (derivative[0] - expected_dn).abs() < math_commons::registry::TOLERANCE_HIGH,
        "Holling Type I derivative dN/dt should match expected"
    );
    assert!(
        (derivative[1] - expected_dc).abs() < math_commons::registry::TOLERANCE_HIGH,
        "Model derivative dC/dt should match expected"
    );

    let holling_ii =
        CannibalismModel::with_handling_time(beta_n, beta_c, k_n, phi_n_c, mu_n, mu_c, 0.02);
    let d_ii = holling_ii.derivative(0.0, &state);
    let expected_pred = phi_n_c * n * c / (1.0 + 0.02 * n);
    let expected_dn_ii = beta_n * n + beta_c * c - k_n * n - expected_pred - mu_n * n;
    assert!(
        (d_ii[0] - expected_dn_ii).abs() < math_commons::registry::TOLERANCE_HIGH,
        "Holling Type II derivative dN/dt should match expected"
    );
}

#[test]
fn test_ode_positivity_and_zero_heap_allocation() {
    let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // Parameters that drove N(t) to -146.70 by t = 15 under unguarded -phi_n_c subtraction
    let ode_model = CannibalismModel::with_handling_time(0.2, 0.05, 0.15, 12.0, 0.1, 0.2, 0.1);
    let mut rk4 = RungeKutta4::new(&Vector2::new(2.0_f64, 5.0_f64));

    let mut hopf_solver = AgeStructuredSolver::new(AgeStructuredConfig::diekmann_hopf());
    let lpa = TriboliumLpaModel::two_cycle_preset(0.7);

    let ((final_ode, final_adult, final_lpa_a), _metrics) = VerifiedEngine::run_verified(|| {
        let mut y = Vector2::new(2.0_f64, 5.0_f64);
        let mut t = 0.0_f64;
        let dt = 0.02_f64;
        for _ in 0..750 {
            rk4.step(&ode_model, t, &mut y, dt);
            t += dt;
            assert!(
                y[0] >= 0.0 && y[1] >= 0.0 && y[0].is_finite() && y[1].is_finite(),
                "Guarded Holling ODE must keep N(t), C(t) >= 0, got {y:?} at t={t:.2}"
            );
        }

        hopf_solver.reconfigure(AgeStructuredConfig::diekmann_hopf());
        for _ in 0..400 {
            hopf_solver.step();
        }

        let mut lpa_state = [1.5, 0.5, 1.0];
        for _ in 0..200 {
            lpa_state = lpa.step(lpa_state);
        }
        (y, hopf_solver.adult_population(), lpa_state[2])
    });

    assert!(final_ode[0] >= 0.0 && final_ode[1] >= 0.0);
    assert!(final_adult > 0.0 && final_adult.is_finite());
    assert!(final_lpa_a > 0.0 && final_lpa_a.is_finite());
}

#[test]
fn test_juvenile_adult_hopf_cohort_oscillations_vs_subcritical_damping() {
    let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let hopf_params = JuvenileAdultParams::default();
    let mut hopf = JuvenileAdultSolver::new(81, 16.0, hopf_params);
    hopf.set_initial_state(16.0, |a| if a < 1.0 { 15.0 } else { 1.0 });

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
fn test_cannibalism_model_rk4_stepping_clamps_non_negative() {
    let _guard = VERIFIED_LOCK.lock().unwrap_or_else(|e| e.into_inner());
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
