//! Integration tests for cannibalism PDE solvers, operators, and bifurcation models.
use domain_applied::applied::cannibalism;
use std::sync::Mutex;

#[global_allocator]
static ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

static ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
#[verified_engine::verified]
fn test_mckendrick_von_foerster() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let t = 0.0;
    let a = 0.0;
    let mu = 0.1;
    let n = 100.0;
    let result = cannibalism::mckendrick_von_foerster(t, a, mu, n);
    assert_eq!(result, -10.0);
}

#[test]
#[verified_engine::verified]
fn test_birth_rate() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let t = 0.0;
    let result = cannibalism::birth_rate(t);
    assert_eq!(result, 100.0);
}

#[test]
#[verified_engine::verified]
fn test_death_rate() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let nu_a = 0.1;
    let c_a = 0.2;
    let k_t = 10.0;
    let phi_c_t = 1.0;
    let result = cannibalism::death_rate(nu_a, c_a, k_t, phi_c_t);
    assert_eq!(result, 2.1);
}

#[test]
#[verified_engine::verified]
fn test_juvenile_dynamics() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let i_t = 0.1;
    let c_a = 0.2;
    let a_t = 10.0;
    let n_t_a = 100.0;
    let result = cannibalism::juvenile_dynamics(i_t, c_a, a_t, n_t_a);
    assert_eq!(result, -210.0);
}

#[test]
#[verified_engine::verified]
fn test_adult_dynamics() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let n_t_alpha = 50.0;
    let f_i_t = 0.1;
    let a_t = 10.0;
    let result = cannibalism::adult_dynamics(n_t_alpha, f_i_t, a_t);
    assert_eq!(result, 49.0);
}

#[test]
#[verified_engine::verified]
fn test_characteristic_advection_and_exponential_mass_balance() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mu_const = 0.25;
    let kernel = cannibalism::CannibalismKernel::new(mu_const, 0.0, 0.0, 2.0, 0.05, 4.0, 0.0, 50.0);
    let mut solver = cannibalism::McKendrickSolver::new(41, 8.0, kernel);
    solver.set_initial_density(|a| if (1.0..=3.0).contains(&a) { 20.0 } else { 0.0 });
    let initial_mass = solver.total_population();
    let da = solver.da();

    // Advance 8 exact characteristic steps (t = 1.6) with zero boundary birth
    for _ in 0..8 {
        solver.step_upwind_with_boundary(da, 0.0);
    }

    let expected_mass = initial_mass * (-mu_const * solver.time()).exp();
    assert!((solver.total_population() - expected_mass).abs() < 1e-10);
    // Support shifted from [1.0, 3.0] to [2.6, 4.6]: ages < 2.4 must be zero
    assert!(solver.density()[10] == 0.0);
    assert!(solver.density()[18] > 10.0);
}

#[test]
#[verified_engine::verified]
fn test_cfl_substepping_and_extinction_zero_cannibalism_guards() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let kernel = cannibalism::CannibalismKernel::new(0.1, 0.01, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0);
    assert_eq!(kernel.ricker_saturation(0.0), 1.0);
    assert_eq!(kernel.vulnerability_at(1.0), 0.0);
    assert_eq!(kernel.fertility_at(5.0, 0.0), 0.0);

    let mut solver = cannibalism::McKendrickSolver::new(33, 8.0, kernel);
    // Extinction state N(t) = 0 with large dt > da (triggers CFL substepping)
    solver.step_upwind(1.5);
    assert_eq!(solver.total_population(), 0.0);
    assert!(solver.density().iter().all(|v| v.is_finite() && *v == 0.0));

    // Negative and NaN initial inputs are clamped to 0.0
    solver.set_initial_density(|a| if a < 2.0 { -5.0 } else { f64::NAN });
    solver.step_upwind(1.0);
    assert_eq!(solver.total_population(), 0.0);
}

#[test]
#[verified_engine::verified]
fn test_zero_heap_allocation_during_pde_stepping() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut mvf =
        cannibalism::McKendrickSolver::new(64, 10.0, cannibalism::CannibalismKernel::default());
    mvf.set_initial_density(|a| 12.0 * (-0.3 * a).exp());

    let mut ja = cannibalism::JuvenileAdultSolver::new(
        64,
        15.0,
        cannibalism::JuvenileAdultParams::default(),
    );
    ja.set_initial_state(15.0, |a| 8.0 * (-0.25 * a).exp());

    verified_engine::metrics::enable_verification();
    verified_engine::allocator::lock_allocations();
    mvf.step_upwind(0.5);
    ja.step_upwind(0.5);
    verified_engine::allocator::unlock_allocations();

    assert!(mvf.total_population() > 0.0 && mvf.total_population().is_finite());
    assert!(ja.adults() > 0.0 && ja.adults().is_finite());
}

#[test]
#[verified_engine::verified]
fn test_upwind_operators_and_non_negative_clamping() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(cannibalism::death_rate(-0.3, -0.5, 10.0, 1.0), 0.0);
    assert_eq!(cannibalism::death_rate(f64::NAN, 0.2, 5.0, 1.0), 0.0);
    assert!(cannibalism::birth_rate(std::f64::consts::PI) > 149.0);

    let rhs = cannibalism::mckendrick_upwind_transport(8.0, 10.0, 0.5, 0.25);
    assert!((rhs - 2.0).abs() < 1.0e-12);

    let juv_rhs = cannibalism::juvenile_upwind_transport(10.0, 14.0, 0.5, 0.1, 0.2, 2.0);
    assert!((juv_rhs - 3.0).abs() < 1.0e-12);

    let mut juv = [12.0, 9.0, 6.0, 3.0];
    let mut scratch = [0.0; 4];
    let attack = [0.4, 0.3, 0.1, 0.0];
    let params = cannibalism::CoupledMaturationParams {
        birth_rate_per_adult: 18.0,
        juvenile_mortality: 0.2,
        adult_mortality: 0.25,
        dt: 0.2,
    };
    let next_adults =
        cannibalism::step_juvenile_adult_coupled(&mut juv, &mut scratch, &attack, 4.0, params);
    assert!(next_adults > 0.0);
    assert!(juv.iter().all(|&v| v >= 0.0));
}

#[test]
fn test_diekmann_equilibrium_and_hopf_cohort_waves() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut sub_solver = cannibalism::AgeStructuredSolver::new(
        cannibalism::AgeStructuredConfig::diekmann_subcritical(),
    );
    let eq = sub_solver
        .diekmann_equilibrium()
        .expect("subcritical Diekmann equilibrium must exist");
    assert!(
        (eq - 3.6715).abs() < 1.0e-4,
        "Expected analytical Diekmann equilibrium A* ≈ 3.6715, got {eq:.6}"
    );

    sub_solver.step_n(2400);
    let converged_a = sub_solver.adult_population();
    assert!(
        (converged_a - eq).abs() < 0.03,
        "Subcritical PDE adult density {converged_a:.4} should converge near A* = {eq:.4}"
    );

    let mut hopf_solver =
        cannibalism::AgeStructuredSolver::new(cannibalism::AgeStructuredConfig::diekmann_hopf());
    hopf_solver.step_n(1800);
    let mut min_a = f64::INFINITY;
    let mut max_a = 0.0_f64;
    for _ in 0..800 {
        hopf_solver.step();
        let a = hopf_solver.adult_population();
        assert!(a > 0.0 && a.is_finite());
        min_a = min_a.min(a);
        max_a = max_a.max(a);
    }
    assert!(
        max_a - min_a > 4.0,
        "Supercritical beta0 = 45.0 must sustain Diekmann Hopf cohort waves, got [{min_a:.4}, {max_a:.4}]"
    );
}

#[test]
fn test_tribolium_callow_two_cycles_and_pde_positivity() {
    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let lpa = cannibalism::TriboliumLpaModel::two_cycle_preset(0.7);
    let mut state = [1.5, 0.5, 1.0];
    for _ in 0..2400 {
        state = lpa.step(state);
    }
    let a_even = state[2];
    state = lpa.step(state);
    let a_odd = state[2];
    state = lpa.step(state);
    let a_even_next = state[2];

    assert!(a_even > 0.0 && a_odd > 0.0);
    assert!(
        (a_even - a_odd).abs() > 1.5,
        "c_pa = 0.7 must produce distinct 2-cycle branches: {a_even:.4} vs {a_odd:.4}"
    );
    assert!(
        (a_even - a_even_next).abs() < 1.0e-5,
        "2-cycle must repeat every 2 steps: {a_even:.6} vs {a_even_next:.6}"
    );

    let mut tri_pde = cannibalism::AgeStructuredSolver::new(
        cannibalism::AgeStructuredConfig::tribolium_cycles(0.7),
    );
    tri_pde.step_n(1200);
    assert!(tri_pde.density().iter().all(|&d| d >= 0.0 && d.is_finite()));
    assert!(tri_pde.adult_population() > 0.0);
}
