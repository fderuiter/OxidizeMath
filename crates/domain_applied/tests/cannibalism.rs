//! Integration tests for cannibalism.
use domain_applied::applied::cannibalism;

#[test]
#[verified_engine::verified]
fn test_mckendrick_von_foerster() {
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
    let t = 0.0;
    let result = cannibalism::birth_rate(t);
    assert_eq!(result, 100.0);
}

#[test]
#[verified_engine::verified]
fn test_death_rate() {
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
    let n_t_alpha = 50.0;
    let f_i_t = 0.1;
    let a_t = 10.0;
    let result = cannibalism::adult_dynamics(n_t_alpha, f_i_t, a_t);
    assert_eq!(result, 49.0);
}

#[test]
#[verified_engine::verified]
fn test_characteristic_advection_and_exponential_mass_balance() {
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

#[global_allocator]
static ALLOCATOR: verified_engine::allocator::VerifiedAllocator =
    verified_engine::allocator::VerifiedAllocator;

#[test]
#[verified_engine::verified]
fn test_zero_heap_allocation_during_pde_stepping() {
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
