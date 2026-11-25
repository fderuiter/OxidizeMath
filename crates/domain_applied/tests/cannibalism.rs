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

#[test]
#[verified_engine::verified]
fn test_non_unit_cfl_grid_refinement_convergence_mckendrick() {
    use std::f64::consts::PI;

    let (mu, max_age, t_end, cfl) = (0.2, 6.0, 0.8, 0.5);
    let grids = [61_usize, 121, 241, 481];
    let mut mvf_errors = [0.0_f64; 4];
    let init_profile = |a: f64| -> f64 {
        if (1.0..=4.0).contains(&a) {
            10.0 * (PI * (a - 1.0) / 3.0).sin().powi(4)
        } else {
            0.0
        }
    };

    for (idx, &n_ages) in grids.iter().enumerate() {
        let kernel = cannibalism::CannibalismKernel::new(mu, 0.0, 0.0, 2.0, 0.0, 4.0, 0.0, 50.0);
        let mut solver = cannibalism::McKendrickSolver::new(n_ages, max_age, kernel);
        solver.set_initial_density(init_profile);
        let steps = (t_end / (cfl * solver.da())).round() as usize;
        let exact_dt = t_end / steps as f64;
        for _ in 0..steps {
            solver.step_upwind_with_boundary(exact_dt, 0.0);
        }
        let mut l2_sq = 0.0;
        for (&age, &num) in solver.ages().iter().zip(solver.density().iter()) {
            let exact = if age >= t_end {
                init_profile(age - t_end) * (-mu * t_end).exp()
            } else {
                0.0
            };
            l2_sq += (num - exact).powi(2) * solver.da();
        }
        mvf_errors[idx] = l2_sq.sqrt();
    }

    let mvf_order = (mvf_errors[2] / mvf_errors[3]).ln() / 2.0_f64.ln();
    assert!(mvf_errors[3] < mvf_errors[2] && mvf_errors[2] < mvf_errors[1]);
    assert!(
        mvf_order > 0.85,
        "McKendrickSolver expected O(h) non-unit CFL convergence, got {mvf_order:.3}"
    );
}

#[test]
#[verified_engine::verified]
fn test_non_unit_cfl_grid_refinement_convergence_juvenile_adult() {
    use std::f64::consts::PI;

    let params = cannibalism::JuvenileAdultParams::default();
    let run_ja = |n_ages: usize| -> (f64, f64) {
        let mut ja = cannibalism::JuvenileAdultSolver::new(n_ages, 12.0, params);
        ja.set_initial_state(12.0, |a| 6.0 * (PI * a / params.maturation_age).sin());
        let steps = (0.5 / (0.5 * ja.da())).round() as usize;
        let step_dt = 0.5 / steps as f64;
        for _ in 0..steps {
            ja.step_upwind(step_dt);
        }
        (ja.adults(), ja.total_juveniles())
    };

    let (a_ref, j_ref) = run_ja(961);
    let mut ja_errs = [0.0_f64; 3];
    for (idx, &n_ages) in [61_usize, 121, 241].iter().enumerate() {
        let (a_n, j_n) = run_ja(n_ages);
        ja_errs[idx] = ((a_n - a_ref).powi(2) + (j_n - j_ref).powi(2)).sqrt();
    }
    let ja_order = (ja_errs[1] / ja_errs[2]).ln() / 2.0_f64.ln();
    assert!(
        ja_errs[2] < ja_errs[1] && ja_errs[1] < ja_errs[0] && ja_order > 0.85,
        "JuvenileAdultSolver expected O(h) convergence at CFL=0.5, errs={ja_errs:?}, order={ja_order:.3}"
    );
}
