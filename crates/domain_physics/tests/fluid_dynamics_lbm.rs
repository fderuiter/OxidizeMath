//! Integration tests for fluid dynamics lbm.
use approx::assert_relative_eq;
use domain_physics::physics::fluid_dynamics::lattice_boltzmann::LatticeBoltzmannD2Q9;

#[test]
#[verified_engine::verified]
fn test_lbm_initialization() {
    let width = 20;
    let height = 10;
    let tau = 1.0;
    let solver = LatticeBoltzmannD2Q9::new(width, height, tau);

    // Check initial density is 1.0 everywhere
    for y in 0..height {
        for x in 0..width {
            assert_relative_eq!(
                solver.get_density(x, y),
                1.0,
                epsilon = math_commons::registry::TOLERANCE_FAST
            );
            let (ux, uy) = solver.get_velocity(x, y);
            assert_relative_eq!(ux, 0.0, epsilon = math_commons::registry::TOLERANCE_FAST);
            assert_relative_eq!(uy, 0.0, epsilon = math_commons::registry::TOLERANCE_FAST);
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_lbm_mass_conservation() {
    let width = 20;
    let height = 10;
    let tau = 1.0;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);

    let mut initial_mass = 0.0;
    for x in 0..width {
        for y in 0..height {
            initial_mass += solver.get_density(x, y);
        }
    }

    // Run for some steps
    for _ in 0..50 {
        solver.step();
    }

    let mut final_mass = 0.0;
    for x in 0..width {
        for y in 0..height {
            final_mass += solver.get_density(x, y);
        }
    }

    assert_relative_eq!(
        initial_mass,
        final_mass,
        epsilon = math_commons::registry::TOLERANCE_HIGH
    );
}

#[test]
#[verified_engine::verified]
fn test_lbm_velocity_propagation() {
    let width = 20;
    let height = 10;
    let tau = 0.6; // Low viscosity
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, tau);

    // Set inlet velocity at x=0..2 across channel height
    solver.set_inlet(0, 0, 2, height, 0.1, 0.0);

    // Run enough steps for wake momentum to propagate well past x = 2
    for _ in 0..25 {
        solver.set_inlet(0, 0, 2, height, 0.1, 0.0);
        solver.step();
    }

    // Check that velocity has propagated downstream past x = 2 (e.g. x = 10)
    let (ux_downstream, _) = solver.get_velocity(10, 5);
    assert!(
        ux_downstream > 0.02,
        "Downstream wake should propagate past x = 2, got ux = {ux_downstream}"
    );
}

#[test]
#[verified_engine::verified]
fn test_lbm_evolution_engine_step_no_double_swap() {
    use pure_math::pure_math::analysis::evolution::DoubleBufferedEvolutionEngine;

    let mut direct = LatticeBoltzmannD2Q9::new(16, 8, 0.7);
    let mut engine = LatticeBoltzmannD2Q9::new(16, 8, 0.7);
    let mut external_state = engine.state.clone();
    let mut rng = oxidize_core::rng::OxidizeRng::default();

    for _ in 0..12 {
        direct.set_inlet(0, 0, 2, 8, 0.08, 0.0);
        direct.step();

        std::mem::swap(&mut engine.state, &mut external_state);
        engine.set_inlet(0, 0, 2, 8, 0.08, 0.0);
        std::mem::swap(&mut engine.state, &mut external_state);
        DoubleBufferedEvolutionEngine::step_buffered(
            &mut engine,
            &mut external_state,
            &mut (),
            &mut rng,
            1.0,
        )
        .expect("DoubleBufferedEvolutionEngine::step_buffered should succeed");
    }

    std::mem::swap(&mut engine.state, &mut external_state);
    for y in 0..8 {
        for x in 0..16 {
            let (ux_a, uy_a) = direct.get_velocity(x, y);
            let (ux_b, uy_b) = engine.get_velocity(x, y);
            assert_relative_eq!(ux_a, ux_b, epsilon = 1e-12);
            assert_relative_eq!(uy_a, uy_b, epsilon = 1e-12);
        }
    }
}

#[test]
#[verified_engine::verified]
fn test_lbm_karman_vortex_shedding_wake() {
    let width = 80;
    let height = 40;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, 0.555);

    // Offset circular cylinder slightly off centerline to trigger Karman instability deterministically
    let cx = 18_isize;
    let cy = 20_isize;
    let r_sq = 9_isize; // radius = 3
    for y in 0..height {
        for x in 0..width {
            let dx = x as isize - cx;
            let dy = y as isize - cy;
            if dx * dx + dy * dy <= r_sq {
                solver.set_obstacle(x, y, true);
            }
        }
    }

    // Small initial transverse perturbation in the near wake to break symmetry rapidly
    solver.set_inlet(0, 0, width, height, 0.11, 0.0);
    solver.set_inlet(23, 17, 6, 6, 0.11, 0.02);

    let mut min_uy = f64::INFINITY;
    let mut max_uy = f64::NEG_INFINITY;
    let mut sign_flips = 0_usize;
    let mut prev_sign = 0_i8;

    for step in 0..560 {
        solver.set_inlet(0, 1, 2, height - 2, 0.11, 0.0);
        solver.step();
        solver.apply_open_outlet();

        if step >= 150 {
            let (_, uy_probe) = solver.get_velocity(38, 20);
            if uy_probe < min_uy {
                min_uy = uy_probe;
            }
            if uy_probe > max_uy {
                max_uy = uy_probe;
            }
            let sign = if uy_probe > 1e-3 {
                1
            } else if uy_probe < -1e-3 {
                -1
            } else {
                0
            };
            if sign != 0 && prev_sign != 0 && sign != prev_sign {
                sign_flips += 1;
            }
            if sign != 0 {
                prev_sign = sign;
            }
        }
    }

    let (ux_far, _) = solver.get_velocity(58, 20);
    assert!(
        ux_far > 0.03,
        "Inlet momentum should reach far wake x=58, got ux={ux_far}"
    );
    assert!(
        min_uy < -2e-3 && max_uy > 2e-3 && sign_flips >= 2,
        "Expected alternating Karman vortices in cylinder wake: min_uy={min_uy}, max_uy={max_uy}, flips={sign_flips}"
    );
}

#[test]
#[verified_engine::verified]
fn test_lbm_obstacle() {
    let width = 10;
    let height = 10;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, 1.0);

    solver.set_obstacle(5, 5, true);
    assert!(solver.is_obstacle(5, 5));

    // Velocity inside obstacle should be 0
    let (ux, uy) = solver.get_velocity(5, 5);
    assert_relative_eq!(ux, 0.0);
    assert_relative_eq!(uy, 0.0);

    solver.step();

    // Should still be 0
    let (ux, uy) = solver.get_velocity(5, 5);
    assert_relative_eq!(ux, 0.0);
    assert_relative_eq!(uy, 0.0);
}
