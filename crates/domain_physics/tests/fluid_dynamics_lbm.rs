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

    // Set inlet velocity at x=5, y=5
    solver.set_inlet(5, 5, 2, 2, 0.1, 0.0);

    // Run steps
    for _ in 0..10 {
        solver.step();
    }

    // Check that velocity has propagated
    let (ux_center, _) = solver.get_velocity(6, 5);
    assert!(ux_center > 0.0, "Velocity should propagate from inlet");
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
    assert_eq!(solver.obstacle_cell_count(), 1);
}

#[test]
#[verified_engine::verified]
fn test_lbm_multistep_channel_flow_propagation() {
    let width = 50;
    let height = 25;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, 0.62);

    for _ in 0..80 {
        solver.set_inlet(0, 0, 2, height, 0.1, 0.0);
        solver.step();
    }

    let mut active_cells = 0usize;
    let mut interior_cells = 0usize;
    for y in 2..(height - 2) {
        for x in 2..(width - 2) {
            interior_cells += 1;
            if solver.get_velocity_magnitude(x, y) > 1e-4 {
                active_cells += 1;
            }
        }
    }

    assert!(
        active_cells * 10 >= interior_cells * 8,
        "Expected >=80% active interior cells after 80 steps, got {active_cells}/{interior_cells}"
    );
}

#[test]
#[verified_engine::verified]
fn test_lbm_cylinder_drag_and_vorticity_curl_antisymmetry() {
    let width = 60;
    let height = 31; // Odd height places y = 15 on exact horizontal symmetry axis
    let cy = 15i32;
    let cx = 18i32;
    let radius = 3i32;
    let mut solver = LatticeBoltzmannD2Q9::new(width, height, 0.65);

    let mut expected_solid = 0usize;
    for y in 0..height {
        for x in 0..width {
            let dx = x as i32 - cx;
            let dy = y as i32 - cy;
            if dx * dx + dy * dy <= radius * radius {
                solver.set_obstacle(x, y, true);
                expected_solid += 1;
            }
        }
    }
    assert_eq!(solver.obstacle_cell_count(), expected_solid);

    for _ in 0..100 {
        solver.set_inlet(0, 0, 2, height, 0.08, 0.0);
        solver.step();
    }

    let (fx, fy) = solver.net_obstacle_force();
    assert!(
        fx > 0.01,
        "Expected positive streamwise drag on cylinder, got fx={fx}"
    );
    assert!(
        fy.abs() < 1e-10,
        "Expected near-zero net lift for symmetric cylinder flow, got fy={fy}"
    );

    // Vorticity inside obstacle is strictly zero and antisymmetric across centerline y = 15
    assert_eq!(solver.vorticity_curl(cx as usize, cy as usize), 0.0);
    let mut max_wake_curl = 0.0_f64;
    for x in 22..40 {
        for offset in 1..=8usize {
            let curl_top = solver.vorticity_curl(x, (cy as usize) + offset);
            let curl_bot = solver.vorticity_curl(x, (cy as usize) - offset);
            max_wake_curl = max_wake_curl.max(curl_top.abs());
            assert_relative_eq!(curl_top, -curl_bot, epsilon = 1e-11);
        }
    }
    assert!(
        max_wake_curl > 1e-4,
        "Expected non-trivial shear vorticity in cylinder wake, got {max_wake_curl}"
    );
}
