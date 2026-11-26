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
}
