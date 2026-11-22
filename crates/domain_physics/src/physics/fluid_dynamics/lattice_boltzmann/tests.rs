#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {

    use crate::physics::fluid_dynamics::lattice_boltzmann::model::*;

    const Q: usize = 9;

    #[test]
    #[verified_engine::verified]
    fn test_bgk_collision() {
        let tau = 1.0;
        let strategy = BgkCollision { tau };
        let mut f = [0.0; Q];
        let rho = 1.0;
        let ux = 0.0;
        let uy = 0.0;

        let eq = D2Q9::equilibrium(rho, ux, uy);
        for k in 0..Q {
            f[k] = eq[k] + 0.1;
        }

        <BgkCollision as CollisionModel<9, D2Q9>>::apply(&strategy, &mut f, rho, ux, uy);

        for k in 0..Q {
            assert!((f[k] - eq[k]).abs() < math_commons::registry::TOLERANCE_STANDARD);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_solver_initialization() {
        let solver = LatticeBoltzmannD2Q9::new(10, 10, 1.0);
        assert_eq!(solver.state.rho.len(), 100);
        for i in 0..100 {
            assert!((solver.state.rho[i] - 1.0).abs() < math_commons::registry::TOLERANCE_STANDARD);
            assert!(solver.state.ux[i].abs() < math_commons::registry::TOLERANCE_STANDARD);
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_gui_compliance_dynamic_inputs() {
        let width = 20;
        let height = 10;
        let tau = 1.0;

        let mut solver: LatticeBoltzmannD2Q9<BgkCollision> =
            LatticeBoltzmannD2Q9::new(width, height, tau);

        solver.set_inlet(0, 4, 1, 2, 0.1, 0.0);
        solver.step();
        let inlet_u = solver.get_velocity_magnitude(0, 4);
        assert!(inlet_u > 0.0);

        solver.collision_model.tau = 2.0;
        solver.step();
        assert!(solver.state.rho[0].is_finite());

        let obs_x = 10;
        let obs_y = 5;
        solver.set_obstacle(obs_x, obs_y, true);
        solver.step();

        assert!(solver.is_obstacle(obs_x, obs_y));
        let (ux, uy) = solver.get_velocity(obs_x, obs_y);
        assert_eq!(ux, 0.0);
        assert_eq!(uy, 0.0);

        solver.clear_obstacles();
        assert!(!solver.is_obstacle(obs_x, obs_y));

        solver = LatticeBoltzmannD2Q9::new(width, height, 0.6);
        assert!(
            (solver.collision_model.tau - 0.6).abs() < math_commons::registry::TOLERANCE_STANDARD
        );
    }

    #[test]
    fn test_inlet_propagates_downstream_and_zero_allocation_step() {
        let width = 40;
        let height = 20;
        let mut solver = LatticeBoltzmannD2Q9::new(width, height, 0.56);
        solver.set_obstacle(8, 10, true);

        let mut ux_row = vec![0.0; width * height];
        let mut uy_row = vec![0.0; width * height];
        let mut obs_row = vec![false; width * height];

        verified_engine::engine::VerifiedEngine::run_verified(|| {
            for _ in 0..30 {
                solver.set_inlet(0, 1, 1, height - 2, 0.08, 0.0);
                solver.step();
            }
            solver
                .state
                .copy_macroscopic_row_major(&mut ux_row, &mut uy_row, &mut obs_row);
        });

        let downstream_speed = solver.get_velocity_magnitude(15, 10);
        assert!(
            downstream_speed > 1e-4,
            "Expected non-zero downstream velocity at x=15 within 30 steps, got {downstream_speed}"
        );

        let obs_idx = math_commons::math_kernel::types::flatten_2d_index(8, 10, width);
        assert!(obs_row[obs_idx]);
        assert_eq!(ux_row[obs_idx], 0.0);
        assert_eq!(uy_row[obs_idx], 0.0);

        let sample_idx = math_commons::math_kernel::types::flatten_2d_index(15, 10, width);
        let (ux_expected, uy_expected) = solver.get_velocity(15, 10);
        assert!((ux_row[sample_idx] - ux_expected).abs() < 1e-12);
        assert!((uy_row[sample_idx] - uy_expected).abs() < 1e-12);
    }
}
