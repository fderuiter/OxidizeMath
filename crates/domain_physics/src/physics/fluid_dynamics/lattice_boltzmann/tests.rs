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
    #[verified_engine::verified]
    fn test_row_major_velocity_and_obstacle_copier_transposes_dmatrix() {
        use math_commons::math_kernel::types::flatten_2d_index;

        let width = 7;
        let height = 4;
        let mut solver: LatticeBoltzmannD2Q9<BgkCollision> =
            LatticeBoltzmannD2Q9::new(width, height, 0.8);

        solver.set_obstacle(5, 2, true);
        solver.set_inlet(1, 3, 2, 1, 0.09, -0.04);

        let total = width * height;
        let mut ux = vec![0.0; total];
        let mut uy = vec![0.0; total];
        let mut obs = vec![false; total];
        solver.copy_velocity_and_obstacles_row_major(&mut ux, &mut uy, &mut obs);

        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                let (expected_ux, expected_uy) = solver.get_velocity(x, y);
                assert!((ux[idx] - expected_ux).abs() < 1e-14);
                assert!((uy[idx] - expected_uy).abs() < 1e-14);
                assert_eq!(obs[idx], solver.is_obstacle(x, y));
            }
        }
        assert!(obs[flatten_2d_index(5, 2, width)]);
        assert!((ux[flatten_2d_index(1, 3, width)] - 0.09).abs() < 1e-14);
        assert!((uy[flatten_2d_index(1, 3, width)] + 0.04).abs() < 1e-14);
    }
}
