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
    fn test_row_major_velocity_and_obstacle_projection_zero_alloc() {
        use crate::physics::fluid_dynamics::lattice_boltzmann::{
            advance_wind_tunnel_step, project_obstacles_row_major_into,
            project_velocity_row_major_into, sync_obstacles_row_major,
        };
        use math_commons::math_kernel::types::flatten_2d_index;

        let (width, height) = (12, 7);
        let mut solver = LatticeBoltzmannD2Q9::new(width, height, 0.65);
        let mut row_obs = vec![false; width * height];
        row_obs[flatten_2d_index(5, 3, width)] = true;

        let mut out_ux = vec![0.0; width * height];
        let mut out_uy = vec![0.0; width * height];
        let mut out_obs = vec![false; width * height];

        let ((), _metrics) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            sync_obstacles_row_major(&mut solver, &row_obs);
            for _ in 0..8 {
                advance_wind_tunnel_step(&mut solver, 0.1);
            }
            project_velocity_row_major_into(&solver, &mut out_ux, &mut out_uy);
            project_obstacles_row_major_into(&solver, &mut out_obs);
        });

        assert!(solver.is_obstacle(5, 3));
        assert_eq!(out_obs, row_obs);
        assert_eq!(out_ux[flatten_2d_index(5, 3, width)], 0.0);
        assert_eq!(out_uy[flatten_2d_index(5, 3, width)], 0.0);

        for y in 0..height {
            for x in 0..width {
                let idx = flatten_2d_index(x, y, width);
                let (vx, vy) = solver.get_velocity(x, y);
                assert!((out_ux[idx] - vx).abs() < 1e-12);
                assert!((out_uy[idx] - vy).abs() < 1e-12);
            }
        }

        // Erasing obstacle restores rest equilibrium safely
        row_obs[flatten_2d_index(5, 3, width)] = false;
        sync_obstacles_row_major(&mut solver, &row_obs);
        assert!(!solver.is_obstacle(5, 3));
        assert!((solver.get_density(5, 3) - 1.0).abs() < 1e-12);
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod proptest_suite {
    use crate::physics::fluid_dynamics::lattice_boltzmann::model::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    #[verified_engine::verified]
    fn total_fluid_mass(s: &LatticeBoltzmannD2Q9<BgkCollision>) -> f64 {
        s.state
            .f
            .iter()
            .zip(s.state.obstacles.iter())
            .filter(|&(_, &obs)| !obs)
            .map(|(cell, _)| cell.iter().sum::<f64>())
            .sum()
    }

    #[test]
    #[verified_engine::verified]
    fn proptest_lbm_bgk_and_streaming_mass_momentum_conservation() {
        let mut runner = TestRunner::new_with_rng(
            Config::with_cases(48),
            TestRng::deterministic_rng(RngAlgorithm::ChaCha),
        );
        let strategy = (
            0.6f64..1.6f64,
            0.7f64..1.4f64,
            -0.08f64..0.08f64,
            -0.08f64..0.08f64,
        );
        runner
            .run(&strategy, |(tau, rho0, ux0, uy0)| {
                let (cx, cy) = (D2Q9::directions_x(), D2Q9::directions_y());
                let mut f = D2Q9::equilibrium(rho0, ux0, uy0);
                let bgk = BgkCollision { tau };
                <BgkCollision as CollisionModel<9, D2Q9>>::apply(&bgk, &mut f, rho0, ux0, uy0);
                let m: f64 = f.iter().sum();
                let jx: f64 = f.iter().enumerate().map(|(k, &v)| v * cx[k] as f64).sum();
                let jy: f64 = f.iter().enumerate().map(|(k, &v)| v * cy[k] as f64).sum();
                prop_assert!((m - rho0).abs() < 1e-12);
                prop_assert!((jx - rho0 * ux0).abs() < 1e-12);
                prop_assert!((jy - rho0 * uy0).abs() < 1e-12);

                let mut lbm = LatticeBoltzmannD2Q9::new(8, 6, tau);
                lbm.set_obstacle(3, 2, true);
                lbm.state.f[(2, 2)] = D2Q9::equilibrium(rho0, ux0, uy0);
                let initial_mass = total_fluid_mass(&lbm);
                for _ in 0..8 {
                    lbm.step();
                }
                prop_assert!((total_fluid_mass(&lbm) - initial_mass).abs() < 1e-11);
                Ok(())
            })
            .unwrap();
    }
}
