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
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod proptest_suite {
    use crate::physics::fluid_dynamics::lattice_boltzmann::model::*;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};

    #[test]
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
                let cx = D2Q9::directions_x();
                let cy = D2Q9::directions_y();
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
                let fluid_mass = |s: &LatticeBoltzmannD2Q9<BgkCollision>| -> f64 {
                    s.state
                        .f
                        .iter()
                        .zip(s.state.obstacles.iter())
                        .filter(|&(_, &obs)| !obs)
                        .map(|(cell, _)| cell.iter().sum::<f64>())
                        .sum()
                };
                let initial_mass = fluid_mass(&lbm);
                for _ in 0..8 {
                    lbm.step();
                }
                prop_assert!((fluid_mass(&lbm) - initial_mass).abs() < 1e-11);
                Ok(())
            })
            .unwrap();
    }
}
