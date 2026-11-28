#[cfg(not(target_arch = "wasm32"))]
mod host_sweeps {
    use domain_applied::applied::clinical_trials::survival_analysis::{
        Observation, kaplan_meier, log_rank_test,
    };
    use domain_applied::applied::clinical_trials::types::SurvivalTime;
    use domain_engineering::engineering::battery_degradation::{
        Cycles, DepthOfDischarge, PowerLawModel,
    };
    use domain_engineering::engineering::radar::FmcwConfig;
    use domain_engineering::engineering::{
        CuttingSpeed, MaterialConstant, ToolExponent, calculate_taylor_tool_life,
    };
    use domain_physics::physics::fluid_dynamics::lattice_boltzmann::{
        BgkCollision, CollisionModel, D2Q9, Lattice2D, LatticeBoltzmannD2Q9,
    };
    use domain_physics::physics::quantum::schrodinger::{
        evolve_state, state_norm_squared, time_evolution_operator,
    };
    use domain_physics::physics::quantum::types::{QuantumOperator, QuantumState};
    use nalgebra::{DMatrix, DVector};
    use num_complex::Complex;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest::test_runner::{Config, RngAlgorithm, TestRng, TestRunner};
    use pure_math::pure_math::special_functions::bessel::{bessel_j, bessel_y};
    use pure_math::pure_math::statistics::copula::{Correlation, GaussianCopula, Probability};
    use std::f64::consts::PI;

    fn seeded_runner(cases: u32) -> TestRunner {
        let cfg = Config {
            cases,
            failure_persistence: None,
            ..Config::default()
        };
        TestRunner::new_with_rng(cfg, TestRng::deterministic_rng(RngAlgorithm::ChaCha))
    }

    pub fn sweep_copula_and_special_functions() -> Result<usize, String> {
        let mut runner = seeded_runner(48);
        runner
            .run(
                &(-0.92f64..0.92f64, 0.03f64..0.95f64, 0.03f64..0.95f64),
                |(rho, u, v)| {
                    let cop = GaussianCopula::bivariate(Correlation::new(rho).unwrap()).unwrap();
                    let pu = Probability::new(u).unwrap();
                    let pv = Probability::new(v).unwrap();
                    let c = cop.cdf(&[pu, pv]).unwrap().value();
                    prop_assert!(c + 1e-9 >= (u + v - 1.0).max(0.0) && c <= u.min(v) + 1e-9);
                    Ok(())
                },
            )
            .map_err(|e| format!("Copula proptest failed: {e}"))?;

        let mut runner_bessel = seeded_runner(48);
        runner_bessel
            .run(&(0.2f64..4.5f64, 0.3f64..75.0f64), |(nu, x)| {
                let w = bessel_j(nu, x) * bessel_y(nu + 1.0, x)
                    - bessel_j(nu + 1.0, x) * bessel_y(nu, x);
                let exact = -2.0 / (PI * x);
                prop_assert!((w - exact).abs() < 1e-11 * exact.abs().max(1.0));
                Ok(())
            })
            .map_err(|e| format!("Bessel proptest failed: {e}"))?;

        Ok(2)
    }

    fn sweep_quantum_and_lbm() -> Result<usize, String> {
        let mut q_runner = seeded_runner(48);
        q_runner
            .run(
                &(
                    -2.5f64..2.5f64,
                    -2.5f64..2.5f64,
                    -1.5f64..1.5f64,
                    0.02f64..1.2f64,
                ),
                |(e0, e1, v, t)| {
                    let h = QuantumOperator::new(DMatrix::from_row_slice(
                        2,
                        2,
                        &[
                            Complex::new(e0, 0.0),
                            Complex::new(v, 0.25),
                            Complex::new(v, -0.25),
                            Complex::new(e1, 0.0),
                        ],
                    ));
                    let u = time_evolution_operator(&h, t, 1.0);
                    let id = DMatrix::<Complex<f64>>::identity(2, 2);
                    prop_assert!((u.matrix.adjoint() * &u.matrix - id).norm() < 1e-11);
                    let psi = QuantumState::new(DVector::from_vec(vec![
                        Complex::new(0.8, 0.0),
                        Complex::new(0.0, 0.6),
                    ]));
                    prop_assert!(
                        (state_norm_squared(&evolve_state(&psi, &h, t, 1.0)) - 1.0).abs() < 1e-11
                    );
                    Ok(())
                },
            )
            .map_err(|e| format!("Quantum proptest failed: {e}"))?;

        let mut lbm_runner = seeded_runner(36);
        lbm_runner
            .run(
                &(0.6f64..1.5f64, 0.8f64..1.3f64, -0.06f64..0.06f64),
                |(tau, rho, ux)| {
                    let mut f = D2Q9::equilibrium(rho, ux, -0.02);
                    let bgk = BgkCollision { tau };
                    <BgkCollision as CollisionModel<9, D2Q9>>::apply(&bgk, &mut f, rho, ux, -0.02);
                    prop_assert!((f.iter().sum::<f64>() - rho).abs() < 1e-12);
                    let mut grid = LatticeBoltzmannD2Q9::new(8, 6, tau);
                    grid.set_inlet(3, 2, 1, 1, ux, 0.0);
                    grid.step();
                    let m0: f64 = grid.state.density().iter().sum();
                    grid.step();
                    let m1: f64 = grid.state.density().iter().sum();
                    prop_assert!((m1 - m0).abs() < 1e-11);
                    Ok(())
                },
            )
            .map_err(|e| format!("LBM proptest failed: {e}"))?;
        Ok(2)
    }

    fn sweep_survival() -> Result<usize, String> {
        let mut km_runner = seeded_runner(48);
        let arm = vec((0.2f64..40.0f64, any::<bool>()), 4..18);
        km_runner
            .run(&(arm.clone(), arm), |(a1, a2)| {
                let g1: Vec<Observation> = a1
                    .iter()
                    .map(|&(t, ev)| Observation {
                        time: SurvivalTime::new(t).unwrap(),
                        event_occurred: ev,
                    })
                    .collect();
                let g2: Vec<Observation> = a2
                    .iter()
                    .map(|&(t, ev)| Observation {
                        time: SurvivalTime::new(t).unwrap(),
                        event_occurred: ev,
                    })
                    .collect();
                let mut prev = 1.0;
                for pt in kaplan_meier(&g1) {
                    prop_assert!(
                        pt.survival_probability >= 0.0 && pt.survival_probability <= prev + 1e-12
                    );
                    prev = pt.survival_probability;
                }
                if let (Ok(r1), Ok(r2)) = (log_rank_test(&g1, &g2), log_rank_test(&g2, &g1)) {
                    prop_assert!((r1.statistic - r2.statistic).abs() < 1e-10);
                }
                Ok(())
            })
            .map_err(|e| format!("Survival proptest failed: {e}"))?;
        Ok(1)
    }

    fn sweep_engineering() -> Result<usize, String> {
        let mut eng_runner = seeded_runner(48);
        eng_runner
            .run(
                &(0.25f64..3.5f64, 20.0f64..95.0f64, 25.0f64..90.0f64),
                |(range_m, dod_pct, speed_v)| {
                    let cfg = FmcwConfig::iwr6843_default();
                    let fb = FmcwConfig::beat_frequency(cfg.bandwidth / cfg.chirp_time, range_m);
                    prop_assert!((cfg.range_from_beat_frequency(fb) - range_m).abs() < 1e-11);

                    let battery = PowerLawModel::standard();
                    let dod = DepthOfDischarge::new_clamped(dod_pct);
                    let c10 = f64::from(battery.capacity(Cycles::new_clamped(10.0), dod));
                    let c200 = f64::from(battery.capacity(Cycles::new_clamped(200.0), dod));
                    prop_assert!(c200 <= c10 + 1e-12 && c200 >= 0.0 && c10 <= 1.0);

                    let c_mat = MaterialConstant::new(120.0).unwrap();
                    let n_exp = ToolExponent::new(0.25).unwrap();
                    let v = CuttingSpeed::new(speed_v).unwrap();
                    let t = calculate_taylor_tool_life(v, c_mat, n_exp).value();
                    prop_assert!((v.value() * t.powf(n_exp.value()) - c_mat.value()).abs() < 1e-9);
                    Ok(())
                },
            )
            .map_err(|e| format!("Engineering proptest failed: {e}"))?;
        Ok(1)
    }

    pub fn sweep_quantum_lbm_and_survival() -> Result<usize, String> {
        Ok(sweep_quantum_and_lbm()? + sweep_survival()? + sweep_engineering()?)
    }
}

/// Executes all 6 seeded deterministic Tier 2 property suites on host targets, returning `6`.
pub fn verify_property_sweeps() -> Result<usize, String> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let a = host_sweeps::sweep_copula_and_special_functions()?;
        let b = host_sweeps::sweep_quantum_lbm_and_survival()?;
        Ok(a + b)
    }
    #[cfg(target_arch = "wasm32")]
    {
        Ok(0)
    }
}
