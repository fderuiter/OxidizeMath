//! Integration tests verifying that external library consumers can import and execute all promoted
//! GUI solvers through `math_explorer` with zero dynamic heap allocations under `VerifiedEngine`.

#![cfg(all(
    feature = "pure_math",
    feature = "applied",
    feature = "biology",
    feature = "physics"
))]

use std::sync::Mutex;

use math_explorer::verified_engine::engine::VerifiedEngine;

static ALLOC_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn promoted_kalman_plants_and_grpo_run_zero_alloc_via_math_explorer() {
    use math_explorer::applied::algorithms::kalman::{
        FastCvKalman, FastEkf, LorenzKalmanModel, lorenz_rhs_and_jac, rk4_step_with_jacobian,
    };
    use math_explorer::applied::engineering::{
        LORENZ_UPO_STEPS, precompute_viswanath_upo, upo_features_and_bin,
    };
    use math_explorer::applied::grpo::{
        GRPO_GROUP_SIZE, GrpoController, GrpoUpdateParams, ROLLOUT_HORIZON,
    };
    use math_explorer::applied::pharmacokinetics::{PkClosedLoopConfig, PkClosedLoopTwin};
    use math_explorer::rng::OxidizeRng;

    let mut ekf = FastEkf::try_from_validated_builder(
        LorenzKalmanModel {
            sigma: 10.0,
            rho_eff: 28.0,
            beta: 8.0 / 3.0,
            q_var: 0.02,
            r_var: 0.25,
        },
        0.01,
        [1.0; 3],
        [1.0; 3],
    )
    .expect("valid Lorenz EKF");
    let mut cv = FastCvKalman::with_unit_covariance(0.0, 0.025, 3.0e-4, 2.0e-5);
    let pk_cfg = PkClosedLoopConfig::default();
    let mut pk_twin = PkClosedLoopTwin::try_new(pk_cfg).expect("valid PK twin");
    let mut grpo = GrpoController::new_untrained();
    let mut rng = OxidizeRng::new(19);

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        let (next, _) = rk4_step_with_jacobian(&[1.0, 2.0, 3.0], 0.01, |s| {
            lorenz_rhs_and_jac(s, 10.0, 28.0, 8.0 / 3.0)
        });
        ekf.predict_rk4(0.01, &[0.02; 3], |s| {
            lorenz_rhs_and_jac(s, 10.0, 28.0, 8.0 / 3.0)
        });
        ekf.update_unit_obs(0, next[0], 0.25);
        assert!(ekf.std_dev(0).is_finite());

        cv.predict();
        cv.update(0.004);
        assert!(cv.predict_ahead(0.05).is_finite());

        let (upo_pts, upo_vel) = precompute_viswanath_upo(1.558_652_210_716_2 / 156.0);
        assert_eq!(upo_pts.len(), LORENZ_UPO_STEPS);
        let (phi, bin) = upo_features_and_bin(&ekf.state, &upo_pts, &upo_vel);
        assert!(bin < 64);

        let pk_out = pk_twin.step(0.4, pk_cfg, &mut rng);
        assert!(pk_out.ekf_primary.is_finite());

        for g in 0..GRPO_GROUP_SIZE {
            for t in 0..ROLLOUT_HORIZON {
                let _ = grpo.sample_and_record(g, t, &phi, &mut rng);
            }
            grpo.group_returns[g] = -(g as f64);
        }
        grpo.finish_group_update(GrpoUpdateParams::default());
        assert!(grpo.last_surrogate_after.is_finite());
    });
}

#[test]
fn promoted_wardrop_and_braess_run_zero_alloc_via_math_explorer() {
    use math_explorer::applied::game_theory::equilibrium::{
        BraessWheatstoneParams, braess_avg_physical_latency, solve_braess_wheatstone_flows,
    };
    use math_explorer::pure_math::graph_theory::wardrop::{
        AffineLatency, FrankWolfeScratch, solve_wardrop_in_place,
    };

    let edges = [
        (0, 1, 0.0),
        (1, 3, 1.0),
        (0, 2, 1.0),
        (2, 3, 0.0),
        (1, 2, 0.0),
    ];
    let latencies = [
        AffineLatency::clamped(1.0, 0.0),
        AffineLatency::clamped(0.0, 1.0),
        AffineLatency::clamped(0.0, 1.0),
        AffineLatency::clamped(1.0, 0.0),
        AffineLatency::clamped(0.0, 0.0),
    ];
    let mut wardrop_scratch = FrankWolfeScratch::new();
    wardrop_scratch.prepare_topology(0..4, &edges, &latencies);

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        let mut flows = [0.0; 5];
        let mut lats = [0.0; 5];
        let w =
            solve_wardrop_in_place(&mut wardrop_scratch, 0, 3, 1.0, None, &mut flows, &mut lats)
                .expect("Wardrop converged");
        assert!((w.commuter_latency - 2.0).abs() < 1e-5);

        let params = BraessWheatstoneParams::default();
        let (wardrop, pigou, _) = solve_braess_wheatstone_flows(&params, 1.0);
        assert!(
            braess_avg_physical_latency(wardrop, &params)
                > braess_avg_physical_latency(pigou, &params)
        );
    });
}

#[test]
fn promoted_parrondo_and_auction_run_zero_alloc_via_math_explorer() {
    use math_explorer::applied::game_theory::mechanism_design::{
        AuctionDistEval, AuctionDistributionKind, AuctionEvalInputs, AuctionEvalWorkspace,
        AuctionRuleMode,
    };
    use math_explorer::pure_math::statistics::markov::parrondo::{
        ParrondoSchedule, solve_parrondo_ratchet_stack,
    };

    let sched = ParrondoSchedule::abb();
    let mut auction_ws = AuctionEvalWorkspace::default();
    let dist =
        AuctionDistEval::from_params(AuctionDistributionKind::Uniform, 0.0, 100.0, 0.1, 2.0, 2.0);

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        let par = solve_parrondo_ratchet_stack(3, 0.5, 0.1, 0.75, 0.005, sched.steps())
            .expect("Parrondo stack summary");
        assert!(par.is_paradox);

        let auc = auction_ws.evaluate_analytical(
            &dist,
            AuctionEvalInputs {
                rule: AuctionRuleMode::FirstPrice,
                reserve: 50.0,
                n_bidders: 2,
                focal_v: 100.0,
                candidate_bid: 62.5,
                fixed_point_tol: 1e-4,
            },
        );
        assert!(auc.is_equilibrium_fixed_point);
    });
}

#[test]
fn promoted_relativity_and_chaos_run_zero_alloc_via_math_explorer() {
    use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
    use math_explorer::physics::high_energy::{
        GeodesicKind, KerrEquatorialMetric, KerrRadialOrbitParams, eval_kerr_lensing_cell,
        kerr_radial_rk4_step,
    };
    use math_explorer::pure_math::analysis::ode::RungeKutta4;

    let mut lorenz = LorenzBuilder::new().build(LorenzState::new(1.0, 1.0, 1.0));
    let mut rk4 = RungeKutta4::new(&lorenz.state.vec);
    let kerr = KerrEquatorialMetric::new(2.0, 0.8).expect("valid Kerr");

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        lorenz.step_with(&mut rk4, 0.01);
        assert!(lorenz.state.vec.norm() > 0.0);

        let dil_circ = kerr.worldline_clock_dilation(6.0, 0.0, 0.02).unwrap();
        let dil_rad = kerr.worldline_clock_dilation(6.0, 0.15, 0.02).unwrap();
        assert!(dil_rad < dil_circ);

        let (dr, dvr, dphi) = kerr_radial_rk4_step(
            KerrRadialOrbitParams {
                rs: 2.0,
                spin_a: 0.8,
                energy: 1.0,
                ell: 4.5,
                kappa: 0.0,
            },
            8.0,
            -0.05,
            0.05,
        );
        assert!(dr.is_finite() && dvr.is_finite() && dphi.is_finite());
        let cell = eval_kerr_lensing_cell(GeodesicKind::Null, -0.85, 2.0, 0.6, 8.0, 5.5);
        assert!(cell.primary.is_finite());
    });
}

#[test]
fn promoted_mri_quantum_ising_and_morphogenesis_run_zero_alloc_via_math_explorer() {
    use math_explorer::biology::morphogenesis::{
        classify_gray_scott_dispersion, classify_schnakenberg_dispersion,
    };
    use math_explorer::physics::mri::reconstruction::{
        KSpaceReconWorkspace, ReconOutput, compute_forward_kspace,
    };
    use math_explorer::physics::quantum::exact_single_k_lattice_transmission;
    use math_explorer::physics::solid_state::{
        IsingAtlasRegime, eval_ising_order_parameter, onsager_critical_temp,
    };

    let gt = vec![1.0_f64; 16 * 16];
    let fwd = compute_forward_kspace(&gt);
    let mask = vec![true; 16 * 16];
    let mut mri_ws = KSpaceReconWorkspace::new(16);
    let mut mri_out = ReconOutput::with_dim(16);
    let free_pot = [0.0_f64; 32];

    let _guard = ALLOC_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        mri_ws.reconstruct_from_forward_cache_into(&gt, &fwd, &mask, &mut mri_out);
        assert!(mri_out.psnr_db >= 99.0);

        let t_free = exact_single_k_lattice_transmission(&free_pot, 0.1, 2.0);
        assert!((t_free - 1.0).abs() < 1e-12);

        let tc = onsager_critical_temp(1.0);
        let (reg, mag, _) = eval_ising_order_parameter(1.0, 1.0, 0.0);
        assert!(tc > 2.2 && mag > 0.9 && reg == IsingAtlasRegime::FerroOrSpatialReciprocity);

        let gs = classify_gray_scott_dispersion(0.037, 0.06, 0.16, 0.08, 1.0);
        let sc = classify_schnakenberg_dispersion(0.1, 0.9, 1.0, 40.0, 0.005);
        assert!(gs.primary.is_finite() && sc.primary.is_finite());
    });
}
