#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use math_explorer::pure_math::analysis::ode::RungeKutta4;
use std::sync::atomic::AtomicU64;
use std::sync::RwLock;
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_evaluate_attractor_metrics_in_place_zero_allocations_and_signs() {
    let chaotic = LorenzBuilder::new()
        .sigma(10.0)
        .rho(28.0)
        .beta(8.0 / 3.0)
        .build(LorenzState::new(1.0, 1.0, 1.0));
    let stable = LorenzBuilder::new()
        .sigma(10.0)
        .rho(10.0)
        .beta(8.0 / 3.0)
        .build(LorenzState::new(1.0, 1.0, 1.0));

    let mut history = VecDeque::with_capacity(32);
    let mut sim = chaotic.clone();
    for _ in 0..32 {
        sim.step(0.01);
        history.push_back(sim.state.vec);
    }

    let mut solver = RungeKutta4::new(&Vector3::zeros());
    let mut scratch = [Vector3::zeros(); 64];

    let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((lam_chaos, corr_chaos, lam_stable), _) = VerifiedEngine::run_verified(|| {
        let (lc, cc) = evaluate_attractor_metrics_in_place(
            &chaotic,
            0.01,
            &history,
            &mut solver,
            &mut scratch,
        );
        let (ls, _) =
            evaluate_attractor_metrics_in_place(&stable, 0.01, &history, &mut solver, &mut scratch);
        (lc, cc, ls)
    });

    assert!(lam_chaos.expect("chaotic lyapunov") > 0.0);
    assert!(lam_stable.expect("stable lyapunov") < 0.0);
    let c_val = corr_chaos.expect("correlation sum");
    assert!((0.0..=1.0).contains(&c_val));
}

#[test]
fn test_diverged_attractor_halts_lyapunov_evaluation() {
    let mut plotter = AttractorPlotter::default();
    assert!(plotter.latest_lyapunov.is_some());

    plotter.system.state.vec.x = f64::NAN;
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| plotter.show(c));

    assert!(plotter.diverged());
    assert!(plotter.latest_lyapunov.is_none());
    assert!(plotter.latest_correlation_sum.is_none());
}

#[test]
fn test_lorenz_regime_map_progressive_strides_and_teleport() {
    let req = LorenzRegimeRequest {
        generation: 1,
        beta: 8.0 / 3.0,
        width: 12,
        height: 8,
        high_contrast: false,
    };
    let shared = Arc::new(RwLock::new(req));
    let latest = Arc::new(AtomicU64::new(1));
    let mut runner = LorenzRegimeMapRunner::new(Arc::clone(&shared), Arc::clone(&latest));

    let mut strides = Vec::new();
    for _ in 0..128 {
        runner.step();
        let snap = runner.get_snapshot();
        let meta = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<LorenzRegimeSnapshot>())
            .expect("LorenzRegimeSnapshot expected");
        if strides.last().copied() != Some(meta.stride) {
            strides.push(meta.stride);
        }
        if meta.is_final {
            break;
        }
    }
    assert_eq!(strides, vec![4, 2, 1]);
    assert!(runner.is_complete());

    let final_snap = runner.get_snapshot();
    let meta = final_snap
        .structured_data
        .as_ref()
        .and_then(|d| d.downcast_ref::<LorenzRegimeSnapshot>())
        .unwrap();
    assert!(meta.lambdas.iter().all(|v| v.is_finite()));
    // Top-right corner (rho=100, sigma=50) vs bottom-left (rho=0, sigma=0)
    assert!(meta.lambdas[0] <= 0.02, "rho=0 cell must not be chaotic");

    let mut plotter = AttractorPlotter::default();
    plotter.step_simulation();
    assert!(!plotter.history.is_empty());

    plotter.teleport_regime(14.0, 10.0);
    assert!((plotter.system.rho - 14.0).abs() < 1e-9);
    assert!((plotter.system.sigma - 10.0).abs() < 1e-9);
    assert!(plotter.history.is_empty());
    assert!(plotter.latest_lyapunov.expect("updated lyapunov") < 0.0);
}
