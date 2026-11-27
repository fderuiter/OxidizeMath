//! Analytical Jacobian parity, therapeutic window retention, cold-start safety, and zero-alloc tests for `MpcController`.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::controllers::mpc::PkMpcModel;
use super::state::{ClosedLoopState, ControllerKind, PlantKind};
use super::ClosedLoopControlTool;
use engine_opt::ilqr::ControlModel;

#[test]
fn test_pk_mpc_model_analytical_jacobians_match_central_finite_differences() {
    let model = PkMpcModel::default();
    let x0 = [38.5_f64, 2.05_f64, 0.145_f64];
    let u0 = [24.0_f64];
    let mut x_next = [0.0; 3];
    let mut fx = [0.0; 9];
    let mut fu = [0.0; 3];
    model
        .step_and_linearize(0, &x0, &u0, &mut x_next, &mut fx, &mut fu)
        .expect("PkMpcModel linearization succeeds");

    let eps = 1.0e-6;
    let mut xp = [0.0; 3];
    let mut xm = [0.0; 3];
    let mut dfx = [0.0; 9];
    let mut dfu = [0.0; 3];

    for col in 0..3 {
        let (mut x_plus, mut x_minus) = (x0, x0);
        x_plus[col] += eps;
        x_minus[col] -= eps;
        model
            .step_and_linearize(0, &x_plus, &u0, &mut xp, &mut dfx, &mut dfu)
            .unwrap();
        model
            .step_and_linearize(0, &x_minus, &u0, &mut xm, &mut dfx, &mut dfu)
            .unwrap();
        for row in 0..3 {
            let num = (xp[row] - xm[row]) / (2.0 * eps);
            assert!((fx[row * 3 + col] - num).abs() < 1.0e-8);
        }
    }

    model
        .step_and_linearize(0, &x0, &[u0[0] + eps], &mut xp, &mut dfx, &mut dfu)
        .unwrap();
    model
        .step_and_linearize(0, &x0, &[u0[0] - eps], &mut xm, &mut dfx, &mut dfu)
        .unwrap();
    for row in 0..3 {
        let num = (xp[row] - xm[row]) / (2.0 * eps);
        assert!((fu[row] - num).abs() < 1.0e-8);
    }
}

#[test]
fn test_mpc_controller_holds_pk_twin_in_therapeutic_window_without_cold_start_overdose_zero_alloc()
{
    let cfg = ClosedLoopState {
        plant: PlantKind::Pharmacokinetics,
        controller: ControllerKind::MpcPolicy,
        actuation_budget: 1.35,
        sensor_noise: 0.20,
        process_noise: 0.025,
        pk_mec: 1.2,
        pk_mtc: 3.2,
        kick_magnitude: 0.40,
        warmup_episodes: 0,
        ..ClosedLoopState::default()
    };
    let mut tool = ClosedLoopControlTool {
        pipeline: super::pipeline::ClosedLoopPipeline::new(&cfg),
        state: cfg,
        ..ClosedLoopControlTool::default()
    };
    tool.pipeline.reset_live_traces_only(&tool.state);

    let total_steps = 100usize;
    let (in_window_count, max_cp, max_open_cp) = {
        let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let ((in_win, peak_cp, peak_open), _) =
            verified_engine::engine::VerifiedEngine::run_verified(|| {
                let (mut inside, mut c_max, mut open_max) = (0usize, 0.0_f64, 0.0_f64);
                for step in 0..total_steps {
                    let out = match step {
                        28 => tool.apply_kick(0.55, 0.15),
                        56 => tool.apply_kick(-0.55, 0.0),
                        80 => tool.apply_kick(0.45, 0.10),
                        _ => tool.step_closed_loop(),
                    };
                    c_max = c_max.max(out.controlled_primary);
                    open_max = open_max.max(out.uncontrolled_primary);
                    if (tool.state.pk_mec..=tool.state.pk_mtc).contains(&out.controlled_primary) {
                        inside += 1;
                    }
                }
                (inside, c_max, open_max)
            });
        (in_win, peak_cp, peak_open)
    };

    let retention_ratio = (in_window_count as f64) / (total_steps as f64);
    assert!(max_open_cp > tool.state.pk_mtc + 0.8);
    assert!(
        max_cp <= tool.state.pk_mtc,
        "MpcController must avoid cold-start and post-kick MTC overdoses (max_cp={max_cp})"
    );
    assert!(
        retention_ratio >= 0.95,
        "MpcController window retention {:.1}% < 95%",
        retention_ratio * 100.0
    );
}
