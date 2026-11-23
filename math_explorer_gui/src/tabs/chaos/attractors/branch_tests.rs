#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::async_sim::SimulationRunner;
use crate::framework::timeline_branch::exports::assert_framework_exports_all_branch_formats;
use crate::framework::{InteractiveTool, TimelineBranchAction, MAX_TIMELINE_RING_BYTES};
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;

#[test]
fn test_attractor_ring_buffer_scrub_without_reset_and_zero_perturbation_100_steps() {
    let mut plotter = AttractorPlotter {
        simulation_speed: 1,
        ..Default::default()
    };
    assert!(plotter.sync_branch.timeline.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);

    for _ in 0..30 {
        plotter.step_simulation();
    }
    let state_at_30 = plotter.system.state.vec;
    let state_at_15 = plotter.history[14];

    plotter.scrub_to_step(15);
    assert_eq!(plotter.step_count, 15);
    assert_eq!(plotter.history.len(), 15);
    assert!(
        (plotter.system.state.vec - state_at_15).norm() < 1.0e-12,
        "Scrubbing to k=15 must restore exact 3D coordinates without resetting to (1,1,1)"
    );
    assert!((plotter.system.state.vec - state_at_30).norm() > 1.0e-4);

    // Fork at k=15 with zero perturbation -> identically 0.0 divergence over 100 steps
    clear_test_announcements();
    let ctx = egui::Context::default();
    plotter.apply_timeline_action(TimelineBranchAction::ToggleForkBranch, Some(&ctx));
    assert!(all_announcements()
        .iter()
        .any(|(msg, pri)| pri == "polite" && msg.contains("Forked")));
    for _ in 0..100 {
        plotter.step_simulation();
    }
    let zero_div = plotter.branch_divergence_metrics().expect("Fork active");
    assert_eq!(zero_div.steps_since_fork, 100);
    assert_eq!(zero_div.instantaneous_l2, 0.0);
    assert_eq!(zero_div.peak_abs_delta, 0.0);
    assert_eq!(zero_div.cumulative_area_delta, 0.0);
}

#[test]
fn test_attractor_butterfly_perturbation_3d_overlay_exports_and_worker_parity() {
    let mut plotter = AttractorPlotter {
        simulation_speed: 1,
        ..Default::default()
    };
    for _ in 0..1400 {
        plotter.step_simulation();
    }
    plotter.fork_branch();
    plotter.apply_butterfly_perturbation();
    for _ in 0..1000 {
        plotter.step_simulation();
    }

    let div = plotter.branch_divergence_metrics().expect("Fork active");
    assert!(div.instantaneous_l2 > 1.0e-3);
    assert!(div.peak_abs_delta > 1.0e-2);
    assert!(div.log_divergence_slope > 0.2);

    plotter.populate_scene(egui::Color32::CYAN);
    let ds3 = plotter.export_datasets_3d();
    assert!(ds3.iter().any(|(n, _)| n == "Trajectory"));
    assert!(ds3.iter().any(|(n, _)| n.starts_with("Shadow Baseline")));
    assert!(ds3.iter().any(|(n, _)| n.starts_with("Branch Metadata")));

    let saved = plotter.save_state().expect("save_state");
    assert!(saved.len() <= URL_JSON_BOUNDS.max_bytes && !saved.contains("Shadow Baseline"));

    let params = Arc::clone(&plotter.params);
    params.write().unwrap().simulation_speed = 1;
    let mut runner = AttractorRunner::from_params(params);
    for _ in 0..1400 {
        runner.step();
    }
    runner.process_command(SimCommand::ForkBranch);
    runner.process_command(SimCommand::Custom("PerturbDeltaX:1e-5".to_string()));
    for _ in 0..1000 {
        runner.step();
    }
    let snap = runner.get_snapshot();
    let r_data = snap
        .structured_data
        .as_ref()
        .unwrap()
        .downcast_ref::<AttractorSnapshot>()
        .unwrap();
    assert!((r_data.branch.divergence.instantaneous_l2 - div.instantaneous_l2).abs() < 1.0e-12);
    assert_framework_exports_all_branch_formats(Box::new(plotter), "Shadow Baseline");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_attractor_lockstep_shadow_twin_step_zero_heap_allocations() {
    let mut sys = LorenzBuilder::new().build(LorenzState::new(1.0, 1.0, 1.0));
    let mut branch = branch::AttractorBranchRuntime::default();
    let seed_hist = VecDeque::from([Vector3::new(1.0, 1.0, 1.0)]);
    branch.fork_at(0, &sys, AttractorMode::Lorenz, None, 0.01, &seed_hist);
    sys.state.vec.x += 1.0e-5;

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let ((l2, peak), _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
        sys.step(0.01);
        branch.on_step_in_place(1, 0.01, 2000, sys.state.vec)
    });
    assert!(l2 > 0.0 && peak > 0.0);
}
