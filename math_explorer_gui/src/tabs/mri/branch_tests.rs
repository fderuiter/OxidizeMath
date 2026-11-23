#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::{InteractiveTool, TimelineBranchAction, MAX_TIMELINE_RING_BYTES};
use crate::url_state::URL_JSON_BOUNDS;
use eframe::egui;
use nalgebra::Vector3;
use std::sync::Arc;

#[test]
fn test_mri_ring_buffer_scrub_without_reset_and_zero_perturbation_100_steps() {
    let mut tool = MriTool::default();
    assert!(tool.sync_branch.timeline.memory_budget_bytes() <= MAX_TIMELINE_RING_BYTES);

    for _ in 0..30 {
        tool.step_simulation();
    }
    let m_at_30 = tool.simulator.magnetization;
    let (t_15, m_15) = tool.history[15];

    tool.scrub_to_step(15);
    assert!((tool.current_time - t_15).abs() < 1.0e-12);
    assert!(
        (tool.simulator.magnetization - m_15).norm() < 1.0e-12,
        "Scrubbing to k=15 must restore exact [Mx, My, Mz] without resetting to (1, 0, 0)"
    );
    assert!((tool.simulator.magnetization - m_at_30).norm() > 1.0e-4);

    // Fork at k=15 with zero perturbation -> identically 0.0 divergence over 100 steps
    clear_test_announcements();
    let ctx = egui::Context::default();
    tool.apply_timeline_action(TimelineBranchAction::ToggleForkBranch, Some(&ctx));
    assert!(all_announcements()
        .iter()
        .any(|(msg, pri)| pri == "polite" && msg.contains("Forked")));
    for _ in 0..100 {
        tool.step_simulation();
    }
    let zero_div = tool.branch_divergence_metrics().expect("Fork active");
    assert_eq!(zero_div.steps_since_fork, 100);
    assert_eq!(zero_div.instantaneous_l2, 0.0);
    assert_eq!(zero_div.peak_abs_delta, 0.0);
    assert_eq!(zero_div.cumulative_area_delta, 0.0);

    // Clear branch announces polite ARIA status
    clear_test_announcements();
    tool.apply_timeline_action(TimelineBranchAction::ClearBranch, Some(&ctx));
    assert!(tool.branch_divergence_metrics().is_none());
    assert!(all_announcements()
        .iter()
        .any(|(msg, pri)| pri == "polite" && msg.contains("Cleared")));
}

#[test]
fn test_mri_t2_shift_counterfactual_worker_parity_and_all_exports() {
    use crate::framework::timeline_branch::exports::assert_framework_exports_all_branch_formats;

    let mut tool = MriTool::default();
    for _ in 0..20 {
        tool.step_simulation();
    }
    tool.fork_branch();
    tool.simulator.t2 = 0.45;
    tool.simulator.b_field.x = 2.0e-6;
    for _ in 0..40 {
        tool.step_simulation();
    }
    let div = tool.branch_divergence_metrics().expect("Fork active");
    assert!(div.instantaneous_l2 > 1.0e-3 && div.cumulative_area_delta > 0.0);

    let saved = tool.save_state().expect("save_state");
    assert!(saved.len() <= URL_JSON_BOUNDS.max_bytes && !saved.contains("Shadow Mx"));

    let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.1,
        b_field: Vector3::new(0.0, 0.0, 1.0),
        time_scale: 1.0,
    }));
    let mut runner = MriRunner::new(Arc::clone(&params), Vector3::new(1.0, 0.0, 0.0), 1.0);
    for _ in 0..20 {
        runner.step_once();
    }
    runner.process_command(SimCommand::ForkBranch);
    {
        let mut p = params.write().unwrap();
        p.t2 = 0.45;
        p.b_field.x = 2.0e-6;
    }
    for _ in 0..40 {
        runner.step_once();
    }
    let snap = runner.get_snapshot();
    let r_data = snap
        .structured_data
        .as_ref()
        .unwrap()
        .downcast_ref::<MriSnapshot>()
        .unwrap();
    assert!(r_data.branch.fork_active && r_data.branch.divergence.instantaneous_l2 > 1.0e-4);
    assert_framework_exports_all_branch_formats(Box::new(tool), "Shadow Mx");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn test_mri_runner_forked_lockstep_step_zero_heap_allocations() {
    let params = Arc::new(std::sync::RwLock::new(MriSharedParams {
        t1: 1.0,
        t2: 0.15,
        b_field: Vector3::new(0.0, 0.0, 1.0),
        time_scale: 4.0,
    }));
    let mut runner = MriRunner::new(Arc::clone(&params), Vector3::new(1.0, 0.0, 0.0), 1.0);
    runner.step_once();
    runner.process_command(SimCommand::ForkBranch);
    params.write().unwrap().t2 = 0.35;

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        runner.step();
    });
    assert!(runner.snapshot_branch().divergence.instantaneous_l2 > 0.0);
}
