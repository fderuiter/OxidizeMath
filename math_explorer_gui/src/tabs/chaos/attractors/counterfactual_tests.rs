#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::{
    BranchControlAction, ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework,
};
use crate::tabs::epidemiology::sir::{SirTool, SirTwinStepper};

fn run_framework_export(fw: &mut SimulationFramework, domain: &str, format: ExportFormat) {
    let ctx = egui::Context::default();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        fw.show_export_modal = true;
        fw.queue_export(format);
        fw.show(ctx, domain);
    });
}

#[test]
fn test_attractor_1000_step_zero_perturbation_parity_and_positive_lyapunov_slope() {
    let mut parity_tool = AttractorPlotter {
        simulation_speed: 1,
        ..Default::default()
    };
    for _ in 0..40 {
        parity_tool.step_simulation();
    }
    assert!(parity_tool.apply_counterfactual_action(BranchControlAction::SetPerturbation(0.0)));
    assert!(parity_tool.apply_counterfactual_action(BranchControlAction::ForkHere));
    for _ in 0..1_000 {
        parity_tool.step_simulation();
    }
    let parity_snap = parity_tool
        .counterfactual_snapshot()
        .expect("forked snapshot");
    assert_eq!(parity_snap.l2_separation, 0.0);
    assert_eq!(parity_snap.peak_separation, 0.0);

    let mut butterfly = AttractorPlotter {
        simulation_speed: 1,
        ..Default::default()
    };
    for _ in 0..150 {
        butterfly.step_simulation();
    }
    assert!(butterfly.apply_counterfactual_action(BranchControlAction::SetPerturbation(1e-5)));
    assert!(butterfly.apply_counterfactual_action(BranchControlAction::ForkHere));
    for _ in 0..2_600 {
        butterfly.step_simulation();
    }
    let div = butterfly
        .counterfactual_snapshot()
        .expect("butterfly snapshot");
    assert!(
        div.l2_separation > 1.0,
        "Expected macroscopic lobe separation, got {}",
        div.l2_separation
    );
    assert!(
        div.log_separation_slope > 0.2,
        "Expected positive Lyapunov slope, got {}",
        div.log_separation_slope
    );
    assert!(div.peak_separation >= div.l2_separation);
}

#[test]
fn test_attractor_swap_clear_reset_zero_alloc_and_twin_exports() {
    let mut fw = SimulationFramework::new("chaos");
    fw.apply_starter_card(
        "Lorenz Butterfly Perturbation",
        Some("AttractorPlotter"),
        Some(crate::starter_gallery::PRESET_LORENZ_BUTTERFLY_PERTURBATION),
    )
    .expect("starter preset applies");

    let snap_before = fw
        .active_tool
        .as_ref()
        .and_then(|t| t.counterfactual_snapshot())
        .expect("preset starts forked");
    assert_eq!(snap_before.fork_step, 150);
    assert!(snap_before.l2_separation > 0.5);

    run_framework_export(&mut fw, "chaos", ExportFormat::Csv);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(csv.contains("Branch B") && csv.contains("Branch A"));

    run_framework_export(&mut fw, "chaos", ExportFormat::Png);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let png = fw.export_queue.last_exported_bytes.as_deref().unwrap();
    assert!(png.len() > 1_000);

    assert!(fw.dispatch_branch_action(BranchControlAction::SwapBranches));
    assert!(fw.dispatch_branch_action(BranchControlAction::ClearFork));
    assert!(fw
        .active_tool
        .as_ref()
        .and_then(|t| t.counterfactual_snapshot())
        .is_none());
}

#[test]
fn test_attractor_and_sir_zero_allocation_verified_twin_step() {
    let plotter = AttractorPlotter::default();
    let mut runner = AttractorRunner {
        system: math_explorer::physics::chaos::lorenz::LorenzBuilder::new()
            .build(math_explorer::physics::chaos::lorenz::LorenzState::new(1.0, 1.0, 1.0)),
        params: std::sync::Arc::clone(&plotter.params),
        history: std::collections::VecDeque::with_capacity(ATTRACTOR_RING_CAPACITY),
        step_count: 0,
        execution_state: SolverExecutionState::Running,
        twin: None,
    };
    runner.step_once();
    runner.process_command(crate::async_sim::SimCommand::Custom("fork:0.00001".into()));
    assert!(runner.has_counterfactual_twin());
    runner.step();

    let base_sir = math_explorer::epidemiology::compartmental::SIRModel::new(1000.0, 1.0, 0.5, 0.1)
        .expect("valid SIR baseline");
    let mut sir_stepper = SirTwinStepper::fork_at(&base_sir, 20, 0.1, 1000.0, 0.2, 0.1, 0.0);
    sir_stepper.step_lockstep();

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        runner.step_branch_a();
        runner.step_branch_b();
        sir_stepper.step_lockstep();
    });
}

#[test]
fn test_sir_1000_step_parity_and_lockdown_timing_delta_ipeak() {
    let mut sir_parity = SirTool::default();
    sir_parity.duration = 120.0;
    sir_parity.recalculate();
    assert!(sir_parity.fork_at_day(15.0));
    sir_parity.cf.playhead_step = Some(150);
    sir_parity.step_lockstep(1_000);
    let p_snap = sir_parity
        .counterfactual_snapshot()
        .expect("SIR parity snapshot");
    assert_eq!(p_snap.l2_separation, 0.0);
    assert_eq!(p_snap.peak_separation, 0.0);

    let mut day10 = SirTool::default();
    day10.n = 1000.0;
    day10.i0 = 1.0;
    day10.beta = 0.5;
    day10.gamma = 0.1;
    day10.duration = 100.0;
    day10.recalculate();
    assert!(day10.fork_at_day(10.0));
    day10.beta = 0.2;
    day10.recalculate();
    let averted_day10 = day10
        .counterfactual_snapshot()
        .unwrap()
        .primary_delta
        .unwrap()
        .raw_value;

    let mut day17 = SirTool::default();
    day17.n = 1000.0;
    day17.i0 = 1.0;
    day17.beta = 0.5;
    day17.gamma = 0.1;
    day17.duration = 100.0;
    day17.recalculate();
    assert!(day17.fork_at_day(17.0));
    day17.beta = 0.2;
    day17.recalculate();
    let averted_day17 = day17
        .counterfactual_snapshot()
        .unwrap()
        .primary_delta
        .unwrap()
        .raw_value;

    assert!(averted_day10 > 200.0, "Day-10 lockdown must avert substantial peak infections, got {averted_day10}");
    assert!(
        averted_day10 > averted_day17,
        "Day-10 intervention ({averted_day10}) must avert more peak infections than day-17 ({averted_day17})"
    );
}

#[test]
fn test_sir_swap_clear_reset_and_exports() {
    let mut fw = SimulationFramework::new("epidemiology");
    fw.apply_starter_card(
        "SIR Lockdown Timing",
        Some("SirTool"),
        Some(crate::starter_gallery::PRESET_SIR_LOCKDOWN_TIMING),
    )
    .expect("SIR lockdown preset applies");

    run_framework_export(&mut fw, "epidemiology", ExportFormat::Csv);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(csv.contains("Infected (Branch B)") && csv.contains("Infected (Branch A)"));

    run_framework_export(&mut fw, "epidemiology", ExportFormat::Png);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(fw.export_queue.last_exported_bytes.as_deref().unwrap().len() > 1_000);

    assert!(fw.dispatch_branch_action(BranchControlAction::SwapBranches));
    let swapped = fw
        .active_tool
        .as_ref()
        .and_then(|t| t.counterfactual_snapshot())
        .unwrap();
    assert!(swapped.primary_delta.unwrap().raw_value < 0.0);

    assert!(fw.dispatch_branch_action(BranchControlAction::ClearFork));
    assert!(fw
        .active_tool
        .as_ref()
        .and_then(|t| t.counterfactual_snapshot())
        .is_none());
}
