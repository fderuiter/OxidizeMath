#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::counterfactual::{render_morphogenesis_heatmap, MorphogenesisHeatmapMode};
use super::*;
use crate::app::ThemeMode;
use crate::async_sim::SimulationRunner;
use crate::framework::{
    BranchControlAction, ExportFormat, ExportStatusSeverity, InteractiveTool, SimulationFramework,
};
use crate::tabs::mri::MriTool;
use std::collections::HashSet;

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
fn test_mri_1000_step_parity_and_spin_echo_180_refocusing_gain() {
    let mut mri_parity = MriTool::default();
    mri_parity.step_simulation();
    assert!(mri_parity.apply_counterfactual_action(BranchControlAction::SetPerturbation(0.0)));
    assert!(mri_parity.apply_counterfactual_action(BranchControlAction::ForkHere));
    for _ in 0..1_000 {
        mri_parity.step_simulation();
    }
    let p_snap = mri_parity.counterfactual_snapshot().expect("MRI parity");
    assert_eq!(p_snap.l2_separation, 0.0);
    assert_eq!(p_snap.peak_separation, 0.0);

    let mut echo_tool = MriTool::default();
    echo_tool
        .import_preset_json(crate::starter_gallery::PRESET_MRI_SPIN_ECHO_REFOCUSING)
        .expect("spin-echo preset loads");
    let echo_snap = echo_tool
        .counterfactual_snapshot()
        .expect("spin-echo snapshot");
    let echo_gain = echo_snap.primary_delta.as_ref().unwrap().raw_value;
    assert!(
        echo_gain > 0.15,
        "Expected positive ΔM_xy Hahn spin-echo recovery gain at 2*tau, got {echo_gain}"
    );
    assert!(echo_tool.swap_branches());
    let swapped_gain = echo_tool
        .counterfactual_snapshot()
        .unwrap()
        .primary_delta
        .unwrap()
        .raw_value;
    assert!((swapped_gain + echo_gain).abs() < 1e-9);
    assert!(echo_tool.clear_fork());
    assert!(echo_tool.counterfactual_snapshot().is_none());
}

#[test]
fn test_mri_t2_intervention_zero_alloc_and_twin_exports() {
    let mut fw = SimulationFramework::new("mri");
    fw.apply_starter_card(
        "MRI Spin-Echo Refocusing",
        Some("MriTool"),
        Some(crate::starter_gallery::PRESET_MRI_SPIN_ECHO_REFOCUSING),
    )
    .expect("MRI preset applies");

    run_framework_export(&mut fw, "mri", ExportFormat::Csv);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(csv.contains("Branch B") && csv.contains("Branch A"));

    run_framework_export(&mut fw, "mri", ExportFormat::Png);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(
        fw.export_queue
            .last_exported_bytes
            .as_deref()
            .unwrap()
            .len()
            > 1_000
    );

    let tool = MriTool::default();
    let mut runner = crate::tabs::mri::MriRunner::new(
        std::sync::Arc::clone(&tool.params),
        nalgebra::Vector3::new(1.0, 0.0, 0.0),
        1.0,
    );
    runner.step_once();
    runner.process_command(crate::async_sim::SimCommand::Custom("refocus_180".into()));
    assert!(runner.has_counterfactual_twin());
    runner.step();

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        runner.step_branch_a();
        runner.step_branch_b();
    });
}

#[test]
fn test_morphogenesis_1000_step_zero_perturbation_parity_and_zero_alloc() {
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 1.0,
        d_v: 30.0,
        dt: 0.005,
        high_contrast: false,
        theme_mode: ThemeMode::Dark,
        heatmap_mode: MorphogenesisHeatmapMode::ActiveUb,
    }));
    let mut runner = MorphogenesisRunner::new_seeded(&params, 24, 24);
    for _ in 0..10 {
        runner.step();
    }
    assert!(runner.fork_here(0.0));
    assert_eq!(runner.rng, runner.twin.as_ref().unwrap().rng_a);

    for _ in 0..1_000 {
        runner.step();
    }
    let twin = runner.twin.as_ref().expect("twin active");
    assert_eq!(twin.l2_norm, 0.0);
    assert_eq!(twin.linf_norm, 0.0);
    assert_eq!(twin.tracker.peak_separation(), 0.0);

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        runner.step_branch_a();
        runner.step_branch_b();
    });
}

#[test]
fn test_morphogenesis_heatmap_modes_across_themes_and_norms() {
    let mut tool = MorphogenesisTool::default();
    tool.import_preset_json(crate::starter_gallery::PRESET_MORPHOGENESIS_INHIBITOR_FORK)
        .expect("morphogenesis inhibitor fork preset loads");
    assert!(tool.is_forked());
    let snap = tool
        .counterfactual_snapshot()
        .expect("morphogenesis snapshot");
    assert!(snap.l2_separation > 0.0 && snap.l2_separation.is_finite());
    let linf = snap.primary_delta.as_ref().unwrap().raw_value;
    assert!(linf > 0.0 && linf.is_finite());

    let ua = tool.local_runner.twin.as_ref().unwrap().system_a.u();
    let ub = tool.local_runner.system.u();
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        for mode in MorphogenesisHeatmapMode::ALL {
            let img =
                render_morphogenesis_heatmap(ub, Some(ua), tool.width, tool.height, mode, theme);
            assert_eq!(img.size, [tool.width, tool.height]);
            let distinct: HashSet<egui::Color32> = img.pixels.iter().copied().collect();
            assert!(
                distinct.len() >= 3,
                "Expected non-trivial spatial heatmap for {mode:?} in {theme:?}"
            );
        }
    }

    assert!(tool.swap_branches());
    assert!((tool.params.read().unwrap().d_v - 40.0).abs() < 1e-12);
    assert!(tool.clear_fork());
    assert!(!tool.is_forked());
    tool.reset();
    assert_eq!(tool.local_runner.step_count, 0);
}

#[test]
fn test_morphogenesis_twin_exports_and_finite_state_guard_rollback() {
    let mut fw = SimulationFramework::new("biology");
    fw.apply_starter_card(
        "Morphogenesis Inhibitor Fork",
        Some("MorphogenesisTool"),
        Some(crate::starter_gallery::PRESET_MORPHOGENESIS_INHIBITOR_FORK),
    )
    .expect("morphogenesis fork preset applies");
    assert!(fw.branch_engine.is_forked());

    run_framework_export(&mut fw, "biology", ExportFormat::Csv);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let csv = fw.export_queue.last_exported_content.as_deref().unwrap();
    assert!(csv.contains("Activator u_B") && csv.contains("Activator u_A"));

    run_framework_export(&mut fw, "biology", ExportFormat::Png);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(
        fw.export_queue
            .last_exported_bytes
            .as_deref()
            .unwrap()
            .len()
            > 500
    );

    let pre_fault_state = fw.active_tool.as_ref().unwrap().save_state().unwrap();
    assert!(fw.dispatch_branch_action(BranchControlAction::ClearFork));
    assert!(!fw.branch_engine.is_forked());
    assert!(fw.rollback_step_back());
    assert!(fw.branch_engine.is_forked());
    assert_eq!(
        fw.active_tool.as_ref().unwrap().save_state().as_deref(),
        Some(pre_fault_state.as_str())
    );
}

#[test]
fn test_morphogenesis_sculpted_maze_twin_parity_and_zero_alloc() {
    let params = Arc::new(std::sync::RwLock::new(MorphogenesisSharedParams {
        a: 0.1,
        b: 0.9,
        d_u: 1.0,
        d_v: 30.0,
        dt: 0.005,
        high_contrast: false,
        theme_mode: ThemeMode::Dark,
        heatmap_mode: MorphogenesisHeatmapMode::ActiveUb,
    }));
    let mut runner = MorphogenesisRunner::new_seeded(&params, 32, 32);
    runner.process_command(crate::async_sim::SimCommand::Custom("stamp_maze".into()));
    assert!(runner.system.diffusion.mask.has_active_cells());
    for _ in 0..8 {
        runner.step();
    }
    assert!(runner.fork_here(0.0));
    assert!(runner
        .twin
        .as_ref()
        .unwrap()
        .system_a
        .diffusion
        .mask
        .has_active_cells());
    for _ in 0..100 {
        runner.step();
    }
    let twin = runner.twin.as_ref().expect("masked twin active");
    assert_eq!(twin.l2_norm, 0.0);
    assert_eq!(twin.linf_norm, 0.0);
    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    verified_engine::engine::VerifiedEngine::run_verified(|| {
        runner.step_branch_a();
        runner.step_branch_b();
    });
}

#[test]
fn test_morphogenesis_fork_and_brush_mask_state_roundtrip() {
    let mut tool = MorphogenesisTool::default();
    tool.dispatch_brush_stroke(
        crate::framework::BrushStroke {
            start: [50.0, 50.0],
            end: [55.0, 55.0],
            radius: 3.0,
            mode: crate::framework::BrushMode::ZeroFluxWall,
            drag_vector: [5.0, 5.0],
        },
        None,
    );
    tool.step_simulation();
    assert!(tool.apply_counterfactual_action(BranchControlAction::ForkHere));
    tool.params.write().unwrap().d_v = 45.0;
    tool.step_simulation();
    let saved = tool.save_state().expect("fork + mask state serializes");
    let mut restored = MorphogenesisTool::default();
    restored
        .try_load_state(&saved)
        .expect("fork + mask state restores");
    assert!(restored.is_forked());
    assert!(!restored.mask_rle.read().unwrap().is_empty());
    assert!(restored
        .local_runner
        .system
        .diffusion
        .mask
        .has_active_cells());
    assert!(restored.counterfactual_snapshot().unwrap().l2_separation > 0.0);
}
