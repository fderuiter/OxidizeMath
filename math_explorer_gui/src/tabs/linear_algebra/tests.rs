#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{PRESET_LINALG_HILBERT_RIDGE, PRESET_LINALG_LAUCHLI, PRESET_LINALG_NNLS};
use super::state::{ConditioningWorkbenchTool, MAX_COLS, MAX_ROWS};
use super::LinearAlgebraTab;
use crate::framework::{InteractiveTool, SimulationFramework};
use crate::tabs::ExplorerTab;
use eframe::egui;

#[test]
fn test_linear_algebra_tab_domain_slug_and_tool_selection() {
    let mut tab = LinearAlgebraTab::default();
    assert_eq!(tab.name(), "Linear Algebra");
    assert_eq!(tab.academic_domain(), "Pure Math");
    assert_eq!(tab.url_slug(), "linear_algebra");
    assert_eq!(tab.active_tool_id(), Some("LinearAlgebraConditioningTool"));
    assert!(tab.select_tool_by_name("Conditioning & Least-Squares Workbench"));
}

#[test]
fn test_workbench_lauchli_qr_outperforms_normal_equations_and_renders() {
    let mut tool = ConditioningWorkbenchTool::default();
    tool.import_preset_json(PRESET_LINALG_LAUCHLI).unwrap();
    let normal_err = tool.snapshot.solvers[0].forward_rel_error;
    let qr_err = tool.snapshot.solvers[1].forward_rel_error;
    let svd_err = tool.snapshot.solvers[2].forward_rel_error;
    assert!(
        qr_err < 1e-6 && svd_err < 1e-6 && qr_err * 1e4 < normal_err,
        "QR ({qr_err:e}) and SVD ({svd_err:e}) must beat explicit normal equations ({normal_err:e})"
    );

    tool.import_preset_json(PRESET_LINALG_HILBERT_RIDGE)
        .unwrap();
    assert!(tool.snapshot.cholesky_spd);
    let ridge_err = tool.snapshot.solvers[3].forward_rel_error;
    let unreg_err = tool.snapshot.solvers[1].forward_rel_error;
    assert!(ridge_err < unreg_err);

    tool.import_preset_json(PRESET_LINALG_NNLS).unwrap();
    assert!(tool.snapshot.solvers[4].min_entry >= 0.0);

    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let raw = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| tool.show(ctx));
    assert!(out.platform_output.accesskit_update.is_some());

    let png = egui_plot::export::export_custom_png(
        &tool.export_datasets(),
        &egui_plot::export::ExportOptions::default(),
    )
    .unwrap();
    assert!(!png.is_empty());
}

#[test]
fn test_workbench_dimension_bounds_and_finite_state_rollback() {
    let mut tool = ConditioningWorkbenchTool::default();
    let baseline = tool.save_state().unwrap();

    assert!(tool
        .try_load_state(&format!(
            r#"{{"perturbation_log10":-8.0,"noise_std":0.0,"ridge_lambda_log10":-6.0,"rows":{},"cols":4,"truncation_rank":2,"family":"Lauchli"}}"#,
            MAX_ROWS + 1
        ))
        .is_err());
    assert!(tool
        .try_load_state(&format!(
            r#"{{"perturbation_log10":-8.0,"noise_std":0.0,"ridge_lambda_log10":-6.0,"rows":6,"cols":{},"truncation_rank":2,"family":"Lauchli"}}"#,
            MAX_COLS + 1
        ))
        .is_err());
    assert_eq!(tool.save_state().unwrap(), baseline);

    let mut framework = SimulationFramework::new("linear_algebra");
    framework.cache_valid_snapshot();
    framework
        .active_tool
        .as_mut()
        .unwrap()
        .as_any_mut()
        .unwrap()
        .downcast_mut::<ConditioningWorkbenchTool>()
        .unwrap()
        .config
        .perturbation_log10 = f64::NAN;
    assert!(framework.inspect_runtime_state().is_err());
    assert!(framework.rollback_step_back());
    assert!(!framework.is_diverged());
    assert_eq!(
        framework
            .active_tool
            .as_ref()
            .unwrap()
            .save_state()
            .unwrap(),
        baseline
    );
}

// [cite:advanced_linear_algebra]
