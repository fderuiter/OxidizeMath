#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{LoraHubSimplexTool, LoraTaskPreset};
use crate::framework::InteractiveTool;

#[test]
fn lorahub_simplex_tool_roundtrip_and_stepping() {
    let mut tool = LoraHubSimplexTool::default();
    assert!(tool.has_finite_parameters());
    let init_obj = tool.simplex.best_value();

    for _ in 0..35 {
        tool.step_simulation();
    }
    assert!(
        tool.simplex.best_value() < init_obj,
        "Nelder-Mead stepping should decrease LoraHub objective"
    );
    assert_eq!(tool.merged_matrix.nrows(), 6);
    assert_eq!(tool.merged_matrix.ncols(), 6);

    tool.alpha = 0.75;
    tool.task_preset = LoraTaskPreset::CrossLingualSummary;
    let saved = tool.save_state().expect("serializes");
    assert!(tool.state_bounds().validate_json(&saved).is_ok());

    let mut restored = LoraHubSimplexTool::default();
    restored.try_load_state(&saved).expect("loads valid state");
    assert!((restored.alpha - 0.75).abs() < 1e-9);
    assert_eq!(restored.task_preset, LoraTaskPreset::CrossLingualSummary);

    assert!(restored
        .try_load_state(r#"{"alpha": -1.0, "initial_step": 0.4, "weight_bound": 1.5}"#)
        .is_err());
    assert!(restored
        .try_load_state(r#"{"alpha": 0.2, "extra": true}"#)
        .is_err());
}

#[test]
fn lorahub_simplex_tool_renders_headless_and_exports_datasets() {
    let mut tool = LoraHubSimplexTool::default();
    let datasets = tool.export_datasets();
    assert!(datasets.len() >= 3);
    assert!(datasets.iter().all(|(_, pts)| !pts.is_empty()));

    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
}
