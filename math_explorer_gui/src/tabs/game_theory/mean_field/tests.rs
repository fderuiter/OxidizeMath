#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_mean_field_game_tool_cfl_stability_presets_and_render() {
    let mut tool = MeanFieldGameTool::default();
    assert!(tool.peak_terminal_density.is_finite() && tool.peak_terminal_density > 0.0);
    assert!(tool.origin_initial_value.is_finite());

    tool.import_preset_json(presets::PRESET_HIGH_CONGESTION)
        .expect("High congestion preset loads");
    let dispersed_var = tool.terminal_variance;
    assert!(dispersed_var.is_finite() && dispersed_var > 0.0);

    tool.import_preset_json(presets::PRESET_BIMODAL_COALESCENCE)
        .expect("Bimodal preset loads");
    assert!(tool.density_terminal.iter().all(|p| p[1].is_finite()));
    assert!(!tool.export_datasets().is_empty());

    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}
