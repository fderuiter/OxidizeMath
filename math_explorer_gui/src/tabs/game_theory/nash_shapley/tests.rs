#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_nash_shapley_tool_equilibria_presets_and_render() {
    let mut tool = NashShapleyTool::default();
    assert_eq!(tool.enumerated_count, 3);
    assert!(tool.max_regret < 1e-6);

    tool.import_preset_json(presets::PRESET_RPS_ZERO_SUM)
        .expect("RPS preset loads");
    assert!((tool.row_payoff).abs() < 1e-8);
    assert!((tool.row_strategy[0] - 1.0 / 3.0).abs() < 1e-6);

    tool.import_preset_json(presets::PRESET_CONVEX_SYNERGY_SHAPLEY)
        .expect("Shapley preset loads");
    assert!(tool.convex);
    assert!(tool.in_core);
    let sum_phi: f64 = tool.shapley.iter().sum();
    assert!((sum_phi - 12.0).abs() < 1e-9);
    assert!(!tool.export_datasets().is_empty());

    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}
