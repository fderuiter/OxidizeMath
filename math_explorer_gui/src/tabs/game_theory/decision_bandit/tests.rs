#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_decision_bandit_tool_voi_and_seeded_regret() {
    let mut tool = DecisionBanditTool::default();
    assert!(tool.evsi_val >= 0.0 && tool.evsi_val <= tool.evpi_val + 1e-9);

    tool.import_preset_json(presets::PRESET_CARA_SEISMIC_VOI)
        .expect("CARA preset loads");
    assert!(tool.evsi_val > 0.0 && tool.evsi_val <= tool.evpi_val);

    tool.import_preset_json(presets::PRESET_THOMPSON_VS_UCB1)
        .expect("Bandit preset loads");
    assert_eq!(tool.thompson_regret.len(), 400);
    assert!(!tool.export_datasets().is_empty());

    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}
