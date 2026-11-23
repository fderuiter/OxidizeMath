#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_shannon_channel_tool_entropy_blahut_arimoto_and_render() {
    let mut tool = ShannonChannelTool::default();
    assert!((tool.entropy_h - 1.75).abs() < 1e-9);
    assert!((tool.closed_capacity - tool.ba_capacity).abs() < 1e-7);
    assert!((tool.induced_mi - tool.ba_capacity).abs() < 1e-7);

    tool.import_preset_json(presets::PRESET_Z_CHANNEL_ASYMMETRIC)
        .expect("Z-channel preset loads");
    assert!(tool.opt_input[1] < 0.5 && tool.opt_input[1] > 0.35);
    assert!(tool.ba_capacity > 0.55 && tool.ba_capacity < 0.70);

    tool.import_preset_json(presets::PRESET_AWGN_HARTLEY)
        .expect("AWGN preset loads");
    assert!((tool.closed_capacity - 2.5).abs() < 1e-9);
    assert!((tool.bandwidth_bps - 100_000.0).abs() < 1e-6);
    assert!(!tool.export_datasets().is_empty());

    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}
