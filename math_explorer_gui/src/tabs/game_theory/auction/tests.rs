#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;

#[test]
fn test_optimal_auction_tool_myerson_reserve_presets_and_render() {
    let mut tool = OptimalAuctionTool::default();
    assert!((tool.optimal_reserve - 50.0).abs() < 1e-3);
    assert!((tool.mc_optimal_revenue - 41.666).abs() < 2.0);
    assert!((tool.no_reserve_revenue - 33.333).abs() < 0.2);

    tool.import_preset_json(presets::PRESET_SINGLE_BIDDER_MONOPOLY)
        .expect("Monopoly preset loads");
    assert!((tool.optimal_reserve - 100.0).abs() < 1e-2);
    assert!(tool.no_reserve_revenue < 1e-6);
    assert!((tool.mc_optimal_revenue - 50.0).abs() < 3.0);

    tool.import_preset_json(presets::PRESET_EXPONENTIAL_HAZARD)
        .expect("Exponential preset loads");
    assert!((tool.optimal_reserve - 4.0).abs() < 1e-3);
    assert!(!tool.export_datasets().is_empty());

    let ctx = eframe::egui::Context::default();
    let _ = ctx.run(eframe::egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });
}
