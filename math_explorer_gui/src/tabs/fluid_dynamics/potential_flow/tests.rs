#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn potential_flow_clamps_high_grid_state_and_caches_static_arrow_field() {
    let mut tool = PotentialFlowTool::default();
    let high_grid = r#"{"elements":[{"element_type":"Uniform","strength":10.0,"x":0.0,"y":0.0,"angle":0.0}],"grid_size":1000,"view_range":10.0}"#;
    tool.try_load_state(high_grid)
        .expect("grid_size 1000 should load and clamp to interactive cap");
    assert_eq!(tool.grid_size, crate::presets::MAX_INTERACTIVE_GRID_SIZE);
    assert!(tool.has_finite_parameters());

    let mut live_tool = PotentialFlowTool::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| live_tool.show(c));
    let _ = ctx.run(egui::RawInput::default(), |c| live_tool.show(c));
    assert_eq!(live_tool.cache.eval_count, 1);

    live_tool.grid_size = 24;
    let _ = ctx.run(egui::RawInput::default(), |c| live_tool.show(c));
    assert_eq!(live_tool.cache.eval_count, 2);
}
