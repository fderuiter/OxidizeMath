#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_wasm_potential_flow_wind_tunnel_renders_and_steps_dye() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = PotentialFlowTool {
        elements: presets::lifting_cylinder_elements(),
        dye_playing: true,
        ..Default::default()
    };
    tool.rebuild_field();
    tool.reseed_dye_emitters();

    let head0 = tool.dye_streaks[0].head;
    let out = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(1280.0, 800.0),
            )),
            ..Default::default()
        },
        |c| tool.show(c),
    );
    assert!(!out.shapes.is_empty());
    assert_ne!(tool.dye_streaks[0].head, head0);
    assert_eq!(tool.cached_stagnation.len(), 2);
}
