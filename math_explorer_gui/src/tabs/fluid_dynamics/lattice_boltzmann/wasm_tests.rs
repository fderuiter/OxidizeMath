//! Browser/wasm32 verification tests for [`super::LatticeBoltzmannTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;
use crate::url_state::{encode_share_fragment, parse_share_location};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_lbm_wind_tunnel_sculpt_mask_url_roundtrip_and_drag_telemetry() {
    let mut tool = LatticeBoltzmannTool::default();
    tool.apply_obstacle_preset(bluff_plate_obstacle_grid());
    tool.area_budget = 180;
    tool.brush_radius = 2;
    tool.view_mode = LbmViewMode::Vorticity;
    tool.push_shared_config();
    tool.sculpt_at_cell(42, 24, true);
    assert!(count_solid_cells(&tool.obstacles) <= 180);

    let preset = tool
        .export_preset_json()
        .expect("preset export on wasm32");
    let frag = encode_share_fragment("fluid_dynamics", Some("LatticeBoltzmannTool"), Some(&preset))
        .expect("URL share fragment on wasm32");
    let parsed = parse_share_location(&frag, "")
        .expect("parse share location")
        .expect("params");

    let mut hydrated = LatticeBoltzmannTool::default();
    hydrated
        .import_preset_json(parsed.state_json.as_deref().unwrap())
        .expect("hydrate LBM state on wasm32");
    assert_eq!(hydrated.obstacles, tool.obstacles);

    hydrated.step_simulation();
    assert!(hydrated.telemetry.cd.is_finite());
    assert!(hydrated.telemetry.reynolds > 0.0);
}
