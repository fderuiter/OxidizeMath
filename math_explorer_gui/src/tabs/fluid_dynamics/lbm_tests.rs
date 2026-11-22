#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::*;
use crate::framework::{
    BrushStencil, InteractiveTool, RasterBrushStroke, SymmetryMode, generate_brush_cells,
};
use crate::url_state::{encode_share_fragment, parse_share_location};
use eframe::egui::Context;

#[test]
fn test_lbm_unified_persists_airfoil_across_save_load_reset_and_url() {
    let ctx = Context::default();
    let mut tool = UnifiedSimTool::<LbmUnified>::new();
    let rev_initial = tool.state_revision();

    let airfoil_cells = generate_brush_cells(
        BrushStencil::Airfoil,
        SymmetryMode::None,
        (45, 25),
        (45, 25),
        3,
        100,
        50,
    );
    tool.apply_brush_stroke(
        RasterBrushStroke {
            cells: airfoil_cells.clone(),
            secondary: false,
            via_keyboard: false,
            focus_cell: (45, 25),
        },
        &ctx,
    );

    assert_ne!(tool.state_revision(), rev_initial);
    let saved = tool.save_state().expect("LBM save_state");
    assert!(saved.contains("obstacles_rle"));

    // Verify URL share roundtrip within URL_JSON_BOUNDS:
    let fragment = encode_share_fragment(
        "fluid_dynamics",
        Some("LatticeBoltzmannTool"),
        Some(&saved),
    )
    .expect("encode LBM URL fragment");
    let decoded_url = parse_share_location(&fragment, "")
        .expect("decode LBM URL fragment")
        .expect("present share params");
    assert_eq!(decoded_url.state_json.as_deref(), Some(saved.as_str()));

    // Verify reset_simulation preserves sculpted obstacles:
    tool.reset_simulation(Some(&ctx));
    assert_eq!(tool.save_state().as_deref(), Some(saved.as_str()));

    // Verify try_load_state on fresh tool restores exact obstacles and raster export:
    let mut restored = UnifiedSimTool::<LbmUnified>::new();
    restored
        .try_load_state(decoded_url.state_json.as_deref().unwrap())
        .expect("restore LBM state");
    assert_eq!(restored.save_state().as_deref(), Some(saved.as_str()));
    restored.step_simulation();
    let img = restored.export_raster_image().expect("LBM raster image");
    // Image row y = 50 - 1 - 25 = 24 at x = 45 must be obstacle gray(50):
    assert_eq!(img.pixels[24 * 100 + 45], Color32::from_gray(50));

    // Reject dimension-mismatched RLE:
    let bad_rle = crate::framework::encode_bool_grid(20, 20, &[true; 400]).unwrap();
    let bad_json = serde_json::json!({ "viscosity": 0.02, "obstacles_rle": bad_rle }).to_string();
    assert!(restored.try_load_state(&bad_json).is_err());
}
