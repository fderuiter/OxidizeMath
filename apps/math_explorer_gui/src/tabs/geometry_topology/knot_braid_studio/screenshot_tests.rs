//! Headless software rasterizer verification generating PNG previews of `KnotBraidStudioTool`.

use super::presets::{
    BORROMEAN_RINGS_PRESET_JSON, CHIRAL_TREFOIL_PRESET_JSON, FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
};
use super::KnotBraidStudioTool;
use crate::app::theme::ThemeMode;
use crate::framework::InteractiveTool;

fn render_studio_preset_png(preset_json: &str, out_path: &str) {
    let mut tool = KnotBraidStudioTool::default();
    tool.import_preset_json(preset_json).expect("preset json");
    crate::test_rasterizer::render_ui_to_png(ThemeMode::Dark, [1320, 820], out_path, |ui| {
        tool.show_inside(ui);
    });
}

#[test]
fn generate_knot_braid_studio_verification_screenshots() {
    render_studio_preset_png(
        CHIRAL_TREFOIL_PRESET_JSON,
        "/tmp/knot_studio_chiral_trefoil.png",
    );
    render_studio_preset_png(
        FIGURE_EIGHT_ISOTOPY_PRESET_JSON,
        "/tmp/knot_studio_figure_eight_reidemeister.png",
    );
    render_studio_preset_png(
        BORROMEAN_RINGS_PRESET_JSON,
        "/tmp/knot_studio_borromean_rings.png",
    );
}
