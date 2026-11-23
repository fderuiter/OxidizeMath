#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:graph_parameters_rust]

use egui_plot::export::{color32_image_to_png, format_csv, format_json, ExportOptions};
use egui_plot::export_primitives::export_primitives_svg;
use wasm_bindgen_test::*;

use super::*;
use crate::framework::InteractiveTool;

#[wasm_bindgen_test]
fn wasm_smoke_chladni_wave_tank_presets_stepping_and_exports() {
    let mut tool = ChladniWaveTankTool::default();
    for preset in [
        TankPreset::SquareChladniPlate,
        TankPreset::CircularDrumhead,
        TankPreset::DoubleSlitInterference,
        TankPreset::AcousticLens,
    ] {
        tool.apply_preset(preset);
        tool.step_simulation();
        assert!(tool.has_finite_parameters());

        let saved = tool.save_state().expect("WASM save_state");
        tool.try_load_state(&saved).expect("WASM try_load_state");

        let img = tool.export_raster_image().expect("WASM raster image");
        let png = color32_image_to_png(&img).expect("WASM PNG encode");
        assert!(!png.is_empty());

        let prims = tool.export_plot_primitives();
        let svg = export_primitives_svg(&prims, &ExportOptions::default()).expect("WASM SVG");
        assert!(svg.contains("<svg"));

        let datasets = tool.export_datasets();
        let csv = format_csv(&datasets);
        assert!(csv.starts_with("Dataset,X,Y"));
        let json = format_json(&datasets);
        assert!(json.starts_with('['));
    }
}
