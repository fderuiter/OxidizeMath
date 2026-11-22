#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{PRESET_GRAY_SCOTT_SPIRAL_FLOW, PRESET_KARMAN_TURING_WAKE};
use super::*;
use crate::app::{contrast_ratio, ThemeMode};
use crate::async_sim::{SimCommand, SimulationRunner};
use crate::framework::InteractiveTool;
use eframe::egui::Color32;
use wasm_bindgen_test::{wasm_bindgen_test, wasm_bindgen_test_configure};

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_coupled_morphogenesis_presets_sculpting_and_overlays() {
    let mut tool = MorphogenesisTool::default();
    tool.import_preset_json(PRESET_KARMAN_TURING_WAKE)
        .expect("Karman wake preset must load on wasm32");
    tool.step_simulation();
    let img = tool
        .export_raster_image()
        .expect("Coupled morphogenesis raster export must succeed on wasm32");
    assert_eq!(img.size, [96, 64]);
    assert!(tool.live_peclet >= 0.0 && tool.live_peclet.is_finite());
    assert!(tool.live_damkohler >= 0.0 && tool.live_damkohler.is_finite());

    tool.import_preset_json(PRESET_GRAY_SCOTT_SPIRAL_FLOW)
        .expect("Gray-Scott swirl preset must load on wasm32");
    let mut runner = MorphogenesisTab::create_runner(&tool.params, 24, 24);
    runner.process_command(SimCommand::ApplyBrush {
        cx: 8,
        cy: 8,
        r: 2,
        is_obstacle: true,
    });
    runner.step();
    let snap = runner.get_snapshot();
    assert_eq!(snap.width, 24);

    let hc = render_morphogenesis_overlay(
        &[0.0, 0.5, 1.0, 0.25],
        &[-0.9, 0.0, 0.9, 0.3],
        &[false, true, false, false],
        2,
        2,
        MorphogenesisOverlayMode::DualOverlay,
        ThemeMode::HighContrast,
    );
    for px in hc.pixels {
        assert!(contrast_ratio(px, Color32::BLACK) >= 3.0);
    }
}
