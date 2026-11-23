#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Headless WebAssembly (`wasm32-unknown-unknown`) browser tests for `VolumetricFieldStudioTool`.

use super::presets::{PRESET_1S_2PZ_DIPOLE_JSON, PRESET_3D_TURING_GYROID_JSON};
use super::state::{SliceAxis, VolumetricRenderMode};
use super::VolumetricFieldStudioTool;
use crate::framework::InteractiveTool;
use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_volumetric_studio_quantum_and_turing_modes_step_slice_and_export_obj() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.import_preset_json(PRESET_1S_2PZ_DIPOLE_JSON)
        .expect("dipole preset valid on wasm32");
    tool.config.grid_resolution = 18;
    tool.refresh_zero_alloc();
    tool.step_zero_alloc();

    let ctx = egui::Context::default();
    for mode in [
        VolumetricRenderMode::MarchingCubesMesh,
        VolumetricRenderMode::GaussianRadianceCloud,
        VolumetricRenderMode::HybridShellAndCloud,
    ] {
        tool.config.render_mode = mode;
        tool.refresh_zero_alloc();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(800.0, 600.0),
            )),
            ..Default::default()
        };
        let _ = ctx.run(raw_input, |ctx| {
            tool.show(ctx);
        });
    }

    tool.import_preset_json(PRESET_3D_TURING_GYROID_JSON)
        .expect("turing gyroid preset valid on wasm32");
    tool.config.grid_resolution = 18;
    tool.config.cut_enabled = true;
    tool.config.cut_axis = SliceAxis::Z;
    tool.config.cut_offset = 0.1;
    tool.refresh_zero_alloc();
    tool.step_zero_alloc();

    let obj = tool
        .export_obj_string()
        .expect("OBJ export succeeds on wasm32");
    assert!(obj.contains("\nv ") && obj.contains("\nvn ") && obj.contains("\nf "));
}
