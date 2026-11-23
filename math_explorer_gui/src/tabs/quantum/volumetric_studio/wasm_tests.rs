#![cfg_attr(any(), verified(opt_out = "tests"))]
//! Headless WebAssembly (`wasm32-unknown-unknown`) browser tests for `VolumetricFieldStudioTool`.

use super::presets::{
    PRESET_1S_2PZ_DIPOLE_JSON, PRESET_3D_TURING_GYROID_JSON, PRESET_CSG_BOOLEAN_SOLID_JSON,
    PRESET_GYROID_TPMS_JSON,
};
use super::state::{SliceAxis, VolumetricRenderMode};
use super::VolumetricFieldStudioTool;
use crate::framework::InteractiveTool;
use eframe::egui;
use oxidize_core::mesh::{export_colored_mesh_to_ply_binary, export_mesh_to_stl_binary};
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_volumetric_studio_quantum_turing_and_implicit_cad_solid_and_splat_exports() {
    let mut tool = VolumetricFieldStudioTool::default();
    tool.import_preset_json(PRESET_1S_2PZ_DIPOLE_JSON)
        .expect("dipole preset valid on wasm32");
    tool.config.grid_resolution = 18;
    tool.refresh_zero_alloc();
    tool.step_zero_alloc();

    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(800.0, 600.0),
        )),
        ..Default::default()
    };
    for mode in [
        VolumetricRenderMode::MarchingCubesMesh,
        VolumetricRenderMode::GaussianRadianceCloud,
        VolumetricRenderMode::HybridShellAndCloud,
    ] {
        tool.config.render_mode = mode;
        tool.refresh_zero_alloc();
        let _ = ctx.run(raw_input.clone(), |ctx| tool.show(ctx));
    }

    for preset in [
        PRESET_3D_TURING_GYROID_JSON,
        PRESET_GYROID_TPMS_JSON,
        PRESET_CSG_BOOLEAN_SOLID_JSON,
    ] {
        tool.import_preset_json(preset)
            .expect("volumetric preset valid on wasm32");
        tool.config.grid_resolution = 18;
        tool.config.cut_enabled = true;
        tool.config.cut_axis = SliceAxis::Z;
        tool.config.cut_offset = 0.15;
        tool.refresh_zero_alloc();
        let mesh = tool
            .export_mesh_3d()
            .expect("3D solid capability")
            .expect("closed solid mesh on wasm32");
        let stl = export_mesh_to_stl_binary(&mesh).expect("binary STL on wasm32");
        assert_eq!(stl.len(), 84 + 50 * (mesh.indices.len() / 3));
        assert!(std::str::from_utf8(&stl[..80])
            .unwrap_or("")
            .contains("UNITS=mm"));
    }

    let colored = tool.export_colored_mesh_3d().unwrap().unwrap();
    let ply = export_colored_mesh_to_ply_binary(&colored).expect("PLY export succeeds on wasm32");
    assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));

    tool.config.render_mode = VolumetricRenderMode::GaussianRadianceCloud;
    tool.refresh_zero_alloc();
    let splat_ply = tool.export_splat_ply_3d().unwrap().unwrap();
    assert!(splat_ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));

    tool.import_preset_json(PRESET_3D_TURING_GYROID_JSON)
        .expect("turing gyroid preset valid on wasm32");
    tool.config.grid_resolution = 18;
    tool.refresh_zero_alloc();
    let obj = tool.export_obj_string().expect("OBJ succeeds on wasm32");
    assert!(obj.contains("\nv ") && obj.contains("\nvn ") && obj.contains("\nf "));
}
