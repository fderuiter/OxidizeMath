#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::ManifoldFlowStudio;
use crate::framework::InteractiveTool;
use eframe::egui;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_manifold_flow_studio_wasm_smoke() {
    let mut studio = ManifoldFlowStudio::default();
    studio.step_simulation();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        studio.show(ctx);
    });
    assert!(studio.export_mesh_3d().unwrap().is_ok());
}
