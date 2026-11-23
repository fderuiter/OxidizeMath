#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::CurvilinearVectorStudio;
use crate::framework::InteractiveTool;
use eframe::egui;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_curvilinear_vector_studio_wasm_smoke() {
    let mut studio = CurvilinearVectorStudio::default();
    assert!(studio.grid_res <= 32);
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        studio.show(ctx);
    });
    assert!(studio.diagnostics.gauss_residual.is_finite());
    assert!(studio.diagnostics.stokes_residual.is_finite());
}
