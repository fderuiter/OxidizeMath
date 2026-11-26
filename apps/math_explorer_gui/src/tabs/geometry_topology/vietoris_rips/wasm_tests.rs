#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::state::{SculptBrush, VrSourceMode};
use super::*;
use crate::framework::InteractiveTool;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn wasm_vietoris_rips_3d_lorenz_takens_and_sculpt_roundtrip() {
    let mut tool = VietorisRipsTool::default();
    let lorenz_json = r#"{"mode":"lorenz_3d","radius":7.5,"chaos":{"sigma":10.0,"rho":28.0,"beta":2.6666666666666665,"dt":0.01,"landmark_count":64,"takens_lag":8},"camera":{"pitch":0.35,"yaw":0.55,"zoom":1.0}}"#;
    tool.try_load_state(lorenz_json)
        .expect("wasm 3d lorenz preset load");
    assert_eq!(tool.mode, VrSourceMode::Lorenz3D);
    assert_eq!(tool.points_3d.len(), 64);
    assert!(!tool.barcode_intervals.is_empty());

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tool.show_inside(ui));
    });
    assert!(tool.has_finite_parameters());

    // Switch to Takens 3D delay embedding and step simulation on wasm32.
    tool.mode = VrSourceMode::Takens3D;
    tool.chaos.takens_lag = 10;
    tool.step_simulation();
    assert_eq!(tool.points_3d.len(), 64);

    // Sculpt in 3D and verify lossless state serialization.
    assert!(tool.apply_sculpt_3d_projected(SculptBrush::Erase, [0.0, 0.0], 50.0, false));
    assert_eq!(tool.mode, VrSourceMode::Points3D);
    assert_eq!(tool.points_3d.len(), 63);
    let saved = tool.save_state().expect("wasm save_state");
    let mut restored = VietorisRipsTool::default();
    restored.try_load_state(&saved).expect("wasm restore");
    assert_eq!(restored.points_3d.len(), 63);
}
