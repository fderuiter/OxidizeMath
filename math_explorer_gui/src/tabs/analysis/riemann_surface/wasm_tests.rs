//! Headless `wasm32-unknown-unknown` smoke tests for [`super::RiemannSurfaceStudio`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{RiemannBranchFamily, RiemannSurfaceStudio};
use crate::framework::InteractiveTool;
use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_smoke_riemann_surface_studio_frame_and_roundtrip() {
    let mut studio = RiemannSurfaceStudio {
        branch_family: RiemannBranchFamily::Logarithm,
        winding_turns: 2,
        dirty: true,
        ..Default::default()
    };

    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |c| studio.show(c));

    let json = studio.save_state().expect("state serializes on wasm32");
    let mut restored = RiemannSurfaceStudio::default();
    restored.try_load_state(&json).expect("restores on wasm32");
    assert_eq!(restored.diagnostics.final_sheet, 2);
}
