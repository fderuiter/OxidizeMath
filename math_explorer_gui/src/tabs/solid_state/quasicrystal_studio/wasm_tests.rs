//! Headless `wasm32-unknown-unknown` compatibility tests for [`super::QuasicrystalStudio`].

// [cite:solid_state]

use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

use super::*;
use crate::framework::InteractiveTool;

#[wasm_bindgen_test]
fn wasm_quasicrystal_studio_presets_and_exports_succeed() {
    let ctx = egui::Context::default();
    let mut studio = QuasicrystalStudio::default();

    for payload in [
        presets::PRESET_PENROSE_10FOLD,
        presets::PRESET_AMMANN_BEENKER_8FOLD,
        presets::PRESET_STAMPFLI_12FOLD,
        presets::PRESET_HEXAGONAL_BRAVAIS_BZ,
    ] {
        studio
            .import_preset_json(payload)
            .expect("WASM preset import must succeed");
    }

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| studio.show_inside(ui));
    });

    let img = studio.export_raster_image().expect("WASM raster image");
    assert_eq!(img.size, [256, 256]);
    assert!(!studio.export_datasets().is_empty());
}
