#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::PRESET_OPTICS_VESELAGO;
use super::{OpticsWaveTab, OpticsWaveTool};
use crate::app::ThemeMode;
use crate::framework::InteractiveTool;
use crate::tabs::ExplorerTab;
use eframe::egui::{vec2, Context, Pos2, RawInput, Rect};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_wasm_optics_wave_tab_and_presets() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    ThemeMode::HighContrast.apply_to_context(&ctx);

    let tab = OpticsWaveTab::default();
    assert_eq!(tab.url_slug(), "optics_waves");

    let mut tool = OpticsWaveTool::default();
    tool.import_preset_json(PRESET_OPTICS_VESELAGO)
        .expect("Veselago preset loads on wasm32");
    tool.step_simulation();
    assert!(tool.export_raster_image().is_some());

    let raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(1024.0, 768.0))),
        ..Default::default()
    };
    let out = ctx.run(raw, |ctx| tool.show(ctx));
    assert!(out.platform_output.accesskit_update.is_some());
}
