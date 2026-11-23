#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless `wasm32` integration tests for [`StochasticPdeTab`], `#tab=stochastic_pde` URL
//! hydration, starter presets, and PNG/SVG/CSV exports.

use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, ExportStatusSeverity};
use crate::tabs::ExplorerTab;

fn wasm_input() -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_stochastic_pde_tab_hydrates_renders_presets_and_exports() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings("#tab=stochastic_pde&tool=KpzInterfaceTool", "", None)
        .expect("stochastic_pde share URL must parse");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "stochastic_pde");

    let mut tab = StochasticPdeTab::default();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();

    for (tool_id, preset_json) in [
        ("KpzInterfaceTool", presets::PRESET_KPZ_BALLISTIC_1D),
        (
            "SpinodalDecompositionTool",
            presets::PRESET_CAHN_HILLIARD_MAZE,
        ),
    ] {
        assert!(tab.select_tool_by_name(tool_id));
        tab.framework_mut()
            .expect("framework")
            .active_tool
            .as_mut()
            .expect("active tool")
            .import_preset_json(preset_json)
            .expect("preset must load on wasm32");

        let _ = ctx.run(wasm_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });

        let fw = tab.framework_mut().expect("framework");
        fw.export_options.width = 320;
        fw.export_options.height = 200;
        for fmt in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
            let _ = ctx.run(wasm_input(), |ctx| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(fmt);
                fw.show(ctx, "wasm_spde_export");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        }
    }
}
