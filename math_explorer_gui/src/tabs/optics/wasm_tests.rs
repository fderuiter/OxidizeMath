#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless browser `wasm32` tests for [`super::OpticsTab`], `#tab=optics` URL hydration,
//! starter presets, and PNG/SVG/CSV/JSON exports.

// [cite:graph_parameters_rust]

use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

use super::*;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, ExportStatusSeverity, InteractiveTool};
use crate::tabs::ExplorerTab;

fn wasm_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        events,
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_optics_tab_hydrates_renders_presets_and_exports() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings("#tab=optics&tool=FourierOpticsWorkbenchTool", "", None)
        .expect("optics share URL must parse");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "optics");

    let mut tab = OpticsTab::default();
    assert!(tab.select_tool_by_name("FourierOpticsWorkbenchTool"));
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });

    let mut tool = FourierOpticsWorkbenchTool::default();
    tool.import_preset_json(presets::PRESET_GERCHBERG_SAXTON_HOLOGRAM)
        .expect("GS preset must load on wasm32");
    assert!(!tool.export_datasets().is_empty());

    let fw = tab.framework_mut().expect("framework");
    fw.export_options.width = 320;
    fw.export_options.height = 240;
    for fmt in [
        ExportFormat::Png,
        ExportFormat::Svg,
        ExportFormat::Csv,
        ExportFormat::Json,
    ] {
        let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(fmt);
            fw.show(ctx, "wasm_optics_export");
        });
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    }
}
