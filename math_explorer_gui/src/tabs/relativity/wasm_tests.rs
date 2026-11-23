#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless browser `wasm32` tests for [`RelativityTab`], `#tab=relativity` hydration,
//! `Camera3D` interaction, starter presets, and PNG/SVG/CSV exports.

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
fn wasm_relativity_tab_hydrates_renders_gestures_presets_and_exports() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings("#tab=relativity&tool=GeodesicPlaygroundTool", "", None)
        .expect("relativity share URL must parse");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "relativity");

    let mut tab = RelativityTab::default();
    assert!(tab.select_tool_by_name("GeodesicPlaygroundTool"));
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });

    // Verify wheel gesture inside 3D viewport and CSV/SVG/PNG exports on wasm32
    let wheel = vec![
        egui::Event::PointerMoved(egui::pos2(680.0, 280.0)),
        egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Line,
            delta: egui::vec2(0.0, 1.5),
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(wasm_input(wheel), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
    });

    let mut inspector = ChristoffelInspectorTool::default();
    inspector
        .import_preset_json(presets::PRESET_CHRISTOFFEL_CROSS_CHECK)
        .expect("preset must load on wasm32");
    assert!(!inspector.export_datasets().is_empty());

    let fw = tab.framework_mut().expect("framework");
    fw.export_options.width = 320;
    fw.export_options.height = 200;
    for fmt in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
        let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
            egui_plot::clear_last_plot_datasets();
            fw.show_export_modal = true;
            fw.queue_export(fmt);
            fw.show(ctx, "wasm_relativity_export");
        });
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    }
}
