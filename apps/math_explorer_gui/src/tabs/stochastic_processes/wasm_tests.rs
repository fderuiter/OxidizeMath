#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless browser `wasm32` tests for [`super::StochasticProcessesTab`], URL state hydration,
//! deterministic `OxidizeRng` stepping, and deferred PNG/SVG/CSV exports.

use super::StochasticProcessesTab;
use crate::app::MathExplorerApp;
use crate::framework::{ExportFormat, ExportStatusSeverity};
use crate::tabs::ExplorerTab;
use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

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
fn wasm_stochastic_processes_tab_hydrates_renders_and_exports_all_workbenches() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings(
            "#tab=stochastic_processes&tool=HmmOuWorkbenchTool",
            "",
            None,
        )
        .expect("stochastic_processes share URL must parse on wasm32");
    assert!(hydrated);
    assert_eq!(
        app.tabs[app.selected_tab].url_slug(),
        "stochastic_processes"
    );

    let mut tab = StochasticProcessesTab::default();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();

    for tool_id in [
        "DtmcTensorWorkbenchTool",
        "CtmcQueueWorkbenchTool",
        "HmmOuWorkbenchTool",
    ] {
        assert!(tab.select_tool_by_name(tool_id));
        let _ = ctx.run(wasm_input(), |c| {
            egui::CentralPanel::default().show(c, |ui| tab.show_inside(ui, &mut frame));
        });

        let fw = tab.framework_mut().expect("framework");
        fw.export_options.width = 320;
        fw.export_options.height = 220;
        for fmt in [ExportFormat::Png, ExportFormat::Svg, ExportFormat::Csv] {
            let _ = ctx.run(wasm_input(), |c| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(fmt);
                fw.show(c, "wasm_stochastic_export");
            });
            assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        }
    }
}
