#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless browser `wasm32` tests for [`FinancialMathTab`] risk workbenches
//! (`CopulaRiskTool`, `OuProcessTool`, `MarkovRegimeTool`, and `BetSizeCalculatorTool`).

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
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_financial_math_risk_workbenches_hydrate_render_and_export() {
    for tool_id in [
        "CopulaRiskTool",
        "OuProcessTool",
        "MarkovRegimeTool",
        "BetSizeCalculatorTool",
    ] {
        let mut app = MathExplorerApp::default();
        let fragment = format!("#tab=financial_math&tool={tool_id}");
        let hydrated = app
            .apply_share_location_strings(&fragment, "", None)
            .expect("financial_math share URL must parse");
        assert!(hydrated);

        let mut tab = FinancialMathTab::default();
        assert!(tab.select_tool_by_name(tool_id));
        let ctx = egui::Context::default();
        let mut frame = eframe::Frame::_new_kittest();
        let _ = ctx.run(wasm_input(), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });

        let fw = tab.framework_mut().expect("framework");
        fw.export_options.width = 320;
        fw.export_options.height = 200;
        for fmt in [
            ExportFormat::Png,
            ExportFormat::Svg,
            ExportFormat::Csv,
            ExportFormat::Json,
        ] {
            let _ = ctx.run(wasm_input(), |ctx| {
                egui_plot::clear_last_plot_datasets();
                fw.show_export_modal = true;
                fw.queue_export(fmt);
                fw.show(ctx, "wasm_financial_math_export");
            });
            assert_eq!(
                fw.export_status_severity,
                ExportStatusSeverity::Success,
                "{tool_id} failed {fmt:?} export on wasm32"
            );
        }
    }
}
