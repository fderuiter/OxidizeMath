#![cfg_attr(any(), verified(opt_out = "tests"))]

//! Headless browser `wasm32` tests for [`SubatomicPhysicsTab`], `#tab=subatomic` hydration,
//! `(Z, N)` chart interaction, starter presets, and PNG/SVG/CSV/JSON exports.

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
            egui::Vec2::new(1100.0, 780.0),
        )),
        events,
        ..Default::default()
    }
}

#[wasm_bindgen_test]
fn wasm_subatomic_tab_hydrates_renders_and_loads_presets() {
    let mut app = MathExplorerApp::default();
    let hydrated = app
        .apply_share_location_strings("#tab=subatomic&tool=RelativisticJetTool", "", None)
        .expect("subatomic share URL must parse");
    assert!(hydrated);
    assert_eq!(app.tabs[app.selected_tab].url_slug(), "subatomic");

    let mut tab = SubatomicPhysicsTab::default();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    for tool_id in [
        "NuclearStabilityTool",
        "StandardModelTool",
        "RelativisticJetTool",
    ] {
        assert!(tab.select_tool_by_name(tool_id));
        let _ = ctx.run(wasm_input(Vec::new()), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| tab.show_inside(ui, &mut frame));
        });
    }

    let mut sm = StandardModelTool::default();
    sm.import_preset_json(presets::PRESET_DUNE_MSW_NEUTRINOS)
        .expect("MSW preset must load on wasm32");
    assert!(!sm.export_datasets().is_empty());
}

#[wasm_bindgen_test]
fn wasm_subatomic_finite_state_guard_rollback_and_exports() {
    let mut tab = SubatomicPhysicsTab::default();
    let ctx = egui::Context::default();
    assert!(tab.select_tool_by_name("RelativisticJetTool"));
    tab.framework.cache_valid_snapshot();
    {
        let jet = tab
            .framework
            .active_tool
            .as_mut()
            .expect("active tool")
            .as_any_mut()
            .expect("as_any_mut")
            .downcast_mut::<RelativisticJetTool>()
            .expect("RelativisticJetTool");
        jet.state.beta = f64::NAN;
        assert!(crate::framework::recovery::FiniteStateGuard::diverged(jet));
    }
    assert!(tab.framework.inspect_runtime_state().is_err());
    assert!(tab.framework.rollback_step_back());
    assert!(!tab.framework.is_diverged());

    let fw = tab.framework_mut().expect("framework");
    fw.export_options.width = 320;
    fw.export_options.height = 200;
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
            fw.show(ctx, "wasm_subatomic_export");
        });
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    }
}
