//! UI rendering, gallery discovery, trace overlay, dosing card, and `wasm32` browser tests.

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{InteractiveTool, ToolMetadata};
use crate::tabs::clinical_trials::ClinicalTrialsTab;
use crate::tabs::ExplorerTab;

fn run_headless_tool_frame(tool: &mut PharmacokineticsTool) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            tool.show_inside(ui);
        });
    });
}

#[test]
fn test_clinical_trials_tab_and_metadata_discovery() {
    let meta = inventory::iter::<ToolMetadata>
        .into_iter()
        .find(|m| m.id == "PharmacokineticsTool")
        .expect("PharmacokineticsTool registered in inventory");
    assert_eq!(meta.display_name, "Pharmacokinetics (ADME)");
    assert_eq!(meta.domain, "clinical_trials");
    assert!(meta.tags.contains(&"adme"));

    let mut tab = ClinicalTrialsTab::default();
    assert!(tab.select_tool_by_name("PharmacokineticsTool"));
    assert_eq!(tab.active_tool_id(), Some("PharmacokineticsTool"));

    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 800.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tab.framework_mut()
            .expect("ClinicalTrialsTab has framework")
            .show(ctx, "clinical_trials");
    });
}

#[test]
fn test_ui_renders_all_modes_log_axis_and_solver_error_banner() {
    let mut tool = PharmacokineticsTool {
        entity_mode: EntityMode::Enantiomer,
        release_mode: ReleaseMode::TwoPulse,
        use_loading_dose: true,
        missed_slots: vec![1],
        log10_y: true,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();
    let series = export::build_active_trace_series(&tool);
    assert_eq!(series.len(), 3);
    assert!(tool
        .snapshot_manager
        .add_snapshot("Chiral XR", tool.param_summary(), series));
    run_headless_tool_frame(&mut tool);

    tool.absorption_mode = AbsorptionMode::SolveFromTMax;
    tool.target_t_max = 0.0;
    tool.recalculate();
    assert!(tool.solver_error.is_some());
    run_headless_tool_frame(&mut tool);
}

#[test]
fn test_export_datasets_and_dosing_summary_cards() {
    let mut tool = PharmacokineticsTool {
        entity_mode: EntityMode::Enantiomer,
        release_mode: ReleaseMode::TwoPulse,
        use_loading_dose: true,
        loading_dose_mg: 250.0,
        dose_mg: 125.0,
        n_doses: 4,
        missed_slots: vec![2],
        log10_y: true,
        ..PharmacokineticsTool::default()
    };
    tool.recalculate();

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 3);
    assert!(datasets[0].1.iter().all(|p| p[1] >= 0.0));

    let md = export::format_markdown_card(&tool);
    assert!(md.contains("# Pharmacokinetics (ADME) Dosing Regimen Card"));
    assert!(md.contains("| Peak Concentration (C_max) |"));
    assert!(md.contains("| Time in Therapeutic Window |"));
    assert!(md.contains("| 1 | 0.0 | 250.0 | Loading |"));
    assert!(md.contains("| 2 | 12.0 | 125.0 | Maintenance |"));
    assert!(md.contains("| 3 | 24.0 | 0.0 | Missed |"));

    let csv = export::format_csv_card(&tool);
    assert!(csv.contains("metric,C_max,"));
    assert!(csv.contains("metric,accumulation_ratio,"));
    assert!(csv.contains("schedule,dose_1,0.00,250.00,mg,loading"));
    assert!(csv.contains("schedule,dose_3,24.00,0.00,mg,missed"));
}

#[cfg(target_arch = "wasm32")]
mod wasm_browser_tests {
    use super::*;
    use crate::app::MathExplorerApp;
    use crate::starter_gallery::StarterPresetMetadata;
    use crate::url_state::MAX_URL_FRAGMENT_BYTES;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_pharmacokinetics_starter_and_url_roundtrip() {
        let preset = inventory::iter::<StarterPresetMetadata>
            .into_iter()
            .find(|p| p.id == "pharmacokinetics_adme_starter")
            .expect("pharmacokinetics_adme_starter registered");

        let mut app = MathExplorerApp::default();
        let tab_idx = app
            .tabs
            .iter()
            .position(|t| t.url_slug() == "clinical_trials")
            .expect("clinical_trials tab exists");
        app.selected_tab = tab_idx;
        app.tabs[tab_idx]
            .framework_mut()
            .expect("framework present")
            .apply_starter_card(preset.title, preset.tool_name, Some(preset.generate_preset()))
            .expect("starter card applies cleanly");

        let share_url = app.encode_current_share_url().unwrap();
        let hash_idx = share_url.find('#').expect("fragment delimiter present");
        let fragment = &share_url[hash_idx..];
        assert!(fragment.contains("tool=PharmacokineticsTool"));
        assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);

        let mut restored = MathExplorerApp::default();
        assert!(restored
            .apply_share_location_strings(fragment, "", None)
            .unwrap());
        assert_eq!(
            restored.tabs[tab_idx].active_tool_id(),
            Some("PharmacokineticsTool")
        );
    }

    #[wasm_bindgen_test]
    fn wasm_pharmacokinetics_card_exports_trigger_browser_downloads() {
        let mut tool = PharmacokineticsTool::default();
        let md = export::format_markdown_card(&tool);
        tool.card_export.trigger_save_markdown(md);
        tool.card_export.poll();
        assert!(tool
            .card_export
            .status_info
            .as_deref()
            .unwrap_or("")
            .contains("pk_regimen_card.md"));
        assert!(tool.card_export.status_error.is_none());

        let csv = export::format_csv_card(&tool);
        tool.card_export.trigger_save_csv(csv);
        tool.card_export.poll();
        assert!(tool
            .card_export
            .status_info
            .as_deref()
            .unwrap_or("")
            .contains("pk_regimen_summary.csv"));
        assert!(tool.card_export.status_error.is_none());
    }
}

// [cite:clinical_trials_statistics]
