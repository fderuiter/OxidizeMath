#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::app::{MathExplorerApp, ThemeMode};
use crate::framework::theory_links::{
    resolve_citation_reference_links, resolve_portal_paper_links,
};
use crate::framework::{
    ExportFormat, ExportStatusSeverity, SimulationFramework, StarterPresetMetadata, ToolMetadata,
};
use eframe::egui;

const ALL_SPORTS_TOOLS: [&str; 7] = [
    "MarketDeviggingTool",
    "MatchScoreHeatmapTool",
    "SgpCopulaTool",
    "ZipCountPropsTool",
    "LiveMomentumTool",
    "SportsRatingsTool",
    "ClvRuinTool",
];

fn run_deferred_export(framework: &mut SimulationFramework, format: ExportFormat) {
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        egui_plot::clear_last_plot_datasets();
        framework.show_export_modal = true;
        framework.queue_export(format);
        framework.show(ctx, "sports_export_audit");
    });
}

#[test]
fn test_sports_analytics_and_bet_size_deferred_csv_json_svg_png_exports() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for tool_id in ALL_SPORTS_TOOLS
        .into_iter()
        .chain(std::iter::once("BetSizeCalculatorTool"))
    {
        let meta = inventory::iter::<ToolMetadata>
            .into_iter()
            .find(|m| m.id == tool_id)
            .unwrap_or_else(|| panic!("Missing ToolMetadata for {tool_id}"));
        let mut fw = SimulationFramework::new("__sports_audit__");
        fw.export_options.width = 320;
        fw.export_options.height = 240;
        fw.active_tool = Some((meta.build)());

        run_deferred_export(&mut fw, ExportFormat::Csv);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let csv = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("CSV exported");
        assert!(csv.lines().count() >= 2, "{tool_id} CSV had < 2 lines");

        run_deferred_export(&mut fw, ExportFormat::Json);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let json = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("JSON exported");
        let parsed: serde_json::Value = serde_json::from_str(json).expect("valid JSON dataset");
        assert!(parsed.is_object() || parsed.is_array());

        run_deferred_export(&mut fw, ExportFormat::Svg);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        let svg = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("SVG exported");
        assert!(svg.contains("<svg") && svg.len() > 250);

        run_deferred_export(&mut fw, ExportFormat::Png);
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        assert!(fw.export_queue.last_exported_bytes.is_some());
    }
}

#[test]
fn test_sports_analytics_tab_autodiscovery_and_seven_tools() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let app = MathExplorerApp::default();
    let tab = app
        .tabs
        .iter()
        .find(|t| t.url_slug() == "sports_analytics")
        .expect("SportsAnalyticsTab must be auto-discovered by build.rs");
    assert_eq!(tab.name(), "Sports Analytics");
    assert_eq!(tab.academic_domain(), "Applied Math");

    let mut local = SportsAnalyticsTab::default();
    for tool_id in ALL_SPORTS_TOOLS {
        assert!(
            local.select_tool_by_name(tool_id),
            "Failed to select {tool_id}"
        );
        assert_eq!(local.active_tool_id(), Some(tool_id));
    }
}

#[test]
fn test_sports_analytics_presets_exports_and_deny_unknown_fields() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    for tool_id in ALL_SPORTS_TOOLS {
        let meta = inventory::iter::<ToolMetadata>
            .into_iter()
            .find(|m| m.id == tool_id)
            .unwrap_or_else(|| panic!("Missing ToolMetadata for {tool_id}"));
        let mut tool = (meta.build)();
        let caps = tool.export_capabilities();
        assert!(
            caps.figure && caps.dataset,
            "{tool_id} must enable plot exports"
        );
        let datasets = tool.export_datasets();
        assert!(
            !datasets.is_empty(),
            "{tool_id} must export non-empty datasets"
        );
        assert!(
            datasets.iter().all(|(_, pts)| !pts.is_empty()),
            "{tool_id} exported an empty curve"
        );

        let saved = tool.save_state().expect("serializable state");
        assert!(saved.len() <= 2048);
        tool.try_load_state(&saved).expect("roundtrip load");

        let mut val: serde_json::Value = serde_json::from_str(&saved).unwrap();
        val.as_object_mut()
            .unwrap()
            .insert("unexpected_extra_key".into(), serde_json::json!(1));
        assert!(
            tool.try_load_state(&val.to_string()).is_err(),
            "{tool_id} must reject unknown fields via #[serde(deny_unknown_fields)]"
        );

        let paper_links = resolve_portal_paper_links(tool_id, "sports_analytics");
        assert!(!paper_links.is_empty());
        assert_eq!(
            paper_links[0].paper_name,
            "sports_quantitative_modeling.tex"
        );
        let bib_links = resolve_citation_reference_links(&tool.theory().theory_citation());
        assert!(
            !bib_links.is_empty(),
            "{tool_id} citation must resolve at least one BibTeX external link"
        );
    }

    let sports_presets: Vec<_> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.domain == "sports_analytics")
        .collect();
    assert_eq!(sports_presets.len(), 7);
}

#[test]
fn test_sports_analytics_renders_across_all_three_themes() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    let mut frame = eframe::Frame::_new_kittest();
    let mut app = MathExplorerApp::default();
    let idx = app
        .tabs
        .iter()
        .position(|t| t.url_slug() == "sports_analytics")
        .unwrap();
    app.select_tab(&ctx, idx);
    for tool_id in ALL_SPORTS_TOOLS {
        assert!(app.tabs[idx].select_tool_by_name(tool_id));
        for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
            app.theme = theme;
            let _ = ctx.run(egui::RawInput::default(), |ctx| {
                eframe::App::update(&mut app, ctx, &mut frame);
            });
        }
    }
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_sports_analytics_tab_and_tools_roundtrip() {
        let mut tab = SportsAnalyticsTab::default();
        assert_eq!(tab.url_slug(), "sports_analytics");
        for tool_id in ALL_SPORTS_TOOLS {
            assert!(tab.select_tool_by_name(tool_id));
        }
    }
}

// [cite:sports_quantitative_modeling]
