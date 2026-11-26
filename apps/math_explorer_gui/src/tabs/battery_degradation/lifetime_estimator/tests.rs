#![cfg_attr(any(), verified(opt_out = "tests"))]

// [cite:battery_degradation]

use super::duty::{DutyPhaseKind, DutySegmentSpec};
use super::*;
use crate::framework::InteractiveTool;
use crate::tabs::battery_degradation::cell_card::ChemistryChoice;
use crate::url_state::URL_JSON_BOUNDS;

#[test]
fn test_lifetime_estimator_plot_capabilities_and_coupled_thermal_aging() {
    let mut tool = LifetimeEstimatorTool::default();
    let caps = tool.export_capabilities();
    assert!(caps.figure && caps.dataset);

    let datasets = tool.export_datasets();
    assert_eq!(datasets.len(), 2);
    assert!(datasets[0].1.len() > 50);

    let base_summary = tool.lifetime_summary();
    tool.temperature = 45.0;
    let hot_summary = tool.lifetime_summary();
    assert!(
        hot_summary.years_to_80 < base_summary.years_to_80 * 0.55,
        "45 °C must accelerate coupled warranty degradation: 25 °C = {:.2} yr vs 45 °C = {:.2} yr",
        base_summary.years_to_80,
        hot_summary.years_to_80
    );
}

#[test]
fn test_max_8_duty_segments_storage_zero_cycles_and_url_json_bounds() {
    let tool = LifetimeEstimatorTool {
        chemistry: ChemistryChoice::Lfp,
        duty_segments: vec![
            DutySegmentSpec::from_kind(DutyPhaseKind::Commute),
            DutySegmentSpec::from_kind(DutyPhaseKind::FastCharge),
            DutySegmentSpec::from_kind(DutyPhaseKind::Storage),
            DutySegmentSpec::from_kind(DutyPhaseKind::Commute),
            DutySegmentSpec::from_kind(DutyPhaseKind::FastCharge),
            DutySegmentSpec::from_kind(DutyPhaseKind::Storage),
            DutySegmentSpec::from_kind(DutyPhaseKind::Commute),
            DutySegmentSpec::from_kind(DutyPhaseKind::Storage),
        ],
        ..Default::default()
    };
    assert_eq!(tool.duty_segments.len(), MAX_DUTY_SEGMENTS);

    let state = tool.save_state().expect("serialize 8-segment duty state");
    URL_JSON_BOUNDS
        .validate_json(&state)
        .expect("8-segment duty payload must fit within URL_JSON_BOUNDS (4096 bytes)");

    let mut restored = LifetimeEstimatorTool::default();
    restored.try_load_state(&state).expect("restore 8 segments");
    assert_eq!(restored.duty_segments.len(), 8);
    assert_eq!(restored.chemistry, ChemistryChoice::Lfp);

    restored.duty_segments = vec![DutySegmentSpec::from_kind(DutyPhaseKind::Storage)];
    let storage = restored.lifetime_summary();
    assert!(storage.years_to_80.is_finite() && storage.years_to_80 > 0.0);
    assert!(storage.years_to_70 > storage.years_to_80);
    assert_eq!(storage.cycles_to_80, 0.0);

    let csv = restored.build_warranty_schedule_csv();
    assert!(csv.contains("80% First-Life Warranty"));
    assert!(csv.contains("70% End-of-Life Warranty"));
    assert!(csv.lines().count() >= 5);

    let card_json = restored
        .build_cell_card()
        .to_pretty_json()
        .expect("cell card json");
    assert!(card_json.contains("\"tool\": \"LifetimeEstimatorTool\""));
}

#[test]
fn test_lifetime_estimator_deferred_png_and_svg_export() {
    let tool = LifetimeEstimatorTool {
        chemistry: ChemistryChoice::Nmc811,
        duty_segments: vec![
            DutySegmentSpec::from_kind(DutyPhaseKind::Commute),
            DutySegmentSpec::from_kind(DutyPhaseKind::FastCharge),
            DutySegmentSpec::from_kind(DutyPhaseKind::Storage),
        ],
        ..Default::default()
    };
    let mut fw = crate::framework::SimulationFramework::new("__lifetime_export__");
    fw.active_tool = Some(Box::new(tool));
    let ctx = eframe::egui::Context::default();
    let raw = eframe::egui::RawInput {
        screen_rect: Some(eframe::egui::Rect::from_min_size(
            eframe::egui::Pos2::ZERO,
            eframe::egui::Vec2::new(1100.0, 760.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(raw.clone(), |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(crate::framework::ExportFormat::Png);
        fw.show(ctx, "lifetime_export_png");
    });
    let png = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("PNG exported");
    assert!(png.len() > 200);

    let _ = ctx.run(raw, |ctx| {
        egui_plot::clear_last_plot_datasets();
        fw.show_export_modal = true;
        fw.queue_export(crate::framework::ExportFormat::Svg);
        fw.show(ctx, "lifetime_export_svg");
    });
    let svg = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("SVG exported");
    assert!(svg.contains("<svg"));
}
