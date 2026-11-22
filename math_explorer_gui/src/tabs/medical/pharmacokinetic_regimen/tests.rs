//! Unit and URL-state contract tests for [`super::PharmacokineticRegimenTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use egui_plot::export::ExportOptions;
use egui_plot::export_primitives::{
    decode_png_rgba, export_primitives_svg, render_primitives_image,
};
use math_explorer::applied::pharmacokinetics::{MAX_REGIMEN_EVENTS, t_max};

use crate::framework::{InteractiveTool, StarterPresetMetadata};
use crate::url_state::{
    MAX_DECODED_URL_STATE_BYTES, MAX_URL_FRAGMENT_BYTES, encode_share_fragment,
    parse_share_location,
};

use super::PharmacokineticRegimenTool;
use super::export::{format_exposure_report_json, format_schedule_csv};
use super::state::{CompactDoseEvent, RegimenState};

#[test]
fn test_max_64_events_fits_url_json_bounds_and_roundtrips_share_link() {
    let mut tool = PharmacokineticRegimenTool::default();
    tool.state.events = (0..MAX_REGIMEN_EVENTS)
        .map(|i| CompactDoseEvent {
            t: i as f64 * 12.0,
            d: if i == 0 { 200.0 } else { 100.0 },
            xr: i % 2 == 1,
        })
        .collect();
    tool.recalculate();

    let compact_envelope = crate::presets::serialize_compact_preset_envelope(
        tool.name(),
        tool.domain(),
        &tool.state,
    )
    .expect("compact preset envelope");
    assert!(
        compact_envelope.len() <= MAX_DECODED_URL_STATE_BYTES,
        "64-event envelope was {} bytes > {MAX_DECODED_URL_STATE_BYTES}",
        compact_envelope.len()
    );

    let fragment = encode_share_fragment(
        "medical",
        Some("PharmacokineticRegimenTool"),
        Some(&compact_envelope),
    )
    .expect("share fragment");
    assert!(fragment.starts_with("#tab=medical&tool=PharmacokineticRegimenTool&state="));
    assert!(fragment.len() <= MAX_URL_FRAGMENT_BYTES);

    let parsed = parse_share_location(&fragment, "")
        .expect("parse share location")
        .expect("share params");
    let mut restored = PharmacokineticRegimenTool::default();
    restored
        .import_preset_json(parsed.state_json.as_deref().unwrap())
        .expect("import restored 64-event preset");
    assert_eq!(restored.state, tool.state);
}

#[test]
fn test_starter_presets_verify_loading_dose_and_missed_dose_metrics() {
    let presets: Vec<&'static StarterPresetMetadata> = inventory::iter::<StarterPresetMetadata>
        .into_iter()
        .filter(|p| p.tool_name == Some("PharmacokineticRegimenTool"))
        .collect();
    assert_eq!(presets.len(), 3);

    let default_tool = PharmacokineticRegimenTool::default();
    let maint_summary = default_tool.curves.summary.expect("default summary");
    assert!((maint_summary.time_in_range_pct - 66.4932).abs() < 1e-3);

    let loading_preset = presets
        .iter()
        .find(|p| p.id == "pk_loading_dose_window")
        .unwrap();
    let mut loading_tool = PharmacokineticRegimenTool::default();
    loading_tool
        .import_preset_json(loading_preset.generate_preset())
        .unwrap();
    let load_summary = loading_tool.curves.summary.expect("loading summary");
    assert!((load_summary.time_in_range_pct - 86.8887).abs() < 1e-3);
    assert_eq!(load_summary.toxic_pct, 0.0);

    let missed_preset = presets
        .iter()
        .find(|p| p.id == "pk_missed_dose_recovery")
        .unwrap();
    let mut catchup_tool = PharmacokineticRegimenTool::default();
    catchup_tool
        .import_preset_json(missed_preset.generate_preset())
        .unwrap();
    let catchup_summary = catchup_tool.curves.summary.expect("catchup summary");
    assert_eq!(catchup_summary.toxic_pct, 0.0);

    let double_tool =
        PharmacokineticRegimenTool::with_state(RegimenState::missed_dose_double_scenario());
    let double_summary = double_tool.curves.summary.expect("double summary");
    assert!((double_summary.toxic_pct - 8.3247).abs() < 1e-3);
}

#[test]
fn test_calibration_and_svg_png_csv_json_exports() {
    let mut tool = PharmacokineticRegimenTool::with_state(RegimenState::chiral_xr_scenario());
    tool.state.cal_t_max_h = 2.5;
    tool.state.cal_half_life_h = 8.0;
    assert!(tool.apply_ka_calibration());
    tool.recalculate();
    assert!((t_max(tool.state.ka, tool.state.ke) - 2.5).abs() < 1e-5);

    let datasets = tool.export_datasets();
    assert!(datasets.len() >= 3);
    let prims = tool.export_plot_primitives();
    let options = ExportOptions::default();
    let svg = export_primitives_svg(&prims, &options).expect("SVG export");
    assert!(svg.contains("<svg") && svg.contains("<polygon"));

    let img = render_primitives_image(&prims, &options).expect("PNG raster");
    let png = egui_plot::export::color32_image_to_png(&img).expect("PNG bytes");
    let (w, h, pixels) = decode_png_rgba(&png).expect("decode PNG");
    assert!(w > 0 && h > 0 && !pixels.is_empty());

    let csv = format_schedule_csv(&tool.state, &tool.curves);
    assert!(csv.contains("event_index,time_h,dose_mg,formulation"));
    let summary = tool.curves.summary.expect("chiral summary");
    let report = format_exposure_report_json(&tool.state, &summary).expect("JSON report");
    assert!(report.contains("time_in_therapeutic_range_pct"));
    assert!(report.contains("not clinical prescribing advice"));
}
