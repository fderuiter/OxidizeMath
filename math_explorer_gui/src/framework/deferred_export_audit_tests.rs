#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use egui_plot::export::ExportOptions;
use std::collections::{BTreeSet, HashSet};

#[path = "subplot_painter_export_tests.rs"]
mod subplot_painter_export_tests;

pub(super) fn build_tool_by_id(id: &str) -> Box<dyn InteractiveTool> {
    let meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == id || m.display_name == id)
        .unwrap_or_else(|| panic!("Missing registered tool '{id}'"));
    (meta.build)()
}

pub(super) fn run_modal_export_after_clear(
    framework: &mut SimulationFramework,
    format: ExportFormat,
) {
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
        framework.show(ctx, "audit_framework");
    });
}

fn assert_no_unnamed_series_in_tabular_exports(
    meta_name: &str,
    framework: &mut SimulationFramework,
) {
    run_modal_export_after_clear(framework, ExportFormat::Csv);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success
    );
    let csv = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing CSV content for {meta_name}"));
    for line in csv.lines().skip(1) {
        let first_col = line.split(',').next().unwrap_or("");
        assert!(
            !first_col.trim().trim_matches('"').is_empty(),
            "Unnamed CSV series row in {meta_name}: {line}"
        );
    }

    run_modal_export_after_clear(framework, ExportFormat::Json);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success
    );
    let json_raw = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing JSON content for {meta_name}"));
    let parsed: serde_json::Value = serde_json::from_str(json_raw)
        .unwrap_or_else(|e| panic!("Invalid JSON for {meta_name}: {e}"));
    let arr = parsed.as_array().expect("Expected JSON array");
    assert!(!arr.is_empty(), "Empty JSON dataset array for {meta_name}");
    for entry in arr {
        let name = entry
            .get("dataset")
            .or_else(|| entry.get("name"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        assert!(
            !name.trim().is_empty(),
            "Unnamed JSON series in {meta_name}: {entry}"
        );
    }
}

pub(super) fn verify_single_figure_tool(
    meta_name: &str,
    tool: Box<dyn InteractiveTool>,
    check_tabular: bool,
) {
    let mut framework = SimulationFramework::new("__audit__");
    framework.export_options.width = 320;
    framework.export_options.height = 240;
    framework.active_tool = Some(tool);

    run_modal_export_after_clear(&mut framework, ExportFormat::Png);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success
    );
    let png_bytes = framework
        .export_queue
        .last_exported_bytes
        .as_deref()
        .unwrap_or_else(|| panic!("Missing PNG bytes for {meta_name}"));
    let (_, _, pixels) = egui_plot::export_primitives::decode_png_rgba(png_bytes)
        .unwrap_or_else(|e| panic!("Invalid PNG for {meta_name}: {e}"));
    let colors: HashSet<[u8; 4]> = pixels.into_iter().collect();
    assert!(colors.len() >= 2, "Blank PNG for {meta_name}");

    run_modal_export_after_clear(&mut framework, ExportFormat::Svg);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success
    );
    let svg = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing SVG content for {meta_name}"));
    assert!(svg.contains("<svg") && svg.len() > 250);

    if check_tabular {
        assert_no_unnamed_series_in_tabular_exports(meta_name, &mut framework);
    }
}

#[test]
fn test_all_registered_tools_capabilities_and_deferred_modal_exports() {
    let mut plot_tool_count = 0usize;
    let mut raster_tool_count = 0usize;
    let mut unsupported_ids = BTreeSet::new();

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        let tool = (meta.build)();
        let caps = tool.export_capabilities();

        if !caps.figure {
            unsupported_ids.insert(meta.id);
            assert!(!caps.dataset);
            assert!(!caps.dataset_disabled_reason.is_empty());
            assert!(!caps.figure_disabled_reason.is_empty());
            continue;
        }

        if caps.dataset {
            plot_tool_count += 1;
        } else {
            raster_tool_count += 1;
        }
        verify_single_figure_tool(meta.id, tool, caps.dataset);
    }

    let expected_exemptions: BTreeSet<&str> = [
        "BetSizeCalculatorTool",
        "CeraTool",
        "FactorizationTool",
        "Gaussian Splatting",
        "LifetimeEstimatorTool",
        "RandomizationTool",
        "SampleSizeCalculatorTool",
        "TokenizationTool",
        "UalbfWidget",
    ]
    .into_iter()
    .collect();

    assert_eq!(plot_tool_count, 37, "Expected 37 Plot dataset+figure tools");
    assert_eq!(raster_tool_count, 6, "Expected 6 raster figure tools");
    assert_eq!(plot_tool_count + raster_tool_count, 43);
    assert_eq!(unsupported_ids, expected_exemptions);
}

fn count_non_bg_in_region(
    pixels: &[[u8; 4]],
    width: usize,
    x_range: std::ops::Range<usize>,
    y_range: std::ops::Range<usize>,
) -> usize {
    let bg = pixels[0];
    pixels
        .chunks_exact(width)
        .skip(y_range.start)
        .take(y_range.len())
        .flat_map(|row| &row[x_range.clone()])
        .filter(|&&px| px != bg)
        .count()
}

fn verify_multi_plot_case(
    id: &str,
    panel_a: &str,
    panel_b: &str,
    horizontal_split: bool,
    verify_gif: bool,
) {
    let mut tool = build_tool_by_id(id);
    if id == "SpikeAnalysisTool" {
        for _ in 0..320 {
            tool.step_simulation();
        }
    } else if id == "TrainingMonitorTool" {
        for _ in 0..6 {
            tool.step_simulation();
        }
    }

    let mut framework = SimulationFramework::new("__subplot_test__");
    framework.export_options.width = 560;
    framework.export_options.height = 360;
    framework.export_gif_frames = 4;
    framework.active_tool = Some(tool);

    run_modal_export_after_clear(&mut framework, ExportFormat::Svg);
    let svg = framework
        .export_queue
        .last_exported_content
        .clone()
        .expect("subplot SVG");
    assert!(svg.contains(&format!("data-subplot-id=\"{panel_a}\"")));
    assert!(svg.contains(&format!("data-subplot-id=\"{panel_b}\"")));

    run_modal_export_after_clear(&mut framework, ExportFormat::Png);
    let png = framework
        .export_queue
        .last_exported_bytes
        .clone()
        .expect("subplot PNG");
    if id == "TrainingMonitorTool" {
        let _ = std::fs::write("/tmp/training_monitor_subplots.png", &png);
    }
    let (w, h, pixels) = egui_plot::export_primitives::decode_png_rgba(&png).expect("valid PNG");
    let (first_hits, second_hits) = if horizontal_split {
        (
            count_non_bg_in_region(&pixels, w, 0..(w / 2 - 10), 0..h),
            count_non_bg_in_region(&pixels, w, (w / 2 + 10)..w, 0..h),
        )
    } else {
        (
            count_non_bg_in_region(&pixels, w, 0..w, 0..(h / 2 - 10)),
            count_non_bg_in_region(&pixels, w, 0..w, (h / 2 + 10)..h),
        )
    };
    assert!(first_hits > 10 && second_hits > 10);

    if verify_gif {
        run_modal_export_after_clear(&mut framework, ExportFormat::Gif);
        let gif = framework
            .export_queue
            .last_exported_bytes
            .as_deref()
            .expect("subplot GIF");
        assert!(gif.starts_with(b"GIF89a") && gif.len() > 256);
    }
}

#[test]
fn test_multi_plot_tools_disjoint_viewports_and_stepped_gifs() {
    for (id, panel_a, panel_b, horizontal, gif) in [
        (
            "TrainingMonitorTool",
            "loss_history",
            "accuracy_history",
            true,
            true,
        ),
        (
            "SpikeAnalysisTool",
            "voltage_trace",
            "isi_histogram",
            false,
            true,
        ),
        (
            "ReplicatorDynamicsTool",
            "replicator_trajectories",
            "population_distribution",
            false,
            true,
        ),
        ("ComplexMappingTool", "z_plane", "w_plane", true, false),
    ] {
        verify_multi_plot_case(id, panel_a, panel_b, horizontal, gif);
    }
}

#[test]
fn test_complex_mapping_inverse_pole_preserves_z_plane_scale() {
    let mut tool = build_tool_by_id("ComplexMappingTool");
    tool.try_load_state(
        r#"{"function":"Inverse","grid_min":-2.0,"grid_max":2.0,"grid_density":9,"inspection_point":[0.0,0.0]}"#,
    )
    .expect("load w = 1/z state with pole at origin");

    let panels = tool.export_subplot_panels();
    assert_eq!(panels.len(), 2);
    let z_bounds = egui_plot::export_primitives::compute_primitive_bounds(&panels[0].primitives)
        .expect("z_plane finite bounds");
    let w_bounds = egui_plot::export_primitives::compute_primitive_bounds(&panels[1].primitives)
        .expect("w_plane finite bounds");
    assert_eq!(z_bounds, [-2.0, 2.0, -2.0, 2.0]);
    assert!(w_bounds[1] >= 1e6, "w_plane should clamp pole to 1e6");

    let opts = ExportOptions {
        width: 640,
        height: 360,
        ..Default::default()
    };
    let png = egui_plot::export_subplot_panels_png(&panels, &opts).expect("pole subplot PNG");
    let _ = std::fs::write("/tmp/complex_mapping_pole_subplots.png", &png);
    let svg = egui_plot::export_subplot_panels_svg(&panels, &opts).expect("pole subplot SVG");
    let z_section = svg
        .split("data-subplot-id=\"w_plane\"")
        .next()
        .expect("z_plane SVG section");
    assert!(z_section.contains("data-subplot-id=\"z_plane\"") && z_section.contains("<polyline"));
}
