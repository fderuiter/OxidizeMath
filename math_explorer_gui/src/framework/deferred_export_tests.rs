#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};
use eframe::egui;
use std::collections::HashSet;

fn build_registered_tool(name: &str) -> Box<dyn InteractiveTool> {
    let meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == name || m.display_name == name)
        .unwrap_or_else(|| panic!("Missing registered tool {name}"));
    (meta.build)()
}

fn new_framework_with_tool(tool: Box<dyn InteractiveTool>) -> SimulationFramework {
    let mut framework = SimulationFramework::new("__export_test__");
    framework.export_options.width = 240;
    framework.export_options.height = 180;
    framework.export_options.scale = 1.0;
    framework.active_tool = Some(tool);
    framework.export_gif_frames = 3;
    framework
}

fn run_deferred_frame(framework: &mut SimulationFramework, format: ExportFormat) {
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
        framework.show(ctx, "export_test_framework");
    });
}

pub(crate) fn decode_rgba_pixels(png_bytes: &[u8]) -> Result<Vec<[u8; 4]>, String> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png_bytes))
        .read_info()
        .map_err(|e| format!("Failed to parse PNG header: {e}"))?;
    let buf_len = reader
        .output_buffer_size()
        .ok_or_else(|| "PNG buffer size overflow".to_string())?;
    let mut buf = vec![0_u8; buf_len];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("Failed to decode PNG frame: {e}"))?;
    if info.color_type != png::ColorType::Rgba || info.bit_depth != png::BitDepth::Eight {
        return Err(format!(
            "Expected 8-bit RGBA PNG, got {:?} {:?}",
            info.color_type, info.bit_depth
        ));
    }
    let (pixels, rem) = buf[..info.buffer_size()].as_chunks::<4>();
    if !rem.is_empty() {
        return Err("Misaligned RGBA pixel buffer".to_string());
    }
    Ok(pixels.to_vec())
}

fn count_distinct_colors(png_bytes: &[u8]) -> usize {
    let pixels = decode_rgba_pixels(png_bytes).expect("valid PNG");
    let colors: HashSet<[u8; 4]> = pixels.into_iter().collect();
    colors.len()
}

fn verify_bar_or_arrow_tool(
    tool: Box<dyn InteractiveTool>,
    expected_series: &[&str],
    expected_svg_tag: &str,
    screenshot_path: Option<&str>,
) {
    let mut framework = new_framework_with_tool(tool);
    run_deferred_frame(&mut framework, ExportFormat::Csv);
    let csv = framework
        .export_queue
        .last_exported_content
        .clone()
        .expect("CSV content generated");
    assert!(csv.lines().count() >= 2, "CSV should have data rows: {csv}");
    for series in expected_series {
        assert!(
            csv.contains(series),
            "CSV missing expected series '{series}'"
        );
    }

    run_deferred_frame(&mut framework, ExportFormat::Json);
    let json = framework
        .export_queue
        .last_exported_content
        .clone()
        .expect("JSON content generated");
    for series in expected_series {
        assert!(
            json.contains(series),
            "JSON missing expected series '{series}'"
        );
    }

    run_deferred_frame(&mut framework, ExportFormat::Png);
    let png = framework
        .export_queue
        .last_exported_bytes
        .clone()
        .expect("PNG bytes generated");
    assert!(
        count_distinct_colors(&png) >= 2,
        "PNG must contain non-background figure pixels"
    );
    if let Some(path) = screenshot_path {
        let _ = std::fs::write(path, &png);
    }

    run_deferred_frame(&mut framework, ExportFormat::Svg);
    let svg = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("SVG content generated");
    assert!(
        svg.contains(expected_svg_tag),
        "SVG missing expected primitive tag '{expected_svg_tag}'"
    );
}

#[test]
fn test_bar_and_arrow_csv_json_and_figure_exports() {
    verify_bar_or_arrow_tool(
        build_registered_tool("PartitionsWidget"),
        &["Coefficients"],
        "<rect",
        None,
    );
    verify_bar_or_arrow_tool(
        build_registered_tool("ClebschGordanTool"),
        &["Coefficient Value"],
        "<rect",
        None,
    );
    verify_bar_or_arrow_tool(
        build_registered_tool("PotentialFlowTool"),
        &["Vector Field"],
        "<polygon",
        Some("/tmp/potential_flow_arrows.png"),
    );
    verify_bar_or_arrow_tool(
        build_registered_tool("ReplicatorDynamicsTool"),
        &["Hawk", "Dove", "Distribution"],
        "<rect",
        Some("/tmp/replicator_mixed_bars.png"),
    );

    let mut spike_tool = build_registered_tool("SpikeAnalysisTool");
    for _ in 0..350 {
        spike_tool.step_simulation();
    }
    verify_bar_or_arrow_tool(
        spike_tool,
        &["Voltage", "Spikes", "ISI Distribution"],
        "<rect",
        None,
    );
}

#[test]
fn test_raster_texture_tools_png_svg_gif_exports() {
    let cases = [
        ("DoseCalculationTool", Some("/tmp/dose_raster_heatmap.png")),
        ("LatticeBoltzmannTool", None),
        ("FractalViewer", Some("/tmp/fractal_mandelbrot_raster.png")),
        ("IsingModelTool", None),
    ];

    for (tool_name, screenshot_path) in cases {
        verify_single_raster_tool(tool_name, screenshot_path);
    }
}

fn verify_single_raster_tool(tool_name: &str, screenshot_path: Option<&str>) {
    let tool = build_registered_tool(tool_name);
    let caps = tool.export_capabilities();
    assert!(caps.dataset && caps.figure, "{tool_name} caps invalid");

    let mut framework = new_framework_with_tool(tool);
    run_deferred_frame(&mut framework, ExportFormat::Csv);
    let csv = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("{tool_name} CSV missing"));
    let line_count = csv.lines().count();
    assert!(
        (2..=crate::framework::MAX_EXPORT_MATRIX_ENTRIES + 16).contains(&line_count),
        "{tool_name} CSV row count out of bounds: {line_count}"
    );

    run_deferred_frame(&mut framework, ExportFormat::Json);
    let json = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("{tool_name} JSON missing"));
    assert!(serde_json::from_str::<serde_json::Value>(json).is_ok());

    run_deferred_frame(&mut framework, ExportFormat::Png);
    let png = framework
        .export_queue
        .last_exported_bytes
        .clone()
        .unwrap_or_else(|| panic!("{tool_name} PNG missing"));
    assert!(count_distinct_colors(&png) >= 2, "{tool_name} PNG blank");
    if let Some(path) = screenshot_path {
        let _ = std::fs::write(path, &png);
    }

    run_deferred_frame(&mut framework, ExportFormat::Svg);
    let svg = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("{tool_name} SVG missing"));
    assert!(svg.contains("data:image/png;base64,"));

    run_deferred_frame(&mut framework, ExportFormat::Gif);
    let gif = framework
        .export_queue
        .last_exported_bytes
        .as_deref()
        .unwrap_or_else(|| panic!("{tool_name} GIF missing"));
    assert!(gif.starts_with(b"GIF89a"), "{tool_name} invalid GIF header");
}

#[test]
fn test_visual_custom_painter_tools_png_svg_gif_and_tables() {
    let cases = [
        (
            "Gaussian Splatting",
            false,
            Some("/tmp/gaussian_splatting_export.png"),
        ),
        ("CeraTool", true, Some("/tmp/cera_model_export.png")),
        (
            "QTableInspectorTool",
            true,
            Some("/tmp/q_table_inspector_export.png"),
        ),
        ("GridWorldTool", true, None),
        (
            "NeuralNetworkVizTool",
            true,
            Some("/tmp/neural_network_viz_export.png"),
        ),
        ("NetworkPropagationTool", false, None),
        ("AlgorithmVisualizerTool", true, None),
        (
            "NetworkMetricsTool",
            false,
            Some("/tmp/network_metrics_export.png"),
        ),
    ];

    for (name, step_first, screenshot) in cases {
        verify_single_visual_tool(name, step_first, screenshot);
    }
    verify_hydrated_graph_editor();
}

fn verify_single_visual_tool(name: &str, step_first: bool, screenshot: Option<&str>) {
    let mut tool = build_registered_tool(name);
    if step_first {
        tool.step_simulation();
    }
    let mut fw = new_framework_with_tool(tool);
    if screenshot.is_some() {
        fw.export_options.width = 640;
        fw.export_options.height = 480;
    }
    run_deferred_frame(&mut fw, ExportFormat::Csv);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    run_deferred_frame(&mut fw, ExportFormat::Json);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    run_deferred_frame(&mut fw, ExportFormat::Png);
    let png = fw.export_queue.last_exported_bytes.clone().expect("PNG");
    assert!(count_distinct_colors(&png) >= 2, "{name} PNG blank");
    if let Some(path) = screenshot {
        let _ = std::fs::write(path, &png);
        fw.export_options.width = 240;
        fw.export_options.height = 180;
    }
    run_deferred_frame(&mut fw, ExportFormat::Svg);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    run_deferred_frame(&mut fw, ExportFormat::Gif);
    let gif = fw.export_queue.last_exported_bytes.as_deref().expect("GIF");
    assert!(gif.starts_with(b"GIF89a"), "{name} GIF invalid");
}

fn verify_hydrated_graph_editor() {
    let mut graph_editor = build_registered_tool("Graph Editor");
    let preset = inventory::iter::<crate::starter_gallery::StarterPresetMetadata>
        .into_iter()
        .find(|p| p.id == "graph_editor_starter")
        .expect("graph_editor_starter preset");
    let cfg = serde_json::from_str::<serde_json::Value>(preset.generate_preset()).unwrap()
        ["config"]
        .to_string();
    graph_editor.try_load_state(&cfg).expect("load graph state");
    let mut ge_fw = new_framework_with_tool(graph_editor);
    ge_fw.export_options.width = 640;
    ge_fw.export_options.height = 480;
    run_deferred_frame(&mut ge_fw, ExportFormat::Png);
    let ge_png = ge_fw.export_queue.last_exported_bytes.clone().expect("PNG");
    assert!(count_distinct_colors(&ge_png) >= 2);
    let _ = std::fs::write("/tmp/graph_editor_export.png", &ge_png);
    ge_fw.export_options.width = 240;
    ge_fw.export_options.height = 180;
    run_deferred_frame(&mut ge_fw, ExportFormat::Gif);
    assert!(ge_fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("Graph Editor GIF")
        .starts_with(b"GIF89a"));
}

struct EmptyPlotTool;
impl scientific_metadata::theory::TheoryDescribable for EmptyPlotTool {
    fn theory_description(&self) -> String {
        "Empty test tool".into()
    }
    fn phonetic_description(&self) -> String {
        "Empty test tool".into()
    }
    fn theory_citation(&self) -> String {
        "Test".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
impl InteractiveTool for EmptyPlotTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Empty Plot Tool"
    }
    fn show(&mut self, _ctx: &egui::Context) {}
}

#[test]
fn test_empty_payload_rejection_status_colors_and_live_region() {
    clear_test_announcements();
    let temp_dir = std::env::temp_dir().join("oxidize_empty_export_guard_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("temp dir created");

    let unpopulated = build_registered_tool("FactorizationTool");
    let caps = unpopulated.export_capabilities();
    assert!(!caps.dataset && !caps.figure);
    assert!(!caps.dataset_disabled_reason.is_empty());
    assert!(!caps.figure_disabled_reason.is_empty());

    let mut unpopulated_fw = new_framework_with_tool(unpopulated);
    unpopulated_fw.export_queue.destination_dir = Some(temp_dir.clone());
    run_deferred_frame(&mut unpopulated_fw, ExportFormat::Csv);
    assert_eq!(
        unpopulated_fw.export_status_severity,
        ExportStatusSeverity::Warning
    );
    assert_eq!(
        unpopulated_fw.export_status_severity.color(),
        crate::app::palette::DARK_PALETTE.warning
    );
    assert_eq!(with_live_region(|r| r.active_slot), 1);

    let mut empty_fw = new_framework_with_tool(Box::new(EmptyPlotTool));
    empty_fw.export_queue.destination_dir = Some(temp_dir.clone());
    run_deferred_frame(&mut empty_fw, ExportFormat::Csv);
    assert_eq!(empty_fw.export_status_severity, ExportStatusSeverity::Error);
    assert_eq!(
        empty_fw.export_status_severity.color(),
        crate::app::palette::DARK_PALETTE.error
    );
    assert_eq!(with_live_region(|r| r.active_slot), 0);

    run_deferred_frame(&mut empty_fw, ExportFormat::Json);
    assert_eq!(empty_fw.export_status_severity, ExportStatusSeverity::Error);
    assert_eq!(with_live_region(|r| r.active_slot), 1);

    run_deferred_frame(&mut empty_fw, ExportFormat::Png);
    assert_eq!(empty_fw.export_status_severity, ExportStatusSeverity::Error);
    assert_eq!(with_live_region(|r| r.active_slot), 0);

    let entries: Vec<_> = std::fs::read_dir(&temp_dir)
        .expect("read temp dir")
        .collect();
    assert!(
        entries.is_empty(),
        "Zero empty CSV/JSON/PNG files may be written to disk"
    );

    let mut ok_fw = new_framework_with_tool(build_registered_tool("PartitionsWidget"));
    run_deferred_frame(&mut ok_fw, ExportFormat::Png);
    assert_eq!(ok_fw.export_status_severity, ExportStatusSeverity::Success);
    assert_eq!(
        ok_fw.export_status_severity.color(),
        crate::app::palette::DARK_PALETTE.success
    );

    let announcements = all_announcements();
    assert!(announcements.iter().any(|(_, p)| p == "assertive"));
    assert!(announcements.iter().any(|(_, p)| p == "polite"));
    let _ = std::fs::remove_dir_all(&temp_dir);
}

#[test]
fn test_deferred_png_clipboard_and_raster_csv_clipboard_capability_guard() {
    let mut ok_fw = new_framework_with_tool(build_registered_tool("PartitionsWidget"));
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui_plot::clear_last_plot_datasets();
        ok_fw.show_export_modal = true;
        ok_fw
            .export_queue
            .push_target(ExportFormat::Png, ExportTarget::Clipboard);
        ok_fw.show(ctx, "export_clipboard_img_test");
    });
    assert_eq!(ok_fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(ok_fw
        .export_status_message
        .as_deref()
        .unwrap_or("")
        .contains("clipboard"));

    let mut raster_fw = new_framework_with_tool(build_registered_tool("DoseCalculationTool"));
    raster_fw.queue_clipboard_export();
    raster_fw.dispatch_pending_exports_with_ctx(&ctx);
    assert_eq!(
        raster_fw.export_status_severity,
        ExportStatusSeverity::Success
    );
    assert!(raster_fw.export_queue.last_clipboard_content.is_some());

    let mut unpop_fw = new_framework_with_tool(build_registered_tool("FactorizationTool"));
    unpop_fw.queue_clipboard_export();
    unpop_fw.dispatch_pending_exports_with_ctx(&ctx);
    assert_eq!(
        unpop_fw.export_status_severity,
        ExportStatusSeverity::Warning
    );
    assert!(unpop_fw.export_queue.last_clipboard_content.is_none());
}
