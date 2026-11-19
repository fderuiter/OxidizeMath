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
    framework.active_tool = Some(tool);
    framework.export_gif_frames = 4;
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
    let _guard = crate::reflective_ui::tests::TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let _ = ctx.run(raw_input, |ctx| {
        egui_plot::clear_last_plot_datasets();
        framework.show_export_modal = true;
        framework.queue_export(format);
        framework.show(ctx, "export_test_framework");
    });
}

fn count_distinct_colors(png_bytes: &[u8]) -> usize {
    let (_, _, pixels) =
        egui_plot::export_primitives::decode_png_rgba(png_bytes).expect("valid PNG");
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
        let tool = build_registered_tool(tool_name);
        let caps = tool.export_capabilities();
        assert!(!caps.dataset, "{tool_name} should disable dataset exports");
        assert!(!caps.dataset_disabled_reason.is_empty());
        assert!(caps.figure, "{tool_name} should support figure exports");

        let mut framework = new_framework_with_tool(tool);
        run_deferred_frame(&mut framework, ExportFormat::Png);
        let png = framework
            .export_queue
            .last_exported_bytes
            .clone()
            .unwrap_or_else(|| panic!("{tool_name} PNG missing"));
        assert!(
            count_distinct_colors(&png) >= 2,
            "{tool_name} PNG should render non-blank raster field"
        );
        if let Some(path) = screenshot_path {
            let _ = std::fs::write(path, &png);
        }

        run_deferred_frame(&mut framework, ExportFormat::Svg);
        let svg = framework
            .export_queue
            .last_exported_content
            .as_deref()
            .unwrap_or_else(|| panic!("{tool_name} SVG missing"));
        assert!(
            svg.contains("data:image/png;base64,"),
            "{tool_name} SVG should embed raster PNG"
        );

        run_deferred_frame(&mut framework, ExportFormat::Gif);
        let gif = framework
            .export_queue
            .last_exported_bytes
            .as_deref()
            .unwrap_or_else(|| panic!("{tool_name} GIF missing"));
        assert!(gif.starts_with(b"GIF89a"), "{tool_name} invalid GIF header");
    }
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

    let unsupported = build_registered_tool("Gaussian Splatting");
    let caps = unsupported.export_capabilities();
    assert!(!caps.dataset && !caps.figure);
    assert!(!caps.dataset_disabled_reason.is_empty());
    assert!(!caps.figure_disabled_reason.is_empty());

    let mut unsupported_fw = new_framework_with_tool(unsupported);
    unsupported_fw.export_queue.destination_dir = Some(temp_dir.clone());
    run_deferred_frame(&mut unsupported_fw, ExportFormat::Csv);
    assert_eq!(
        unsupported_fw.export_status_severity,
        ExportStatusSeverity::Warning
    );
    assert_eq!(
        unsupported_fw.export_status_severity.color(),
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
    {
        let _guard = crate::reflective_ui::tests::TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let _ = ctx.run(egui::RawInput::default(), |ctx| {
            egui_plot::clear_last_plot_datasets();
            ok_fw.show_export_modal = true;
            ok_fw.queue_export_with_target(ExportFormat::Png, ExportTarget::Clipboard);
            ok_fw.show(ctx, "export_clipboard_img_test");
        });
    }
    assert_eq!(ok_fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(ok_fw
        .export_status_message
        .as_deref()
        .unwrap_or("")
        .contains("clipboard"));

    let mut raster_fw = new_framework_with_tool(build_registered_tool("DoseCalculationTool"));
    raster_fw.queue_clipboard_export();
    raster_fw.dispatch_pending_exports();
    assert_eq!(
        raster_fw.export_status_severity,
        ExportStatusSeverity::Warning
    );
    assert!(raster_fw.export_queue.last_clipboard_content.is_none());
}
