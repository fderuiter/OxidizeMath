#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use std::collections::HashSet;

fn run_modal_export_after_clear(framework: &mut SimulationFramework, format: ExportFormat) {
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

fn verify_single_figure_tool(meta_name: &str, tool: Box<dyn InteractiveTool>) {
    let mut framework = SimulationFramework::new("__audit__");
    framework.export_options.width = 320;
    framework.export_options.height = 240;
    framework.active_tool = Some(tool);

    run_modal_export_after_clear(&mut framework, ExportFormat::Png);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred modal PNG failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let png_bytes = framework
        .export_queue
        .last_exported_bytes
        .as_deref()
        .unwrap_or_else(|| panic!("Missing PNG bytes for {meta_name}"));
    let pixels = super::deferred_export_tests::decode_rgba_pixels(png_bytes)
        .unwrap_or_else(|e| panic!("Invalid PNG for {meta_name}: {e}"));
    let colors: HashSet<[u8; 4]> = pixels.into_iter().collect();
    assert!(
        colors.len() >= 2,
        "Deferred modal PNG for {meta_name} was blank (single solid color)"
    );

    run_modal_export_after_clear(&mut framework, ExportFormat::Svg);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred modal SVG failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let svg = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing SVG content for {meta_name}"));
    assert!(
        svg.contains("<svg") && svg.len() > 250,
        "Deferred modal SVG for {meta_name} was empty"
    );
}

fn verify_shader_capability(
    meta_id: &str,
    tool: &dyn InteractiveTool,
    caps: ExportCapabilities,
) -> bool {
    if !caps.shader {
        assert!(
            !caps.shader_disabled_reason.is_empty(),
            "{meta_id} disables shader export without an explanatory reason"
        );
        return false;
    }
    assert!(
        caps.shader_disabled_reason.is_empty(),
        "{meta_id} advertises shader export but has non-empty disabled reason"
    );
    let webgl2 = tool
        .export_glsl_shader(ShaderDialect::WebGl2Es300)
        .unwrap_or_else(|e| panic!("WebGL2 shader failed for {meta_id}: {e}"));
    assert!(
        webgl2.starts_with("#version 300 es") && webgl2.contains("void main()"),
        "Invalid WebGL2 shader header/entry for {meta_id}"
    );
    let shadertoy = tool
        .export_glsl_shader(ShaderDialect::Shadertoy)
        .unwrap_or_else(|e| panic!("Shadertoy shader failed for {meta_id}: {e}"));
    assert!(
        !shadertoy.contains("#version") && shadertoy.contains("void mainImage("),
        "Invalid Shadertoy entry for {meta_id}"
    );
    true
}

#[test]
fn test_all_registered_tools_capabilities_and_deferred_modal_exports() {
    let mut plot_tool_count = 0usize;
    let mut raster_tool_count = 0usize;
    let mut unsupported_tool_count = 0usize;
    let mut shader_tool_count = 0usize;
    let mut non_shader_tool_count = 0usize;

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        let tool = (meta.build)();
        let caps = tool.export_capabilities();
        if verify_shader_capability(meta.id, tool.as_ref(), caps) {
            shader_tool_count += 1;
        } else {
            non_shader_tool_count += 1;
        }

        if !caps.figure {
            unsupported_tool_count += 1;
            assert!(
                !caps.dataset,
                "{} should disable dataset export when figure is unsupported",
                meta.id
            );
            assert!(!caps.dataset_disabled_reason.is_empty());
            assert!(!caps.figure_disabled_reason.is_empty());
            continue;
        }

        if caps.dataset {
            plot_tool_count += 1;
        } else {
            raster_tool_count += 1;
        }
        verify_single_figure_tool(meta.id, tool);
    }

    let expected_plot_tools = if cfg!(feature = "applied") { 39 } else { 34 };
    assert_eq!(
        plot_tool_count, expected_plot_tools,
        "Unexpected Plot dataset+figure tool count"
    );
    assert_eq!(raster_tool_count, 7, "Expected 7 raster figure tools");
    let expected_unsupported = if cfg!(feature = "applied") { 16 } else { 12 };
    assert_eq!(
        unsupported_tool_count, expected_unsupported,
        "Unexpected non-export tool count"
    );
    assert_eq!(
        shader_tool_count, 3,
        "Expected 3 analytical GLSL shader tools"
    );
    let expected_non_shader = if cfg!(feature = "applied") { 59 } else { 50 };
    assert_eq!(
        non_shader_tool_count, expected_non_shader,
        "Unexpected non-shader tool count"
    );
}
