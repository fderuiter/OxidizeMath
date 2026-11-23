#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::starter_gallery::StarterPresetMetadata;
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

fn hydrate_tool_for_audit(meta: &crate::framework::ToolMetadata, tool: &mut dyn InteractiveTool) {
    let initial = tool.export_capabilities();
    if !initial.dataset {
        assert!(
            !initial.dataset_disabled_reason.is_empty(),
            "{} is initially unpopulated without dataset_disabled_reason",
            meta.id
        );
        assert!(
            !initial.figure_disabled_reason.is_empty(),
            "{} is initially unpopulated without figure_disabled_reason",
            meta.id
        );
    }

    if !tool.export_capabilities().dataset {
        tool.step_simulation();
    }

    if !tool.export_capabilities().dataset {
        if let Some(preset) = inventory::iter::<StarterPresetMetadata>
            .into_iter()
            .find(|p| {
                p.tool_name == Some(meta.id)
                    || p.tool_name == Some(meta.display_name)
                    || p.tool_name == Some(tool.name())
            })
        {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(preset.generate_preset()) {
                if let Some(cfg) = val.get("config") {
                    tool.try_load_state(&cfg.to_string())
                        .unwrap_or_else(|e| panic!("Preset hydrate failed for {}: {e}", meta.id));
                }
            }
        }
    }
}

fn verify_dataset_exports(meta_name: &str, framework: &mut SimulationFramework) {
    run_modal_export_after_clear(framework, ExportFormat::Csv);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred modal CSV failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let csv = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing CSV content for {meta_name}"));
    assert!(csv.lines().count() >= 2, "CSV empty for {meta_name}");

    verify_clipboard_csv_export(meta_name, framework);

    run_modal_export_after_clear(framework, ExportFormat::Json);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred modal JSON failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let json = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing JSON content for {meta_name}"));
    let parsed: serde_json::Value =
        serde_json::from_str(json).unwrap_or_else(|e| panic!("Invalid JSON for {meta_name}: {e}"));
    assert!(parsed.is_array() || parsed.is_object());
}

fn verify_clipboard_csv_export(meta_name: &str, framework: &mut SimulationFramework) {
    let ctx = egui::Context::default();
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1024.0, 768.0),
            )),
            ..Default::default()
        },
        |ctx| {
            egui_plot::clear_last_plot_datasets();
            framework.queue_clipboard_export();
            framework.show(ctx, "audit_clipboard_framework");
        },
    );
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Clipboard CSV failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let clip = framework
        .export_queue
        .last_clipboard_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing clipboard CSV for {meta_name}"));
    assert!(
        clip.lines().count() >= 2,
        "Clipboard CSV empty for {meta_name}"
    );
}

fn verify_figure_exports(meta_name: &str, framework: &mut SimulationFramework) {
    run_modal_export_after_clear(framework, ExportFormat::Png);
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

    run_modal_export_after_clear(framework, ExportFormat::Svg);
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
        svg.contains("<svg") && svg.len() > 200,
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

fn audit_single_tool(
    meta: &crate::framework::ToolMetadata,
    plot_tool_count: &mut usize,
    dataset_only_count: &mut usize,
    shader_tool_count: &mut usize,
    non_shader_tool_count: &mut usize,
) {
    let mut tool = (meta.build)();
    hydrate_tool_for_audit(meta, tool.as_mut());

    let caps = tool.export_capabilities();
    if verify_shader_capability(meta.id, tool.as_ref(), caps) {
        *shader_tool_count += 1;
    } else {
        *non_shader_tool_count += 1;
    }

    assert!(caps.dataset, "{} dataset export disabled", meta.id);
    assert!(caps.dataset_disabled_reason.is_empty(), "{}", meta.id);

    let mut framework = SimulationFramework::new("__audit__");
    framework.export_options.width = 200;
    framework.export_options.height = 150;
    framework.export_options.scale = 1.0;
    framework.active_tool = Some(tool);

    verify_dataset_exports(meta.id, &mut framework);

    if caps.figure {
        *plot_tool_count += 1;
        assert!(caps.figure_disabled_reason.is_empty(), "{}", meta.id);
        verify_figure_exports(meta.id, &mut framework);
    } else {
        *dataset_only_count += 1;
        assert!(!caps.figure_disabled_reason.is_empty(), "{}", meta.id);
        run_modal_export_after_clear(&mut framework, ExportFormat::Png);
        assert_eq!(
            framework.export_status_severity,
            ExportStatusSeverity::Warning,
            "Expected Warning when queuing PNG on dataset_only tool {}",
            meta.id
        );
    }
}

#[test]
fn test_all_registered_tools_capabilities_and_deferred_modal_exports() {
    let mut plot_tool_count = 0usize;
    let mut dataset_only_count = 0usize;
    let mut shader_tool_count = 0usize;
    let mut non_shader_tool_count = 0usize;

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        audit_single_tool(
            meta,
            &mut plot_tool_count,
            &mut dataset_only_count,
            &mut shader_tool_count,
            &mut non_shader_tool_count,
        );
    }

    assert_eq!(
        plot_tool_count + dataset_only_count,
        shader_tool_count + non_shader_tool_count
    );

    #[cfg(all(
        feature = "ai",
        feature = "applied",
        feature = "biology",
        feature = "climate",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    {
        assert_eq!(plot_tool_count, 53, "Expected 53 dataset+figure tools");
        assert_eq!(dataset_only_count, 7, "Expected 7 dataset-only tools");
        assert_eq!(shader_tool_count, 3, "Expected 3 analytical GLSL tools");
        assert_eq!(non_shader_tool_count, 57, "Expected 57 non-shader tools");
    }
}
