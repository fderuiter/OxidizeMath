//! Audit and end-to-end deferred dispatch tests for `ExportFormat::ComputeWgsl`.

use super::super::*;
use eframe::egui;

pub(super) fn verify_compute_shader_capability(
    meta_id: &str,
    tool: &dyn InteractiveTool,
    caps: ExportCapabilities,
) {
    if !caps.compute_shader {
        assert!(
            !caps.compute_shader_disabled_reason.is_empty(),
            "{meta_id} disables compute_shader export without an explanatory reason"
        );
        return;
    }
    assert!(
        caps.compute_shader_disabled_reason.is_empty(),
        "{meta_id} advertises compute_shader export but has non-empty disabled reason"
    );
    let wgsl = tool
        .export_compute_shader_wgsl()
        .unwrap_or_else(|e| panic!("Compute WGSL failed for {meta_id}: {e}"));
    assert!(
        wgsl.contains("@compute") && wgsl.contains("@workgroup_size("),
        "Invalid Compute WGSL entrypoint for {meta_id}"
    );
}

#[test]
fn test_volumetric_studio_deferred_compute_wgsl_file_and_clipboard_exports() {
    let tool =
        Box::new(crate::tabs::quantum::volumetric_studio::VolumetricFieldStudioTool::default());
    assert!(tool.export_capabilities().compute_shader);

    let mut fw = SimulationFramework::new("__audit_compute_wgsl__");
    fw.active_tool = Some(tool);
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };

    let _ = ctx.run(raw_input.clone(), |ctx| {
        fw.queue_export(ExportFormat::ComputeWgsl);
        fw.show(ctx, "compute_wgsl_file_audit");
    });
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert_eq!(
        fw.export_queue.last_exported_filename.as_deref(),
        Some("volumetric_field_studio_compute.wgsl")
    );
    let file_wgsl = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("compute wgsl content recorded");
    assert!(file_wgsl.contains("@compute") && file_wgsl.contains("@workgroup_size(4, 4, 4)"));

    let _ = ctx.run(raw_input, |ctx| {
        fw.queue_shader_clipboard_export(ExportFormat::ComputeWgsl);
        fw.show(ctx, "compute_wgsl_clip_audit");
    });
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let clip_wgsl = fw
        .export_queue
        .last_clipboard_content
        .as_deref()
        .expect("clipboard compute wgsl recorded");
    assert!(clip_wgsl.contains("fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>)"));
}
