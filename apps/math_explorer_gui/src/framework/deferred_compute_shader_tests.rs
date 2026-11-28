//! Audit and end-to-end deferred dispatch tests for [`ExportFormat::WgslCompute`].

use super::super::*;
use eframe::egui;
use math_commons::ast::MAX_GLSL_OUTPUT_BYTES;

pub(super) fn verify_compute_shader_capability(
    meta_id: &str,
    tool: &dyn InteractiveTool,
    caps: ExportCapabilities,
) -> bool {
    if !caps.compute_shader {
        assert!(
            !caps.compute_shader_disabled_reason.is_empty(),
            "{meta_id} disables compute_shader export without an explanatory reason"
        );
        return false;
    }
    assert!(
        caps.compute_shader_disabled_reason.is_empty(),
        "{meta_id} advertises compute_shader export but has non-empty disabled reason"
    );
    let wgsl = tool
        .export_compute_shader_wgsl()
        .unwrap_or_else(|e| panic!("Compute WGSL failed for {meta_id}: {e}"));
    assert!(
        wgsl.contains("@compute")
            && wgsl.contains("@workgroup_size(")
            && wgsl.contains("fn cs_main(")
            && wgsl.len() <= MAX_GLSL_OUTPUT_BYTES,
        "Invalid or oversized Compute WGSL entrypoint for {meta_id}"
    );
    true
}

pub(super) fn assert_expected_compute_shader_count(actual: usize) {
    let expected = usize::from(cfg!(feature = "biology"))
        + usize::from(cfg!(feature = "physics")) * 2
        + usize::from(cfg!(feature = "pure_math")) * 2;
    assert_eq!(
        actual, expected,
        "Expected {expected} compute_shader-capable tools"
    );
}

#[test]
fn test_deferred_wgsl_compute_file_and_clipboard_exports_across_five_domain_tools() {
    let compute_ids = [
        (
            "VolumetricFieldStudioTool",
            "volumetric_field_studio_compute.wgsl",
        ),
        ("MorphogenesisTool", "turing_morphogenesis_compute.wgsl"),
        ("WaveSimulator", "wave_simulator_compute.wgsl"),
        ("OdeSolverTool", "ode_solvers_compute.wgsl"),
        (
            "MonteCarloWorkbenchTool",
            "monte_carlo___bootstrap_workbench_compute.wgsl",
        ),
    ];
    let ctx = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1024.0, 768.0),
        )),
        ..Default::default()
    };

    for (tool_id, expected_filename) in compute_ids {
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|m| m.id == tool_id)
            .unwrap_or_else(|| panic!("Missing registered compute_shader tool {tool_id}"));
        let tool = (meta.build)();
        assert!(
            tool.export_capabilities().compute_shader,
            "{tool_id} must advertise compute_shader"
        );

        let mut fw = SimulationFramework::new("__audit_compute_wgsl__");
        fw.active_tool = Some(tool);

        let _ = ctx.run(raw_input.clone(), |ctx| {
            fw.queue_export(ExportFormat::WgslCompute);
            fw.show(ctx, "compute_wgsl_file_audit");
        });
        assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
        assert_eq!(
            fw.export_queue.last_exported_filename.as_deref(),
            Some(expected_filename)
        );
        let file_wgsl = fw
            .export_queue
            .last_exported_content
            .as_deref()
            .expect("compute wgsl content recorded");
        assert!(file_wgsl.contains("@compute") && file_wgsl.contains("@workgroup_size("));

        let _ = ctx.run(raw_input.clone(), |ctx| {
            fw.queue_shader_clipboard_export(ExportFormat::WgslCompute);
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
}
