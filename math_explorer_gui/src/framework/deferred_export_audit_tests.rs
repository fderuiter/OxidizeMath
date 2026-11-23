#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};
use crate::tabs::geometry_topology::export_utils::{
    count_open_and_nonmanifold_edges, signed_volume,
};
use eframe::egui;
use std::collections::HashSet;

fn run_deferred_export_frame(
    framework: &mut SimulationFramework,
    format: ExportFormat,
    open_modal: bool,
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
        framework.show_export_modal = open_modal;
        framework.queue_export(format);
        framework.show(ctx, "audit_framework");
    });
}

fn verify_single_figure_tool(meta_name: &str, tool: Box<dyn InteractiveTool>) {
    let mut framework = SimulationFramework::new("__audit__");
    framework.export_options.width = 320;
    framework.export_options.height = 240;
    framework.active_tool = Some(tool);

    run_deferred_export_frame(&mut framework, ExportFormat::Png, true);
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

    run_deferred_export_frame(&mut framework, ExportFormat::Svg, true);
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
    let mut solid_3d_count = 0usize;
    let mut splat_3d_count = 0usize;
    let mut shader_tool_count = 0usize;
    let mut non_shader_tool_count = 0usize;

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        let tool = (meta.build)();
        let caps = tool.export_capabilities();
        if caps.stl && caps.ply && caps.obj {
            solid_3d_count += 1;
        } else if caps.ply {
            splat_3d_count += 1;
        }
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

    assert_eq!(plot_tool_count, 40, "Expected 40 Plot dataset+figure tools");
    assert_eq!(raster_tool_count, 7, "Expected 7 raster figure tools");
    assert_eq!(unsupported_tool_count, 16, "Expected 16 non-figure tools");
    assert_eq!(solid_3d_count, 6, "Expected 6 solid 3D geometric tools");
    assert_eq!(splat_3d_count, 1, "Expected 1 3DGS splat PLY tool");
    assert_eq!(
        shader_tool_count, 3,
        "Expected 3 analytical GLSL shader tools"
    );
    assert_eq!(non_shader_tool_count, 60, "Expected 60 non-shader tools");
}

fn verify_solid_3d_tool_exports(meta_id: &str, tool: Box<dyn InteractiveTool>) {
    let mesh = tool
        .export_mesh_3d()
        .unwrap_or_else(|| panic!("{meta_id} must implement export_mesh_3d"))
        .unwrap_or_else(|e| panic!("{meta_id} mesh generation failed: {e}"));
    assert!(
        mesh.normals.iter().all(|n| n.is_finite()),
        "{meta_id} emitted non-finite vertex normal"
    );
    assert_eq!(
        count_open_and_nonmanifold_edges(&mesh),
        (0, 0),
        "{meta_id} mesh is not a closed 2-manifold"
    );
    assert!(
        signed_volume(&mesh) > 0.0,
        "{meta_id} mesh winding is not outward"
    );
    let expected_stl_len = 84 + 50 * (mesh.indices.len() / 3);

    let mut fw = SimulationFramework::new("__audit_3d__");
    fw.active_tool = Some(tool);

    clear_test_announcements();
    let slot_before = with_live_region(|r| r.active_slot);
    run_deferred_export_frame(&mut fw, ExportFormat::Stl, false);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert_ne!(with_live_region(|r| r.active_slot), slot_before);
    assert!(all_announcements()
        .iter()
        .any(|(msg, prio)| msg.contains("STL 3D Mesh saved successfully") && prio == "polite"));
    let stl_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("STL bytes recorded");
    assert_eq!(stl_bytes.len(), expected_stl_len);

    clear_test_announcements();
    run_deferred_export_frame(&mut fw, ExportFormat::Ply, false);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    assert!(all_announcements()
        .iter()
        .any(|(msg, prio)| msg.contains("PLY 3D Model saved successfully") && prio == "polite"));
    let ply_bytes = fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("PLY bytes recorded");
    assert!(ply_bytes.starts_with(b"ply\nformat binary_little_endian 1.0\n"));

    run_deferred_export_frame(&mut fw, ExportFormat::Obj, false);
    assert_eq!(fw.export_status_severity, ExportStatusSeverity::Success);
    let obj_text = fw
        .export_queue
        .last_exported_content
        .as_deref()
        .expect("OBJ text recorded");
    assert!(obj_text.contains("\nf ") && !obj_text.contains("NaN"));
}

#[test]
fn test_all_six_3d_tools_deferred_stl_ply_payloads_and_live_region() {
    let solid_ids = [
        "SurfaceViewer",
        "CurvatureHeatmap",
        "AttractorPlotter",
        "CrystalViewer",
        "SpinVisualizer",
        "IsosurfaceTool",
    ];
    for id in solid_ids {
        let meta = inventory::iter::<crate::framework::ToolMetadata>
            .into_iter()
            .find(|m| m.id == id)
            .unwrap_or_else(|| panic!("Missing registered 3D tool {id}"));
        verify_solid_3d_tool_exports(id, (meta.build)());
    }

    let gs_meta = inventory::iter::<crate::framework::ToolMetadata>
        .into_iter()
        .find(|m| m.id == "Gaussian Splatting")
        .expect("Gaussian Splatting registered");
    let mut gs_fw = SimulationFramework::new("__audit_3dgs__");
    gs_fw.active_tool = Some((gs_meta.build)());

    clear_test_announcements();
    run_deferred_export_frame(&mut gs_fw, ExportFormat::Stl, false);
    assert_eq!(gs_fw.export_status_severity, ExportStatusSeverity::Warning);
    assert!(all_announcements()
        .iter()
        .any(|(_, prio)| prio == "assertive"));

    clear_test_announcements();
    let slot_before = with_live_region(|r| r.active_slot);
    run_deferred_export_frame(&mut gs_fw, ExportFormat::Ply, false);
    assert_eq!(gs_fw.export_status_severity, ExportStatusSeverity::Success);
    assert_ne!(with_live_region(|r| r.active_slot), slot_before);
    assert!(all_announcements()
        .iter()
        .any(|(msg, prio)| msg.contains("PLY 3D Model saved successfully") && prio == "polite"));
    let splat_ply = gs_fw
        .export_queue
        .last_exported_bytes
        .as_deref()
        .expect("3DGS PLY bytes recorded");
    assert!(splat_ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
}
