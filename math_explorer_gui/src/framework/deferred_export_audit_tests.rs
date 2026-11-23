#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
#[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
use crate::accessibility::{all_announcements, clear_test_announcements, with_live_region};
use crate::starter_gallery::StarterPresetMetadata;
#[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
use crate::tabs::geometry_topology::export_utils::{
    count_open_and_nonmanifold_edges, signed_volume,
};
use eframe::egui;
use std::collections::HashSet;

#[derive(Default)]
struct AuditCounts {
    plot_tools: usize,
    dataset_only: usize,
    solid_3d: usize,
    splat_3d: usize,
    shader_tools: usize,
    non_shader_tools: usize,
}

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
    run_deferred_export_frame(framework, ExportFormat::Csv, true);
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

    run_deferred_export_frame(framework, ExportFormat::Json, true);
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
    run_deferred_export_frame(framework, ExportFormat::Png, true);
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

    run_deferred_export_frame(framework, ExportFormat::Svg, true);
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

fn audit_single_tool(meta: &crate::framework::ToolMetadata, counts: &mut AuditCounts) {
    let mut tool = (meta.build)();
    hydrate_tool_for_audit(meta, tool.as_mut());

    let caps = tool.export_capabilities();
    if caps.stl && caps.ply && caps.obj {
        counts.solid_3d += 1;
    } else if caps.ply {
        counts.splat_3d += 1;
    }
    if verify_shader_capability(meta.id, tool.as_ref(), caps) {
        counts.shader_tools += 1;
    } else {
        counts.non_shader_tools += 1;
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
        counts.plot_tools += 1;
        assert!(caps.figure_disabled_reason.is_empty(), "{}", meta.id);
        verify_figure_exports(meta.id, &mut framework);
    } else {
        counts.dataset_only += 1;
        assert!(!caps.figure_disabled_reason.is_empty(), "{}", meta.id);
        run_deferred_export_frame(&mut framework, ExportFormat::Png, true);
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
    let mut counts = AuditCounts::default();

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        audit_single_tool(meta, &mut counts);
    }

    assert_eq!(
        counts.plot_tools + counts.dataset_only,
        counts.shader_tools + counts.non_shader_tools
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
        assert_eq!(counts.plot_tools, 57, "Expected 57 dataset+figure tools");
        assert_eq!(counts.dataset_only, 7, "Expected 7 dataset-only tools");
        assert_eq!(counts.solid_3d, 5, "Expected 5 solid 3D geometric tools");
        assert_eq!(counts.splat_3d, 1, "Expected 1 3DGS splat PLY tool");
        assert_eq!(counts.shader_tools, 3, "Expected 3 analytical GLSL tools");
        assert_eq!(counts.non_shader_tools, 61, "Expected 61 non-shader tools");
    }
}

#[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
fn assert_closed_outward_mesh(meta_id: &str, mesh: &oxidize_core::mesh::Mesh) -> usize {
    assert!(
        mesh.normals.iter().all(|n| n.is_finite()),
        "{meta_id} emitted non-finite vertex normal"
    );
    assert_eq!(
        count_open_and_nonmanifold_edges(mesh),
        (0, 0),
        "{meta_id} mesh is not a closed 2-manifold"
    );
    assert!(
        signed_volume(mesh) > 0.0,
        "{meta_id} mesh winding is not outward"
    );
    84 + 50 * (mesh.indices.len() / 3)
}

#[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
fn verify_solid_3d_tool_exports(meta_id: &str, tool: Box<dyn InteractiveTool>) {
    let mesh = tool
        .export_mesh_3d()
        .unwrap_or_else(|| panic!("{meta_id} must implement export_mesh_3d"))
        .unwrap_or_else(|e| panic!("{meta_id} mesh generation failed: {e}"));
    let expected_stl_len = assert_closed_outward_mesh(meta_id, &mesh);

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

#[cfg(all(feature = "ai", feature = "physics", feature = "pure_math"))]
#[test]
fn test_all_six_3d_tools_deferred_stl_ply_payloads_and_live_region() {
    let solid_ids = [
        "SurfaceViewer",
        "CurvatureHeatmap",
        "AttractorPlotter",
        "CrystalViewer",
        "SpinVisualizer",
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
