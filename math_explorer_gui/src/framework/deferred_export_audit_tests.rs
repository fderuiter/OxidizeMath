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

fn verify_audio_capability(meta_id: &str, caps: ExportCapabilities) -> bool {
    if !caps.audio {
        assert!(
            !caps.audio_disabled_reason.is_empty(),
            "{meta_id} disables audio export without an explanatory reason"
        );
        return false;
    }
    assert!(
        caps.audio_disabled_reason.is_empty(),
        "{meta_id} advertises audio export but has non-empty disabled reason"
    );
    true
}

fn verify_dataset_only_tool(meta_name: &str, tool: Box<dyn InteractiveTool>) {
    let mut framework = SimulationFramework::new("__audit_dataset_only__");
    framework.active_tool = Some(tool);

    run_deferred_export_frame(&mut framework, ExportFormat::Csv, false);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred CSV failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let csv = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing CSV content for {meta_name}"));
    assert!(
        csv.starts_with("Dataset,X,Y\n") && csv.lines().count() >= 2,
        "Invalid CSV content for {meta_name}: {csv}"
    );

    run_deferred_export_frame(&mut framework, ExportFormat::Json, false);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Success,
        "Deferred JSON failed for {meta_name}: {:?}",
        framework.export_status_message
    );
    let json = framework
        .export_queue
        .last_exported_content
        .as_deref()
        .unwrap_or_else(|| panic!("Missing JSON content for {meta_name}"));
    assert!(
        json.contains("\"dataset\":") && !json.contains("NaN"),
        "Invalid JSON content for {meta_name}: {json}"
    );

    run_deferred_export_frame(&mut framework, ExportFormat::Png, false);
    assert_eq!(
        framework.export_status_severity,
        ExportStatusSeverity::Warning,
        "Expected Warning on PNG export for dataset_only tool {meta_name}"
    );
}

fn verify_non_figure_tool(
    meta_id: &str,
    tool: Box<dyn InteractiveTool>,
    caps: ExportCapabilities,
    dataset_only_tool_count: &mut usize,
    unsupported_tool_count: &mut usize,
) {
    assert!(
        !caps.figure_disabled_reason.is_empty(),
        "{meta_id} disables figure export without an explanatory reason"
    );
    if caps.dataset {
        *dataset_only_tool_count += 1;
        assert!(
            caps.dataset_disabled_reason.is_empty(),
            "{meta_id} advertises dataset export but has non-empty disabled reason"
        );
        verify_dataset_only_tool(meta_id, tool);
    } else {
        *unsupported_tool_count += 1;
        assert!(
            !caps.dataset_disabled_reason.is_empty(),
            "{meta_id} disables dataset export without an explanatory reason"
        );
    }
}

#[test]
fn test_all_registered_tools_capabilities_and_deferred_modal_exports() {
    let (mut plot_tool_count, mut raster_tool_count) = (0usize, 0usize);
    let (mut dataset_only_tool_count, mut unsupported_tool_count) = (0usize, 0usize);
    let (mut solid_3d_count, mut splat_3d_count) = (0usize, 0usize);
    let (mut shader_tool_count, mut non_shader_tool_count) = (0usize, 0usize);
    let mut audio_tool_count = 0usize;

    for meta in inventory::iter::<crate::framework::ToolMetadata> {
        let tool = (meta.build)();
        let caps = tool.export_capabilities();
        if verify_audio_capability(meta.id, caps) {
            audio_tool_count += 1;
        }
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
            verify_non_figure_tool(
                meta.id,
                tool,
                caps,
                &mut dataset_only_tool_count,
                &mut unsupported_tool_count,
            );
            continue;
        }
        if caps.dataset {
            plot_tool_count += 1;
        } else {
            raster_tool_count += 1;
        }
        verify_single_figure_tool(meta.id, tool);
    }

    assert_expected_capability_counts([
        plot_tool_count,
        raster_tool_count,
        dataset_only_tool_count,
        unsupported_tool_count,
        solid_3d_count,
        splat_3d_count,
        shader_tool_count,
        non_shader_tool_count,
        audio_tool_count,
    ]);
}

fn assert_expected_capability_counts(counts: [usize; 9]) {
    let [plot, raster, dataset_only, unsupported, solid_3d, splat_3d, shader, non_shader, audio] =
        counts;
    let has_surface_morpho = cfg!(all(feature = "pure_math", feature = "biology"));
    let bio_extra = if cfg!(feature = "biology") { 4 } else { 0 };
    let expected_audio = usize::from(cfg!(feature = "biology")) * 2
        + usize::from(cfg!(feature = "physics")) * 3
        + usize::from(cfg!(feature = "pure_math"));
    assert_eq!(
        audio, expected_audio,
        "Expected {expected_audio} WAV audio-capable tools"
    );
    let expected_plot = (if has_surface_morpho { 75 } else { 74 }) + bio_extra;
    let expected_solid = if has_surface_morpho { 8 } else { 7 };
    let expected_non_shader = (if has_surface_morpho { 89 } else { 88 }) + bio_extra;
    assert_eq!(
        plot, expected_plot,
        "Expected {expected_plot} Plot dataset+figure tools"
    );
    assert_eq!(raster, 8, "Expected 8 raster figure tools");
    assert_eq!(dataset_only, 0, "Expected 0 dataset-only registered tools");
    assert_eq!(unsupported, 9, "Expected 9 non-figure tools");
    assert_eq!(
        solid_3d, expected_solid,
        "Expected {expected_solid} solid 3D geometric tools"
    );
    assert_eq!(splat_3d, 1, "Expected 1 3DGS splat PLY tool");
    assert_eq!(shader, 3, "Expected 3 analytical GLSL shader tools");
    assert_eq!(
        non_shader, expected_non_shader,
        "Expected {expected_non_shader} non-shader tools"
    );
}

#[derive(Default)]
struct DatasetOnlyFixtureTool;

impl scientific_metadata::theory::TheoryDescribable for DatasetOnlyFixtureTool {
    fn theory_description(&self) -> String {
        "Dataset-only fixture".into()
    }
    fn phonetic_description(&self) -> String {
        "Dataset-only fixture".into()
    }
    fn theory_citation(&self) -> String {
        "Fixture".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

impl InteractiveTool for DatasetOnlyFixtureTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "DatasetOnlyFixture"
    }
    fn domain(&self) -> &'static str {
        "financial_math"
    }
    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::dataset_only()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![("FixtureSeries".into(), vec![[0.0, 1.0], [1.0, 2.0]])]
    }
    fn show_inside(&mut self, _ui: &mut egui::Ui) {}
}

#[test]
fn test_dataset_only_capability_profile_exports_csv_json_and_warns_on_png() {
    let tool: Box<dyn InteractiveTool> = Box::new(DatasetOnlyFixtureTool);
    let caps = tool.export_capabilities();
    let (mut dataset_only_count, mut unsupported_count) = (0usize, 0usize);
    verify_non_figure_tool(
        "DatasetOnlyFixtureTool",
        tool,
        caps,
        &mut dataset_only_count,
        &mut unsupported_count,
    );
    assert_eq!(dataset_only_count, 1);
    assert_eq!(unsupported_count, 0);
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
fn test_all_eight_3d_tools_deferred_stl_ply_payloads_and_live_region() {
    let solid_ids = [
        "SurfaceViewer",
        "CurvatureHeatmap",
        #[cfg(all(feature = "pure_math", feature = "biology"))]
        "SurfaceMorphogenesisTool",
        "AttractorPlotter",
        "CrystalViewer",
        "SpinVisualizer",
        "CorticalMorphometryTool",
        "VolumetricFieldStudioTool",
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
    verify_splat_mode_ply_and_assertive_stl_obj_warnings((gs_meta.build)());

    let mut vol_splat =
        crate::tabs::quantum::volumetric_studio::VolumetricFieldStudioTool::default();
    vol_splat.config.render_mode =
        crate::tabs::quantum::volumetric_studio::state::VolumetricRenderMode::GaussianRadianceCloud;
    vol_splat.refresh_zero_alloc();
    verify_splat_mode_ply_and_assertive_stl_obj_warnings(Box::new(vol_splat));
}

fn verify_splat_mode_ply_and_assertive_stl_obj_warnings(tool: Box<dyn InteractiveTool>) {
    let mut gs_fw = SimulationFramework::new("__audit_3dgs__");
    gs_fw.active_tool = Some(tool);

    for blocked_fmt in [ExportFormat::Stl, ExportFormat::Obj] {
        clear_test_announcements();
        run_deferred_export_frame(&mut gs_fw, blocked_fmt, false);
        assert_eq!(gs_fw.export_status_severity, ExportStatusSeverity::Warning);
        assert!(all_announcements()
            .iter()
            .any(|(_, prio)| prio == "assertive"));
    }

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
