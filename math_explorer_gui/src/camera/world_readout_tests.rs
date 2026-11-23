#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::{ExportFormat, InteractiveTool, SimulationFramework};
use crate::tabs::{
    ai::gaussian_splatting::GaussianSplattingTool,
    chaos::attractors::AttractorPlotter,
    geometry_topology::{curvature_heatmap::CurvatureHeatmap, surface_viewer::SurfaceViewer},
    quantum::spin_viz::SpinVisualizer,
    solid_state::crystal_viewer::CrystalViewer,
};
use nalgebra::Point3;

fn run_tool_and_capture_3d(tool: &mut dyn InteractiveTool) -> Vec<(String, Vec<[f64; 3]>)> {
    let ctx = egui::Context::default();
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(960.0, 720.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(input, |sub_ctx| {
        egui_plot::clear_last_plot_datasets();
        tool.show(sub_ctx);
    });
    egui_plot::get_last_plot_datasets_3d()
}

fn assert_datasets_3d_invariant(
    label: &str,
    before: &[(String, Vec<[f64; 3]>)],
    after: &[(String, Vec<[f64; 3]>)],
) {
    assert!(
        !before.is_empty(),
        "{label} should emit non-empty 3D datasets"
    );
    assert_eq!(before.len(), after.len(), "{label} series count changed");
    for ((name_a, pts_a), (name_b, pts_b)) in before.iter().zip(after.iter()) {
        assert_eq!(name_a, name_b, "{label} series name changed");
        assert_eq!(pts_a.len(), pts_b.len(), "{label} vertex count changed");
        for (pa, pb) in pts_a.iter().zip(pts_b.iter()) {
            for k in 0..3 {
                assert!(
                    (pa[k] - pb[k]).abs() < 1e-12,
                    "{label} world coordinate axis {k} drifted under camera pose: {pa:?} vs {pb:?}"
                );
            }
        }
    }
}

fn transformed_camera(base: Camera3D) -> Camera3D {
    let mut cam = base;
    cam.pitch = -0.65;
    cam.yaw = 1.35;
    cam.zoom = 1.85;
    cam.pan_x = 0.75;
    cam.pan_y = -0.55;
    cam.projection = ProjectionMode::Perspective;
    cam
}

#[test]
fn test_all_six_3d_tools_emit_camera_invariant_world_xyz() {
    let mut splat = GaussianSplattingTool::default();
    let splat_a = run_tool_and_capture_3d(&mut splat);
    splat.camera = transformed_camera(splat.camera);
    let splat_b = run_tool_and_capture_3d(&mut splat);
    assert_datasets_3d_invariant("GaussianSplatting", &splat_a, &splat_b);

    let mut attractor = AttractorPlotter::default();
    for _ in 0..10 {
        attractor.step_simulation();
    }
    attractor.execution_state = crate::tabs::chaos::attractors::SolverExecutionState::Paused;
    let attr_a = run_tool_and_capture_3d(&mut attractor);
    attractor.camera = transformed_camera(attractor.camera);
    let attr_b = run_tool_and_capture_3d(&mut attractor);
    assert_datasets_3d_invariant("AttractorPlotter", &attr_a, &attr_b);

    let mut surface = SurfaceViewer::default();
    let surf_a = run_tool_and_capture_3d(&mut surface);
    surface.camera = transformed_camera(surface.camera);
    let surf_b = run_tool_and_capture_3d(&mut surface);
    assert_datasets_3d_invariant("SurfaceViewer", &surf_a, &surf_b);

    let mut curvature = CurvatureHeatmap::default();
    let curv_a = run_tool_and_capture_3d(&mut curvature);
    curvature.camera = transformed_camera(curvature.camera);
    let curv_b = run_tool_and_capture_3d(&mut curvature);
    assert_datasets_3d_invariant("CurvatureHeatmap", &curv_a, &curv_b);

    let mut spin = SpinVisualizer::default();
    spin.step_once();
    let spin_a = run_tool_and_capture_3d(&mut spin);
    spin.camera = transformed_camera(spin.camera);
    let spin_b = run_tool_and_capture_3d(&mut spin);
    assert_datasets_3d_invariant("SpinVisualizer", &spin_a, &spin_b);

    let mut crystal = CrystalViewer::default();
    let crys_a = run_tool_and_capture_3d(&mut crystal);
    crystal.camera = transformed_camera(crystal.camera);
    let crys_b = run_tool_and_capture_3d(&mut crystal);
    assert_datasets_3d_invariant("CrystalViewer", &crys_a, &crys_b);
    assert_eq!(crys_a[0].1[0], [0.0, 0.0, 0.0]);
}

#[test]
fn test_gaussian_splatting_zero_percent_axis_mismatch_in_parallel_and_perspective() {
    for projection in [ProjectionMode::Parallel, ProjectionMode::Perspective] {
        let mut cam = Camera3D::new(0.5, 0.5, 1.25);
        cam.pan_x = 0.35;
        cam.pan_y = -0.25;
        cam.projection = projection;
        let mut tool = GaussianSplattingTool::default();
        tool.camera = cam;

        let mut builder = Scene3DBuilder::new(cam);
        builder.add_axes(1.4, true);
        let axis_tips = [
            Point3::new(1.4, 0.0, 0.0),
            Point3::new(0.0, 1.4, 0.0),
            Point3::new(0.0, 0.0, 1.4),
        ];
        for tip in axis_tips {
            let splat_proj = tool.project_splat_point(tip);
            let axis_proj = cam.project(&[tip.x, tip.y, tip.z]);
            let err = (splat_proj[0] - axis_proj[0]).hypot(splat_proj[1] - axis_proj[1]);
            assert!(
                err < 1e-12,
                "Expected 0.0% axis mismatch in {projection:?} for {tip:?}, got {err}"
            );
        }
    }
}

#[test]
fn test_framework_export_queue_csv_json_and_url_roundtrip_preserve_3d_state() {
    let mut crystal = CrystalViewer::default();
    crystal.camera = transformed_camera(crystal.camera);
    let preset_json = crystal
        .export_preset_json()
        .expect("3D preset envelope should serialize");
    assert!(preset_json.contains("\"Perspective\""));
    let fragment = crate::url_state::encode_share_fragment(
        "solid_state",
        Some("CrystalViewer"),
        Some(&preset_json),
    )
    .expect("3D state should fit within 4 KiB URL state codec limit");
    assert!(fragment.starts_with("#tab=solid_state&tool=CrystalViewer&state="));
    let parsed = crate::url_state::parse_share_location(&fragment, "")
        .expect("3D URL fragment should parse")
        .expect("ShareUrlParams expected");
    let mut restored = CrystalViewer::default();
    restored
        .import_preset_json(&parsed.state_json.expect("decoded state_json"))
        .expect("Decoded 3D URL state should restore");
    assert_eq!(restored.camera, crystal.camera);

    let temp_dir = std::env::temp_dir().join("oxidize_3d_sidecar_export_test");
    let _ = std::fs::remove_dir_all(&temp_dir);
    std::fs::create_dir_all(&temp_dir).expect("temp dir");

    let mut fw = SimulationFramework::new("__3d_export_test__");
    fw.active_tool = Some(Box::new(restored));
    fw.export_queue.destination_dir = Some(temp_dir.clone());
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| {
        fw.queue_export(ExportFormat::Csv);
        fw.show(sub_ctx, "fw_3d_csv");
    });
    let csv = fw
        .export_queue
        .last_exported_content
        .clone()
        .expect("3D CSV content");
    assert!(csv.starts_with("Dataset,X,Y,Z\n"), "Got: {csv}");
    assert!(csv.contains("Atoms,0,0,0\n"));

    let _ = ctx.run(egui::RawInput::default(), |sub_ctx| {
        fw.queue_export(ExportFormat::Json);
        fw.show(sub_ctx, "fw_3d_json");
    });
    let json = fw
        .export_queue
        .last_exported_content
        .clone()
        .expect("3D JSON content");
    assert!(json.contains("\"z\": 0"), "Got: {json}");
}
