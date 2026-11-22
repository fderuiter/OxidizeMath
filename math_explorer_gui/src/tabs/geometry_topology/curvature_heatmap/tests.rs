#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::CUSTOM_SADDLE_CURVATURE_PRESET_JSON;
use super::*;

#[test]
fn curvature_heatmap_camera_preset_and_export_work() {
    let tool = CurvatureHeatmap {
        camera: Camera3D::new(0.7, -0.35, 1.4),
        ..Default::default()
    };
    let saved = tool.save_state().expect("state should serialize");

    let mut restored = CurvatureHeatmap::default();
    restored
        .try_load_state(&saved)
        .expect("saved state should load");
    assert!((restored.camera.pitch - 0.7).abs() < 1e-6);
    assert!((restored.camera.yaw + 0.35).abs() < 1e-6);
    assert!((restored.camera.zoom - 1.4).abs() < 1e-6);

    let legacy = r#"{"surface_type":"Torus","curvature_type":"Mean","sphere_radius":1.0,"torus_major_radius":2.0,"torus_minor_radius":0.5,"klein_radius":2.0,"u_resolution":20,"v_resolution":20,"pitch":0.4,"yaw":0.8,"zoom":1.2}"#;
    restored
        .try_load_state(legacy)
        .expect("legacy pitch/yaw/zoom preset should load");
    assert!((restored.camera.yaw - 0.8).abs() < 1e-6);

    restored
        .import_preset_json(CUSTOM_SADDLE_CURVATURE_PRESET_JSON)
        .expect("custom saddle curvature preset should load");
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| restored.show(ctx));
    assert!(!egui_plot::get_last_plot_datasets().is_empty());
}

#[test]
fn curvature_heatmap_builds_watertight_colored_ply_for_all_surfaces() {
    let mut tool = CurvatureHeatmap {
        u_resolution: 18,
        v_resolution: 18,
        ..Default::default()
    };
    for stype in [
        SurfaceType::Sphere,
        SurfaceType::Torus,
        SurfaceType::KleinBottle,
        SurfaceType::CustomHeightfield,
        SurfaceType::CustomParametric,
    ] {
        tool.surface_type = stype;
        let colored = tool
            .export_colored_mesh_3d()
            .expect("colored mesh hook present")
            .expect("colored curvature mesh builds");
        assert_eq!(colored.colors.len(), colored.mesh.vertices.len());
        assert!(colored.mesh.normals.iter().all(|n| n.is_finite()));
        assert_eq!(
            super::super::export_utils::count_open_and_nonmanifold_edges(&colored.mesh),
            (0, 0)
        );
        assert!(super::super::export_utils::signed_volume(&colored.mesh) > 0.0);
        let ply = oxidize_core::mesh::export_colored_mesh_to_ply_binary(&colored)
            .expect("colored PLY serializes");
        assert!(ply.starts_with(b"ply\nformat binary_little_endian 1.0\n"));
    }
}
