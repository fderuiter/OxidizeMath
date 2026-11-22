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

    restored.surface_type = SurfaceType::CustomParametric;
    restored
        .custom_sandbox
        .set_uv_bounds(SurfaceType::CustomParametric, [0.0, 3.0], [0.5, 2.5])
        .expect("valid parametric bounds");
    let custom_saved = restored.save_state().expect("save custom bounds");
    let mut custom_restored = CurvatureHeatmap::default();
    custom_restored
        .try_load_state(&custom_saved)
        .expect("restore custom bounds");
    assert_eq!(
        custom_restored
            .custom_sandbox
            .uv_domain_for(SurfaceType::CustomParametric),
        ((0.0, 3.0), (0.5, 2.5))
    );
    assert!(!custom_restored.build_colored_lines().is_empty());
}
