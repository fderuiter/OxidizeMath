#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::presets::{CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON, STARTER_PRESET_JSON};
use super::*;

#[test]
fn surface_viewer_viewport_and_plot_export_work() {
    let mut tool = SurfaceViewer::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| tool.show(ctx));
    assert!(!egui_plot::get_last_plot_datasets().is_empty());
    assert!(!egui_plot::get_last_plot_primitives().is_empty());
}

#[test]
fn surface_viewer_custom_heightfield_and_obj_export_work() {
    let mut tool = SurfaceViewer::default();
    tool.import_preset_json(CUSTOM_RIPPLE_MANIFOLD_PRESET_JSON)
        .expect("custom ripple preset should load");
    let obj = tool.build_obj_string();
    assert!(obj.contains("v "));
    assert!(obj.contains("f "));
    assert!(!obj.contains("NaN"));
    assert!(tool.import_preset_json(STARTER_PRESET_JSON).is_ok());
    assert!(
        (tool.wall_thickness - super::super::export_utils::DEFAULT_WALL_THICKNESS).abs() < 1e-9
    );
}

#[test]
fn surface_viewer_custom_parametric_union_parameter_cap_preserves_valid_mesh() {
    let mut tool = SurfaceViewer {
        surface_type: SurfaceType::CustomParametric,
        ..Default::default()
    };
    let baseline_pos = tool.custom_sandbox.build_parametric().position(0.5, 0.5);
    tool.custom_sandbox
        .param_x_bar
        .set_expression("p1 * cos(u) + p2");
    tool.custom_sandbox
        .param_y_bar
        .set_expression("p3 * sin(u) + p4");
    tool.custom_sandbox
        .param_z_bar
        .set_expression("p5 * sin(v) + p6 + p7");
    tool.custom_sandbox.recompile_parametric_triple();
    assert!(tool
        .custom_sandbox
        .parametric_error
        .as_deref()
        .is_some_and(|m| m.contains("maximum 6")));
    let preserved_pos = tool.custom_sandbox.build_parametric().position(0.5, 0.5);
    assert!((preserved_pos - baseline_pos).norm() < 1e-12);
}

#[test]
fn surface_viewer_watertight_3d_exports_for_all_surface_types() {
    let mut tool = SurfaceViewer {
        u_resolution: 20,
        v_resolution: 18,
        wall_thickness: 0.20,
        ..Default::default()
    };
    for (stype, expected_chi) in [
        (SurfaceType::Sphere, 2),
        (SurfaceType::Torus, 0),
        (SurfaceType::KleinBottle, 0),
        (SurfaceType::CustomHeightfield, 2),
        (SurfaceType::CustomParametric, 0),
    ] {
        tool.surface_type = stype;
        let mesh = tool
            .export_mesh_3d()
            .expect("3D mesh hook present")
            .expect("3D mesh generation succeeds");
        assert!(mesh.normals.iter().all(|n| n.is_finite()));
        assert_eq!(
            super::super::export_utils::count_open_and_nonmanifold_edges(&mesh),
            (0, 0)
        );
        assert_eq!(
            super::super::export_utils::euler_characteristic(&mesh),
            expected_chi
        );
        assert!(super::super::export_utils::signed_volume(&mesh) > 0.0);
    }
}

#[test]
fn surface_viewer_clamps_high_res_state_and_reuses_cached_3d_mesh_on_orbit() {
    let mut tool = SurfaceViewer::default();
    let high_res = r#"{"surface_type":"Torus","sphere_radius":1.0,"torus_major_radius":2.0,"torus_minor_radius":0.5,"klein_radius":2.0,"u_resolution":1000,"v_resolution":1000,"pitch":0.5,"yaw":0.5,"zoom":1.0}"#;
    tool.try_load_state(high_res)
        .expect("1000x1000 state should clamp cleanly");
    assert_eq!(
        tool.u_resolution,
        crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION
    );
    assert_eq!(
        tool.v_resolution,
        crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION
    );

    let mut cached_tool = SurfaceViewer::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| cached_tool.show(c));
    let _ = ctx.run(egui::RawInput::default(), |c| cached_tool.show(c));
    assert_eq!(cached_tool.cache.surface_eval_count, 1);
    assert_eq!(cached_tool.cache.projection_count, 1);

    cached_tool.camera.yaw += 0.35;
    cached_tool.camera.zoom *= 1.15;
    let _ = ctx.run(egui::RawInput::default(), |c| cached_tool.show(c));
    assert_eq!(cached_tool.cache.surface_eval_count, 1);
    assert_eq!(cached_tool.cache.projection_count, 2);
}
