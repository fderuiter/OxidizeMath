#![cfg_attr(any(), verified(opt_out = "tests"))]

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
fn surface_viewer_custom_domain_bounds_mesh_obj_and_state_roundtrip() {
    let mut tool = SurfaceViewer {
        surface_type: SurfaceType::CustomHeightfield,
        u_resolution: 10,
        v_resolution: 10,
        ..Default::default()
    };
    tool.custom_sandbox.heightfield_bar.set_expression("x + 2 * y");
    tool.custom_sandbox
        .set_uv_bounds(SurfaceType::CustomHeightfield, [1.0, 5.0], [-3.0, 4.0])
        .expect("valid domain bounds");

    let obj = tool.build_obj_string();
    assert!(obj.contains("v 5.000000 4.000000 13.000000") || obj.contains("v 5 4 13"));

    let saved = tool.save_state().expect("save state");
    assert!(saved.len() < 4096);
    let mut restored = SurfaceViewer::default();
    restored.try_load_state(&saved).expect("load state");
    assert_eq!(
        restored.custom_sandbox.uv_domain_for(SurfaceType::CustomHeightfield),
        ((1.0, 5.0), (-3.0, 4.0))
    );
    assert!(
        tool.custom_sandbox
            .set_uv_bounds(SurfaceType::CustomHeightfield, [4.0, 1.0], [-1.0, 1.0])
            .is_err()
    );
}

