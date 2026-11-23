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

fn open_parameters_header(ctx: &egui::Context, tool: &mut SurfaceViewer, screen: Option<egui::Rect>) {
    let out = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    let ak = out.platform_output.accesskit_update.unwrap();
    let b = ak
        .nodes
        .iter()
        .find(|(_, n)| n.label().unwrap_or("").contains("Parameters"))
        .and_then(|(_, n)| n.bounds())
        .expect("Parameters collapsing header must exist");
    let pos = egui::pos2(((b.x0 + b.x1) * 0.5) as f32, ((b.y0 + b.y1) * 0.5) as f32);
    let events = vec![
        egui::Event::PointerMoved(pos),
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            events,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
}

#[test]
fn surface_viewer_animated_slider_sweeps_continuous_radii() {
    use crate::reflective_ui::slider::{
        has_active_sweeps, reset_frame_sweep_flags, save_slider_state, AnimatedSliderState,
        SweepMode,
    };

    let mut tool = SurfaceViewer::default();
    let initial_major = tool.torus_major_radius;
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    save_slider_state(
        &ctx,
        egui::Id::new("surface_viewer_torus_major"),
        AnimatedSliderState {
            playing: true,
            direction: 1.0,
            sweep_mode: SweepMode::Oscillate,
        },
    );
    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1280.0, 800.0),
    ));
    open_parameters_header(&ctx, &mut tool, screen);
    reset_frame_sweep_flags(&ctx);
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            predicted_dt: 0.1,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert!(tool.torus_major_radius > initial_major);
    assert!(has_active_sweeps(&ctx));
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
    let (segs, _) = restored.ensure_projected_wireframe();
    assert!(!segs.is_empty());
    assert!(
        tool.custom_sandbox
            .set_uv_bounds(SurfaceType::CustomHeightfield, [4.0, 1.0], [-1.0, 1.0])
            .is_err()
    );
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
