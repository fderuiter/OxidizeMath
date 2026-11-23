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
    assert!(!egui_plot::get_last_plot_datasets_3d().is_empty());

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
    custom_restored.populate_scene(egui::Color32::WHITE);
    assert!(!custom_restored
        .scene_builder
        .sorted_projected_segments()
        .is_empty());
}

fn open_parameters_header(
    ctx: &egui::Context,
    tool: &mut CurvatureHeatmap,
    screen: Option<egui::Rect>,
) {
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
fn curvature_heatmap_animated_slider_sweeps_continuous_radii() {
    use crate::reflective_ui::slider::{save_slider_state, AnimatedSliderState, SweepMode};

    let mut tool = CurvatureHeatmap::default();
    let initial_r = tool.torus_minor_radius;
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    save_slider_state(
        &ctx,
        egui::Id::new("curvature_heatmap_torus_minor"),
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
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            predicted_dt: 0.1,
            ..Default::default()
        },
        |ctx| tool.show(ctx),
    );
    assert!(tool.torus_minor_radius > initial_r);
}

#[test]
fn curvature_heatmap_clamps_1000_grid_caches_curvature_on_orbit_and_caps_batches() {
    let mut tool = CurvatureHeatmap::default();
    let high_res = r#"{"surface_type":"Torus","curvature_type":"Gaussian","sphere_radius":1.0,"torus_major_radius":2.0,"torus_minor_radius":0.5,"klein_radius":2.0,"u_resolution":1000,"v_resolution":1000,"pitch":0.4,"yaw":0.8,"zoom":1.2}"#;
    tool.try_load_state(high_res)
        .expect("1000x1000 curvature state should clamp cleanly");
    assert_eq!(
        tool.u_resolution,
        crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION
    );
    assert_eq!(
        tool.v_resolution,
        crate::presets::MAX_INTERACTIVE_SURFACE_RESOLUTION
    );

    let mut orbit_tool = CurvatureHeatmap {
        u_resolution: 40,
        v_resolution: 40,
        ..Default::default()
    };
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |c| orbit_tool.show(c));
    egui_plot::clear_last_plot_datasets();
    let _ = ctx.run(egui::RawInput::default(), |c| orbit_tool.show(c));
    assert_eq!(orbit_tool.cache.curvature_eval_count, 1);
    assert_eq!(orbit_tool.cache.projection_count, 1);
    let datasets = egui_plot::get_last_plot_datasets();
    assert!(!datasets.is_empty() && datasets.len() <= cache::CURVATURE_COLOR_BINS);

    orbit_tool.camera.yaw += 0.5;
    orbit_tool.camera.pitch -= 0.25;
    orbit_tool.camera.zoom *= 1.2;
    let _ = ctx.run(egui::RawInput::default(), |c| orbit_tool.show(c));
    assert_eq!(orbit_tool.cache.curvature_eval_count, 1);
    assert_eq!(orbit_tool.cache.projection_count, 2);
}
