#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::tabs::{
    chaos::attractors::AttractorPlotter,
    geometry_topology::{curvature_heatmap::CurvatureHeatmap, surface_viewer::SurfaceViewer},
    quantum::spin_viz::SpinVisualizer,
    solid_state::crystal_viewer::CrystalViewer,
};
use nalgebra::Point3;

fn interior_intersection(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Option<(f64, f64)> {
    let dax = a1[0] - a0[0];
    let day = a1[1] - a0[1];
    let dbx = b1[0] - b0[0];
    let dby = b1[1] - b0[1];
    let denom = dax * dby - day * dbx;
    if denom.abs() < 1e-10 {
        return None;
    }
    let rx = b0[0] - a0[0];
    let ry = b0[1] - a0[1];
    let t = (rx * dby - ry * dbx) / denom;
    let u = (rx * day - ry * dax) / denom;
    (0.02..0.98)
        .contains(&t)
        .then_some((t, u))
        .filter(|(_, u_val)| (0.02..0.98).contains(u_val))
}

fn count_rear_over_front_overdraws(segments: &[([f64; 2], [f64; 2], f64)]) -> usize {
    let mut overdraws = 0usize;
    for i in 0..segments.len() {
        let (a0, a1, za) = segments[i];
        for &(b0, b1, zb) in &segments[(i + 1)..] {
            if za > zb + 1e-9 && interior_intersection(a0, a1, b0, b1).is_some() {
                overdraws += 1;
            }
        }
    }
    overdraws
}

fn run_viewport_frame(
    ctx: &egui::Context,
    id: &'static str,
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
    camera: &mut Camera3D,
) -> egui::Id {
    let mut captured_id = egui::Id::NULL;
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            modifiers,
            events,
            ..Default::default()
        },
        |sub_ctx| {
            egui::CentralPanel::default().show(sub_ctx, |ui| {
                captured_id = Plot3DViewport::new(id, 4.0)
                    .show(ui, camera, |_| {})
                    .response
                    .id;
            });
        },
    );
    captured_id
}

fn simulate_pointer_drag(
    ctx: &egui::Context,
    id: &'static str,
    button: egui::PointerButton,
    modifiers: egui::Modifiers,
    p0: egui::Pos2,
    p1: egui::Pos2,
    camera: &mut Camera3D,
) {
    let down = egui::Event::PointerButton {
        pos: p0,
        button,
        pressed: true,
        modifiers,
    };
    let up = egui::Event::PointerButton {
        pos: p1,
        button,
        pressed: false,
        modifiers,
    };
    run_viewport_frame(
        ctx,
        id,
        vec![egui::Event::PointerMoved(p0), down],
        modifiers,
        camera,
    );
    run_viewport_frame(
        ctx,
        id,
        vec![egui::Event::PointerMoved(p1)],
        modifiers,
        camera,
    );
    run_viewport_frame(ctx, id, vec![up], modifiers, camera);
}

#[test]
fn test_camera_shift_drag_pans_while_secondary_drag_is_ignored() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::new(0.5, 0.5, 1.0);
    run_viewport_frame(
        &ctx,
        "cam_pan_test",
        Vec::new(),
        egui::Modifiers::NONE,
        &mut cam,
    );
    let (p0, p1) = (egui::pos2(400.0, 300.0), egui::pos2(460.0, 260.0));

    simulate_pointer_drag(
        &ctx,
        "cam_pan_test",
        egui::PointerButton::Primary,
        egui::Modifiers::SHIFT,
        p0,
        p1,
        &mut cam,
    );
    assert!(cam.pan_x > 0.0 && cam.pan_y > 0.0);
    assert_eq!((cam.pitch, cam.yaw), (0.5, 0.5));

    let snapshot = cam;
    simulate_pointer_drag(
        &ctx,
        "cam_pan_test",
        egui::PointerButton::Secondary,
        egui::Modifiers::NONE,
        p0,
        p1,
        &mut cam,
    );
    assert_eq!(cam, snapshot);
}

fn touch_pair_events(phase: egui::TouchPhase, p0: egui::Pos2, p1: egui::Pos2) -> [egui::Event; 2] {
    let device_id = egui::TouchDeviceId(1);
    [
        egui::Event::Touch {
            device_id,
            id: egui::TouchId(1),
            phase,
            pos: p0,
            force: None,
        },
        egui::Event::Touch {
            device_id,
            id: egui::TouchId(2),
            phase,
            pos: p1,
            force: None,
        },
    ]
}

#[test]
fn test_camera_two_finger_touch_translation_pans_without_orbiting() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::new(0.5, 0.5, 1.0);
    let (t0, t1) = (egui::pos2(350.0, 300.0), egui::pos2(390.0, 270.0));
    run_viewport_frame(
        &ctx,
        "cam_touch_test",
        vec![egui::Event::PointerMoved(t0)],
        egui::Modifiers::NONE,
        &mut cam,
    );
    let [s0, s1] = touch_pair_events(egui::TouchPhase::Start, t0, egui::pos2(450.0, 300.0));
    let [m0, m1] = touch_pair_events(egui::TouchPhase::Move, t1, egui::pos2(490.0, 270.0));
    let [e0, e1] = touch_pair_events(egui::TouchPhase::End, t1, egui::pos2(490.0, 270.0));
    let down = egui::Event::PointerButton {
        pos: t0,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: egui::Modifiers::NONE,
    };
    let up = egui::Event::PointerButton {
        pos: t1,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    };
    run_viewport_frame(
        &ctx,
        "cam_touch_test",
        vec![egui::Event::PointerMoved(t0), down, s0, s1],
        egui::Modifiers::NONE,
        &mut cam,
    );
    run_viewport_frame(
        &ctx,
        "cam_touch_test",
        vec![egui::Event::PointerMoved(t1), m0, m1],
        egui::Modifiers::NONE,
        &mut cam,
    );
    run_viewport_frame(
        &ctx,
        "cam_touch_test",
        vec![up, e0, e1],
        egui::Modifiers::NONE,
        &mut cam,
    );
    assert!(cam.pan_x > 0.0 && cam.pan_y > 0.0);
    assert_eq!((cam.pitch, cam.yaw), (0.5, 0.5));
}

#[test]
fn test_camera_shift_wasd_projection_shortcut_and_reset() {
    let ctx = egui::Context::default();
    let mut cam = Camera3D::new(0.5, 0.5, 1.0);
    let plot_id = run_viewport_frame(
        &ctx,
        "cam_key_test",
        Vec::new(),
        egui::Modifiers::NONE,
        &mut cam,
    );
    ctx.memory_mut(|m| m.request_focus(plot_id));
    run_viewport_frame(
        &ctx,
        "cam_key_test",
        vec![egui::Event::Key {
            key: egui::Key::D,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::SHIFT,
        }],
        egui::Modifiers::SHIFT,
        &mut cam,
    );
    assert!(cam.pan_x > 0.0);

    run_viewport_frame(
        &ctx,
        "cam_key_test",
        vec![egui::Event::Key {
            key: egui::Key::P,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        egui::Modifiers::NONE,
        &mut cam,
    );
    assert_eq!(cam.projection, ProjectionMode::Perspective);
    cam.reset();
    assert_eq!(cam, Camera3D::new(0.5, 0.5, 1.0));
}

#[test]
fn test_scene3d_builder_cuts_rear_over_front_overdraws_to_zero_across_five_tools() {
    let mut curvature = CurvatureHeatmap::default();
    curvature.populate_scene(egui::Color32::WHITE);
    let curv_sorted = curvature.scene_builder.sorted_projected_segments();
    let mut curv_reversed = curv_sorted.clone();
    curv_reversed.reverse();
    assert!(
        count_rear_over_front_overdraws(&curv_reversed) > 700,
        "Expected >700 rear-over-front overdraws without front-to-back order on 30x30 torus"
    );
    assert_eq!(count_rear_over_front_overdraws(&curv_sorted), 0);

    let mut surface = SurfaceViewer::default();
    surface.populate_scene(egui::Color32::WHITE);
    let surf_sorted = surface.scene_builder.sorted_projected_segments();
    let mut surf_reversed = surf_sorted.clone();
    surf_reversed.reverse();
    assert!(count_rear_over_front_overdraws(&surf_reversed) > 700);
    assert_eq!(count_rear_over_front_overdraws(&surf_sorted), 0);

    let mut spin = SpinVisualizer::default();
    spin.populate_scene(
        0.4,
        -0.5,
        0.75,
        egui::Color32::GRAY,
        egui::Color32::RED,
        true,
    );
    assert_eq!(
        count_rear_over_front_overdraws(&spin.scene_builder.sorted_projected_segments()),
        0
    );

    let mut attractor = AttractorPlotter::default();
    for _ in 0..40 {
        crate::framework::InteractiveTool::step_simulation(&mut attractor);
    }
    attractor.populate_scene(egui::Color32::LIGHT_BLUE);
    assert_eq!(
        count_rear_over_front_overdraws(&attractor.scene_builder.sorted_projected_segments()),
        0
    );

    let mut crystal = CrystalViewer::default();
    crystal.populate_scene(egui::Color32::GRAY, egui::Color32::RED);
    assert_eq!(
        count_rear_over_front_overdraws(&crystal.scene_builder.sorted_projected_segments()),
        0
    );
}

#[test]
fn test_scene3d_builder_axes_and_floor_grid_project_in_parallel_and_perspective() {
    let mut cam = Camera3D::new(0.5, 0.5, 1.0);
    let mut builder_par = Scene3DBuilder::new(cam);
    builder_par.add_axes(2.0, true);
    builder_par.add_floor_grid(
        2.0,
        4,
        -1.0,
        egui::Stroke::new(1.0_f32, egui::Color32::GRAY),
    );
    let segs_par = builder_par.sorted_projected_segments();
    assert_eq!(segs_par.len(), 3 + 10);

    cam.projection = ProjectionMode::Perspective;
    let mut builder_persp = Scene3DBuilder::new(cam);
    builder_persp.add_segment(
        "DepthTest",
        Point3::new(1.0, 0.0, -1.0),
        Point3::new(1.0, 0.0, 1.0),
        egui::Stroke::new(1.0_f32, egui::Color32::WHITE),
        true,
    );
    let segs_persp = builder_persp.sorted_projected_segments();
    assert_ne!(segs_persp[0].0, segs_persp[0].1);
}
