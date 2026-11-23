#![cfg_attr(any(), verified(opt_out = "tests"))]

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

use super::spin_interaction::{BlochSteerTarget, b_field_visual_tip};
use super::*;
use crate::camera::ProjectionMode;
use crate::framework::Plot3DViewport;
use crate::url_state::{
    MAX_URL_FRAGMENT_BYTES, encode_share_fragment, parse_share_location, validate_url_preset_json,
};
use verified_engine::engine::VerifiedEngine;

#[test]
fn test_spin_visualizer_initialization_and_reset() {
    let mut viz = SpinVisualizer::default();
    assert_eq!(viz.time, 0.0);
    assert!(viz.paused);

    let ez = viz.sz.expectation_value(&viz.psi).re;
    assert!((ez - 1.0).abs() < 1e-6);

    viz.step_once();
    assert!(viz.paused);
    assert!((viz.time - 0.05).abs() < 1e-9);

    viz.reset();
    assert_eq!(viz.time, 0.0);
    let ez_reset = viz.sz.expectation_value(&viz.psi).re;
    assert!((ez_reset - 1.0).abs() < 1e-6);
}

#[test]
fn test_cached_operators_match_pauli() {
    let viz = SpinVisualizer::default();
    let sx_ref = spin::sigma_x();
    let sy_ref = spin::sigma_y();
    let sz_ref = spin::sigma_z();

    assert_eq!(viz.sx.matrix, sx_ref.matrix);
    assert_eq!(viz.sy.matrix, sy_ref.matrix);
    assert_eq!(viz.sz.matrix, sz_ref.matrix);
}

#[test]
fn test_spin_visualizer_propagator_cache_and_parameter_changes() {
    use math_explorer::physics::quantum::evolve_state;

    let mut viz = SpinVisualizer::default();
    assert_eq!(viz.cached_key, None);
    assert_eq!(viz.scratch.len(), 2);

    viz.b_field = [1.0, 0.5, -0.5];
    viz.step(0.05);
    let key_initial = viz.cached_key;
    assert!(key_initial.is_some());
    let ptr_initial = viz.propagator.matrix.as_ptr();

    viz.step(0.05);
    assert_eq!(viz.cached_key, key_initial);
    assert_eq!(viz.propagator.matrix.as_ptr(), ptr_initial);

    viz.step(0.02);
    let key_dt_changed = viz.cached_key;
    assert_ne!(key_dt_changed, key_initial);

    viz.reset();
    viz.b_field = [2.0, 0.0, 0.0];
    let mut expected = QuantumState::spin_zero();
    let h_mat = (&viz.sx.matrix * Complex::new(2.0, 0.0)) * Complex::new(0.5, 0.0);
    let h_op = QuantumOperator::new(h_mat);
    for _ in 0..4 {
        viz.step(0.05);
        expected = evolve_state(&expected, &h_op, 0.05, 1.0).normalize();
    }
    assert_ne!(viz.cached_key, key_dt_changed);
    assert!((&viz.psi.vector - &expected.vector).norm() < 1e-12);
}

#[test]
fn test_spin_visualizer_viewport_and_watertight_b_arrow_export() {
    let mut viz = SpinVisualizer::default();
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| viz.show(ctx));
    assert!(!egui_plot::get_last_plot_datasets().is_empty());
    assert!(!egui_plot::get_last_plot_datasets_3d().is_empty());
    assert!(!egui_plot::get_last_plot_primitives().is_empty());

    let cm_with_b = viz
        .export_colored_mesh_3d()
        .expect("Bloch 3D export hook present")
        .expect("Bloch 3D mesh builds");
    assert_eq!(cm_with_b.colors.len(), cm_with_b.mesh.vertices.len());
    assert!(cm_with_b.mesh.normals.iter().all(|n| n.is_finite()));
    assert_eq!(
        crate::tabs::geometry_topology::export_utils::count_open_and_nonmanifold_edges(
            &cm_with_b.mesh
        ),
        (0, 0)
    );
    assert!(crate::tabs::geometry_topology::export_utils::signed_volume(&cm_with_b.mesh) > 0.0);

    viz.b_field = [0.0, 0.0, 0.0];
    let cm_zero_b = viz.export_colored_mesh_3d().unwrap().unwrap();
    assert!(cm_with_b.mesh.vertices.len() > cm_zero_b.mesh.vertices.len());
}

#[test]
fn test_camera_unproject_to_unit_sphere_parallel_and_perspective() {
    for mode in [ProjectionMode::Parallel, ProjectionMode::Perspective] {
        let mut cam = Camera3D::new(0.45, -0.65, 1.35);
        cam.pan_x = 0.15;
        cam.pan_y = -0.10;
        cam.projection = mode;

        // Front-facing camera-space unit point round-trips through project -> unproject
        let cam_space = [0.35, -0.25, (1.0_f64 - 0.35 * 0.35 - 0.25 * 0.25).sqrt()];
        let world_in = cam.rotate_from_camera_space(&cam_space);
        let proj = cam.project(&world_in);
        let world_out = cam.unproject_to_unit_sphere(proj);
        for k in 0..3 {
            assert!(
                (world_in[k] - world_out[k]).abs() < 1e-9,
                "mode={mode:?} axis {k}: expected {}, got {}",
                world_in[k],
                world_out[k]
            );
        }

        // Far outside silhouette clamps smoothly onto S^2 with unit norm
        let rim = cam.unproject_to_unit_sphere([25.0, -18.0]);
        let norm = (rim[0] * rim[0] + rim[1] * rim[1] + rim[2] * rim[2]).sqrt();
        assert!((norm - 1.0).abs() < 1e-12);

        let fallback = cam.unproject_to_unit_sphere([f64::NAN, 0.0]);
        assert!(fallback.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn test_plot3d_viewport_allow_camera_drag_guard() {
    let ctx = egui::Context::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));
    let mut cam = Camera3D::new(0.4, 0.5, 1.0);
    let initial = cam;

    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |c| {
            egui::CentralPanel::default().show(c, |ui| {
                let _ = Plot3DViewport::new("guard_test", 1.5)
                    .allow_camera_drag(false)
                    .show(ui, &mut cam, |_| {});
            });
        },
    );
    let center = egui::pos2(400.0, 300.0);
    let moved = egui::pos2(480.0, 360.0);
    for evs in [
        vec![
            egui::Event::PointerMoved(center),
            egui::Event::PointerButton {
                pos: center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![egui::Event::PointerMoved(moved)],
        vec![egui::Event::PointerButton {
            pos: moved,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ] {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                events: evs,
                ..Default::default()
            },
            |c| {
                egui::CentralPanel::default().show(c, |ui| {
                    let _ = Plot3DViewport::new("guard_test", 1.5)
                        .allow_camera_drag(false)
                        .show(ui, &mut cam, |_| {});
                });
            },
        );
    }
    assert_eq!(cam.yaw, initial.yaw);
    assert_eq!(cam.pitch, initial.pitch);
}

fn drag_viz_handle(
    ctx: &egui::Context,
    viz: &mut SpinVisualizer,
    screen: egui::Rect,
    start: egui::Pos2,
    dest: egui::Pos2,
) {
    for evs in [
        vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![egui::Event::PointerMoved(dest)],
        vec![egui::Event::PointerButton {
            pos: dest,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
    ] {
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                events: evs,
                ..Default::default()
            },
            |c| viz.show(c),
        );
    }
}

#[test]
fn test_bloch_pointer_drag_steers_spin_and_b_without_rotating_camera() {
    let ctx = egui::Context::default();
    let mut viz = SpinVisualizer {
        b_field: [1.5, 0.0, 0.0],
        ..Default::default()
    };
    let cam_before = viz.camera;
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));

    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |c| viz.show(c),
    );
    let rect = viz.last_plot_rect.expect("plot rect recorded");
    let bounds = viz.last_plot_bounds.expect("plot bounds recorded");
    let to_px = |pt: [f64; 3]| {
        let p = cam_before.project(&pt);
        let u = ((p[0] - bounds[0]) / (bounds[1] - bounds[0])) as f32;
        let v = ((bounds[3] - p[1]) / (bounds[3] - bounds[2])) as f32;
        egui::pos2(
            rect.left() + u * rect.width(),
            rect.top() + v * rect.height(),
        )
    };

    let spin_start = to_px([0.0, 0.0, 1.0]);
    let spin_dest = to_px([0.85, 0.0, (1.0_f64 - 0.85 * 0.85).sqrt()]);
    drag_viz_handle(&ctx, &mut viz, screen, spin_start, spin_dest);
    assert_eq!(viz.camera.yaw, cam_before.yaw);
    assert_eq!(viz.camera.pitch, cam_before.pitch);
    assert!(viz.sx.expectation_value(&viz.psi).re > 0.5);

    let (b_pt, _) = b_field_visual_tip(viz.b_field).unwrap();
    let b_start = to_px([b_pt.x, b_pt.y, b_pt.z]);
    drag_viz_handle(&ctx, &mut viz, screen, b_start, to_px([0.0, 1.2, 0.2]));
    assert_eq!(viz.camera.yaw, cam_before.yaw);
    assert!(viz.b_field[1] > 0.5);
}

#[test]
fn test_bloch_keyboard_arrow_stepping_and_non_trapping_tab_focus() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    crate::accessibility::clear_test_announcements();
    let mut viz = SpinVisualizer {
        b_field: [1.5, 0.0, 0.0],
        ..Default::default()
    };
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 800.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(screen),
            ..Default::default()
        },
        |c| viz.show(c),
    );
    let center = viz.last_plot_rect.unwrap().center();
    drag_viz_handle(&ctx, &mut viz, screen, center, center);

    let theta_before = viz.current_initial_angles()[0];
    for key in [egui::Key::ArrowDown, egui::Key::Tab] {
        let focused_before = ctx.memory(|m| m.focused());
        let _ = ctx.run(
            egui::RawInput {
                screen_rect: Some(screen),
                events: vec![egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                }],
                ..Default::default()
            },
            |c| viz.show(c),
        );
        if key == egui::Key::ArrowDown {
            assert!(viz.current_initial_angles()[0] > theta_before + 0.04);
            assert!(
                crate::accessibility::all_announcements()
                    .iter()
                    .any(|(msg, prio)| prio == "polite" && msg.contains("theta"))
            );
        } else {
            assert_ne!(ctx.memory(|m| m.focused()), focused_before);
        }
    }
    let b_msg = viz.step_steer_angles(BlochSteerTarget::MagneticField, 0.08, 0.12);
    assert!(b_msg.contains("Magnetic field B"));
}

#[test]
fn test_initial_bloch_angles_preset_url_round_trip_and_zero_alloc_step() {
    let mut viz = SpinVisualizer::default();
    viz.set_initial_bloch_angles(FRAC_PI_2, FRAC_PI_4);
    viz.b_field = [0.75, -0.5, 1.25];

    let preset_json = viz.export_preset_json().expect("preset JSON");
    validate_url_preset_json(&preset_json).expect("within URL state budget");
    let frag = encode_share_fragment(
        "quantum-mechanics",
        Some("SpinVisualizer"),
        Some(&preset_json),
    )
    .expect("URL share fragment");
    assert!(frag.len() <= MAX_URL_FRAGMENT_BYTES);
    let parsed = parse_share_location(&frag, "").unwrap().unwrap();

    let mut restored = SpinVisualizer::default();
    restored
        .import_preset_json(parsed.state_json.as_deref().unwrap())
        .expect("preset restores");
    let [th, ph] = restored.current_initial_angles();
    assert!((th - FRAC_PI_2).abs() < 1e-12 && (ph - FRAC_PI_4).abs() < 1e-12);
    assert!((restored.sx.expectation_value(&restored.psi).re - FRAC_PI_4.cos()).abs() < 1e-12);

    assert!(
        restored
            .try_load_state(
                r#"{"b_field":[0.0,0.0,1.0],"paused":true,"initial_bloch_angles":[9.0,0.0]}"#
            )
            .is_err()
    );

    let _guard = crate::VERIFIED_ALLOC_TEST_MUTEX
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    VerifiedEngine::run_verified(|| {
        for idx in 0..1_000usize {
            restored.b_field[0] = ((idx % 8) as f64) * 0.25;
            restored.step(0.01);
        }
    });
}
