#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::camera::CAMERA_3D_HELPER_TEXT;

fn test_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1000.0, 720.0),
        )),
        events,
        ..Default::default()
    }
}

#[test]
fn test_gaussian_splatting_renders_interactive_3d_scene_without_placeholder() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = GaussianSplattingTool::default();

    let out = ctx.run(test_input(Vec::new()), |sub_ctx| tool.show(sub_ctx));

    assert_eq!(tool.projected_splats.len(), 7);
    assert!(tool.texture.is_some());
    let lit_pixels = tool
        .pixel_buffer
        .iter()
        .filter(|px| px.a() > 16)
        .count();
    assert!(
        lit_pixels > 200,
        "Expected alpha-blended Gaussian splats to illuminate scene pixels, got {lit_pixels}"
    );
    assert!(!egui_plot::get_last_plot_datasets().is_empty());
    assert!(!egui_plot::get_last_plot_primitives().is_empty());

    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit output expected");
    let texts: Vec<&str> = ak
        .nodes
        .iter()
        .flat_map(|(_, n)| [n.label(), n.value()])
        .flatten()
        .collect();
    assert!(
        !texts.iter().any(|l| l.contains("Placeholder")),
        "Placeholder copy must not appear in Gaussian Splatting UI"
    );
    assert!(
        texts.iter().any(|l| l.contains("Reset View")),
        "Reset View button must be visible on initial load"
    );
    assert!(
        texts.iter().any(|l| l.contains(CAMERA_3D_HELPER_TEXT)),
        "Canonical 3D control helper text must be rendered"
    );
}

#[test]
fn test_gaussian_splatting_orbit_zoom_scale_and_reset() {
    let ctx = egui::Context::default();
    let mut tool = GaussianSplattingTool::default();

    let _ = ctx.run(test_input(Vec::new()), |sub_ctx| tool.show(sub_ctx));
    let initial_centers = tool.splat_centers.clone();

    // Drag across central canvas to orbit 3D splats around origin
    let start = egui::pos2(620.0, 360.0);
    let end = egui::pos2(700.0, 300.0);
    let drag_seq = [
        vec![
            egui::Event::PointerMoved(start),
            egui::Event::PointerButton {
                pos: start,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        vec![egui::Event::PointerMoved(end)],
        vec![
            egui::Event::PointerButton {
                pos: end,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::Zoom(1.8),
        ],
    ];
    for evs in drag_seq {
        let _ = ctx.run(test_input(evs), |sub_ctx| tool.show(sub_ctx));
    }
    let _ = ctx.run(test_input(Vec::new()), |sub_ctx| tool.show(sub_ctx));

    assert!((tool.camera.yaw - 0.5).abs() > 0.1);
    assert!((tool.camera.pitch - 0.5).abs() > 0.1);
    assert!((tool.camera.zoom - 1.8).abs() < 1e-4);
    assert_ne!(tool.splat_centers, initial_centers);

    // Changing scale enlarges projected covariance ellipse span
    let ring_before = tool.ellipse_rings[0].0;
    tool.scale = 2.2;
    let _ = ctx.run(test_input(Vec::new()), |sub_ctx| tool.show(sub_ctx));
    let ring_after = tool.ellipse_rings[0].0;
    let span_before = (ring_before[0][0] - ring_before[12][0]).hypot(ring_before[0][1] - ring_before[12][1]);
    let span_after = (ring_after[0][0] - ring_after[12][0]).hypot(ring_after[0][1] - ring_after[12][1]);
    assert!(span_after > span_before * 1.8);

    // Keyboard R resets orientation and zoom while canvas retains focus
    for pressed in [true, false] {
        let _ = ctx.run(
            test_input(vec![egui::Event::Key {
                key: egui::Key::R,
                physical_key: None,
                pressed,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            }]),
            |sub_ctx| tool.show(sub_ctx),
        );
    }
    assert_eq!(tool.camera, Camera3D::new(0.5, 0.5, 1.0));
}

#[test]
fn test_gaussian_splatting_state_persistence_and_legacy_presets() {
    let mut tool = GaussianSplattingTool {
        scale: 2.75,
        camera: Camera3D::new(-0.35, 1.15, 18.5),
        ..Default::default()
    };
    let saved = tool.save_state().expect("save_state should succeed");

    let mut restored = GaussianSplattingTool::default();
    restored
        .try_load_state(&saved)
        .expect("saved state with camera should load");
    assert!((restored.scale - 2.75).abs() < 1e-9);
    assert!((restored.camera.pitch + 0.35).abs() < 1e-6);
    assert!((restored.camera.yaw - 1.15).abs() < 1e-6);
    assert!((restored.camera.zoom - 18.5).abs() < 1e-5);

    // Legacy preset without camera fields restores default camera pose
    restored
        .try_load_state(r#"{"scale":1.5}"#)
        .expect("legacy scale-only preset should load");
    assert!((restored.scale - 1.5).abs() < 1e-9);
    assert_eq!(restored.camera, Camera3D::default());

    // Legacy flat pitch/yaw/zoom also loads cleanly
    restored
        .try_load_state(r#"{"scale":1.2,"pitch":0.2,"yaw":0.8,"zoom":0.05}"#)
        .expect("legacy flat camera preset should load");
    assert!((restored.camera.zoom - 0.05).abs() < 1e-6);

    // Starter preset envelope roundtrips cleanly
    tool.import_preset_json(STARTER_PRESET_JSON)
        .expect("starter preset should import");
    assert!((tool.scale - 1.5).abs() < 1e-9);
    assert!((tool.camera.zoom - 1.15).abs() < 1e-6);

    // Out-of-bounds scale or zoom is rejected
    assert!(restored.try_load_state(r#"{"scale":0.0}"#).is_err());
    assert!(restored
        .try_load_state(r#"{"scale":1.0,"camera":{"pitch":0.0,"yaw":0.0,"zoom":150.0}}"#)
        .is_err());
}
