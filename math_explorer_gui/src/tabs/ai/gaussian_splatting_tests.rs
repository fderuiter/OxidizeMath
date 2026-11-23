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
fn test_gaussian_splatting_image_and_density_respond_to_scale_and_camera() {
    let small = GaussianSplattingTool {
        scale: 0.3,
        ..Default::default()
    };
    let large = GaussianSplattingTool {
        scale: 2.5,
        ..Default::default()
    };
    let orbited = GaussianSplattingTool {
        scale: 0.3,
        camera: Camera3D::new(-0.4, 1.2, 1.35),
        ..Default::default()
    };

    let img_small = small.render_splat_image();
    let img_large = large.render_splat_image();
    let img_orbited = orbited.render_splat_image();
    assert_eq!(img_small.size, [SPLAT_GRID_SIZE, SPLAT_GRID_SIZE]);
    assert_ne!(img_small.pixels, img_large.pixels);
    assert_ne!(img_small.pixels, img_orbited.pixels);

    let counts_small = GaussianSplattingTool::compute_density_counts_for_scale(0.3);
    let counts_large = GaussianSplattingTool::compute_density_counts_for_scale(2.5);
    assert!(counts_small.clone > counts_large.clone);
    assert!(counts_large.split > counts_small.split);
    assert_eq!(counts_small.prune, 1);
    assert_eq!(counts_small.keep, 3);
}

#[test]
fn test_gaussian_splatting_renders_interactive_3d_scene_without_placeholder() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut tool = GaussianSplattingTool::default();

    let out = ctx.run(test_input(Vec::new()), |sub_ctx| tool.show(sub_ctx));

    assert_eq!(tool.projected_splats.len(), 8);
    assert!(tool.texture.is_some());
    let lit_pixels = tool.pixel_buffer.iter().filter(|px| px.a() > 16).count();
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
        texts.iter().any(|l| l.contains("Adaptive Density Control")),
        "Adaptive Density Control metrics must be rendered in side panel"
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
    let span_before =
        (ring_before[0][0] - ring_before[12][0]).hypot(ring_before[0][1] - ring_before[12][1]);
    let span_after =
        (ring_after[0][0] - ring_after[12][0]).hypot(ring_after[0][1] - ring_after[12][1]);
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
    assert_eq!(
        restored.cached_counts,
        GaussianSplattingTool::compute_density_counts_for_scale(2.75)
    );

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

#[test]
fn test_gaussian_splatting_16_9_frustum_rasterizes_beyond_2_5_and_thresholds_vary_all_actions() {
    let ctx = egui::Context::default();
    let mut tool = GaussianSplattingTool::default();
    tool.camera.projection = ProjectionMode::Perspective;
    tool.camera.pan_x = 3.2;

    let widescreen = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1920.0, 1080.0),
        )),
        ..Default::default()
    };
    let _ = ctx.run(widescreen, |sub_ctx| tool.show(sub_ctx));

    let (x_half, y_half) = tool.frustum_half_extents;
    assert!(
        x_half > 3.5 && (y_half - 2.5).abs() < 1e-6,
        "Expected 16:9 viewport x_half > 3.5 without [-2.5, 2.5] clipping, got ({x_half}, {y_half})"
    );
    // Right-hand columns correspond to x > +2.5 where panned splats reside
    let min_right_col =
        (((2.6 + x_half) / (2.0 * x_half)) * (SPLAT_GRID_SIZE as f64)).ceil() as usize;
    let right_lit = (0..SPLAT_GRID_SIZE)
        .flat_map(|r| (min_right_col..SPLAT_GRID_SIZE).map(move |c| (r, c)))
        .filter(|&(r, c)| {
            let idx = math_commons::math_kernel::types::flatten_2d_index(c, r, SPLAT_GRID_SIZE);
            tool.pixel_buffer[idx].a() > 16
        })
        .count();
    assert!(
        right_lit > 10,
        "Expected unclipped radiance beyond x > 2.5 in 16:9 frustum, got {right_lit} lit pixels"
    );

    // Verify all four density counts (Split, Clone, Keep, Prune) vary across threshold sliders
    let base = GaussianSplattingTool::compute_density_counts(1.0, 0.10, 0.05);
    let low_grad = GaussianSplattingTool::compute_density_counts(1.0, 0.03, 0.05);
    let high_grad = GaussianSplattingTool::compute_density_counts(1.0, 0.35, 0.05);
    let low_opac = GaussianSplattingTool::compute_density_counts(1.0, 0.10, 0.01);
    let high_opac = GaussianSplattingTool::compute_density_counts(1.0, 0.10, 0.85);

    assert_ne!(base.split, low_grad.split);
    assert_ne!(base.clone, high_grad.clone);
    assert_ne!(base.keep, high_grad.keep);
    assert_eq!(low_opac.prune, 0);
    assert!(high_opac.prune > base.prune);

    tool.grad_threshold = 0.22;
    tool.opacity_threshold = 0.60;
    let json = tool.save_state().expect("threshold state saves");
    let mut roundtripped = GaussianSplattingTool::default();
    roundtripped
        .try_load_state(&json)
        .expect("threshold state loads");
    assert!((roundtripped.grad_threshold - 0.22).abs() < 1e-9);
    assert!((roundtripped.opacity_threshold - 0.60).abs() < 1e-9);
    assert_eq!(
        roundtripped.cached_counts,
        GaussianSplattingTool::compute_density_counts(1.0, 0.22, 0.60)
    );
}

#[test]
fn test_gaussian_splatting_exports_valid_inria_3dgs_ply_binary() {
    let tool = GaussianSplattingTool {
        scale: 1.5,
        ..Default::default()
    };
    let caps = tool.export_capabilities();
    assert!(!caps.dataset && !caps.figure);
    assert!(caps.ply && !caps.stl && !caps.obj);

    let ply_bytes = tool
        .export_splat_ply_3d()
        .expect("3DGS PLY hook present")
        .expect("3DGS PLY serializes");
    let header_end = ply_bytes
        .windows(b"end_header\n".len())
        .position(|w| w == b"end_header\n")
        .expect("end_header present")
        + b"end_header\n".len();
    let header = std::str::from_utf8(&ply_bytes[..header_end]).expect("ASCII PLY header");
    assert!(header.contains("element vertex 8\n"));
    assert!(header.contains("property float f_dc_0\n"));
    assert!(header.contains("property float opacity\n"));
    assert!(header.contains("property float scale_0\n"));
    assert!(header.contains("property float rot_0\n"));
    // 8 splats * 17 f32 properties * 4 bytes = 544 payload bytes
    assert_eq!(ply_bytes.len() - header_end, 8 * 17 * 4);
}
