#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::async_sim::SimulationRunner;
use crate::framework::InteractiveTool;
use crate::web_io::PlatformInstant;
use eframe::egui;
use std::time::Duration;

fn raw_input_for_size(width: f32, height: f32, max_texture_side: Option<usize>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(width, height),
        )),
        max_texture_side,
        ..Default::default()
    }
}

fn find_image_rect(output: &egui::FullOutput, screen: egui::Rect) -> egui::Rect {
    if let Some(update) = &output.platform_output.accesskit_update {
        for (_id, node) in &update.nodes {
            if node.role() == egui::accesskit::Role::Image {
                if let Some(b) = node.bounds() {
                    return egui::Rect::from_min_max(
                        egui::pos2(b.x0 as f32, b.y0 as f32),
                        egui::pos2(b.x1 as f32, b.y1 as f32),
                    );
                }
            }
        }
    }
    egui::Rect::from_min_max(
        egui::pos2(screen.width() * 0.35, 20.0),
        egui::pos2(screen.width() - 10.0, screen.height() - 10.0),
    )
}

fn run_drag_sequence(
    ctx: &egui::Context,
    viewer: &mut FractalViewer,
    start_pos: egui::Pos2,
    end_pos: egui::Pos2,
) -> Duration {
    let mut press = raw_input_for_size(1024.0, 768.0, Some(2048));
    press.events = vec![
        egui::Event::PointerMoved(start_pos),
        egui::Event::PointerButton {
            pos: start_pos,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    let _ = ctx.run(press, |c| viewer.show(c));

    let mut drag = raw_input_for_size(1024.0, 768.0, Some(2048));
    drag.events = vec![egui::Event::PointerMoved(end_pos)];
    let t0 = PlatformInstant::now();
    let _ = ctx.run(drag, |c| viewer.show(c));
    let drag_elapsed = t0.elapsed();

    let mut release = raw_input_for_size(1024.0, 768.0, Some(2048));
    release.events = vec![egui::Event::PointerButton {
        pos: end_pos,
        button: egui::PointerButton::Primary,
        pressed: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let _ = ctx.run(release, |c| viewer.show(c));
    drag_elapsed
}

#[test]
fn test_progressive_strides_and_exact_final_parity_mandelbrot_and_julia() {
    for mode in [FractalMode::Mandelbrot, FractalMode::Julia] {
        let req = FractalRenderRequest {
            generation: 1,
            width: 96,
            height: 72,
            center: Complex::new(-0.5, 0.1),
            zoom: 1.5,
            max_iter: 120,
            mode,
            julia_c: Complex::new(-0.7, 0.27015),
            high_contrast: false,
        };
        let init_req = FractalRenderRequest {
            generation: 0,
            width: 16,
            height: 12,
            ..req
        };
        let shared = Arc::new(RwLock::new(init_req));
        let latest_gen = Arc::new(AtomicU64::new(0));
        let mut runner = FractalRunner::new(Arc::clone(&shared), Arc::clone(&latest_gen));

        *shared.write().unwrap() = req;
        latest_gen.store(1, Ordering::Release);

        let mut observed_strides = Vec::new();
        for _ in 0..512 {
            runner.step();
            let snap = runner.get_snapshot();
            let meta = snap
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<FractalPassMeta>())
                .copied()
                .expect("FractalPassMeta expected");
            if observed_strides.last().copied() != Some(meta.stride) {
                observed_strides.push(meta.stride);
            }
            if meta.is_final {
                break;
            }
        }

        assert_eq!(observed_strides, vec![8, 4, 2, 1]);
        assert!(runner.is_complete());

        let final_snap = runner.get_snapshot();
        let expected = render_fractal_color_image(&req);
        let actual = final_snap.pixels.read().unwrap().clone();
        assert_eq!(actual, expected.pixels);
    }
}

#[test]
fn test_drag_140_pixels_updates_center_and_cancels_active_worker_pass() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut viewer = FractalViewer {
        max_iter: math_commons::registry::MAX_ITERATIONS as u32,
        ..Default::default()
    };
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));

    let out0 = ctx.run(raw_input_for_size(1024.0, 768.0, Some(2048)), |c| {
        viewer.show(c);
    });
    let img_rect = find_image_rect(&out0, screen);
    let view_w = f64::from(img_rect.width());
    let view_h = f64::from(img_rect.height());
    let gen_before_drag = viewer.request_generation;
    let initial_re = viewer.center.re;
    let initial_im = viewer.center.im;

    let start_pos = img_rect.center();
    let drag_elapsed = run_drag_sequence(
        &ctx,
        &mut viewer,
        start_pos,
        start_pos + egui::vec2(140.0, 0.0),
    );

    let scale_y = 3.0 / viewer.zoom;
    let scale_x = scale_y * (view_w / view_h);
    let expected_re = (initial_re - (140.0 / view_w) * scale_x).clamp(COORD_MIN, COORD_MAX);
    assert!(
        (viewer.center.re - expected_re).abs() < 1e-5,
        "Expected center.re ≈ {expected_re}, got {}",
        viewer.center.re
    );
    assert!((viewer.center.im - initial_im).abs() < 1e-9);
    assert!(viewer.request_generation > gen_before_drag);
    assert_eq!(
        viewer.latest_generation.load(Ordering::Acquire),
        viewer.request_generation
    );
    assert!(
        drag_elapsed < Duration::from_micros(16_670),
        "Drag frame stalled UI thread: {drag_elapsed:?}"
    );
}

#[test]
fn test_1920x1080_max_iterations_5000_frame_latency_under_16_67ms() {
    let ctx = egui::Context::default();
    let mut viewer = FractalViewer {
        max_iter: math_commons::registry::MAX_ITERATIONS as u32,
        ..Default::default()
    };
    assert_eq!(viewer.max_iter, 5_000);

    let _ = ctx.run(raw_input_for_size(1920.0, 1080.0, Some(2048)), |_| {});

    for frame_idx in 0..5 {
        let start = PlatformInstant::now();
        let _ = ctx.run(raw_input_for_size(1920.0, 1080.0, Some(2048)), |c| {
            viewer.show(c);
        });
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_micros(16_670),
            "Frame {frame_idx} at 1920x1080 max_iter=5000 took {elapsed:?} (>= 16.67 ms)"
        );
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::sleep(Duration::from_millis(4));
    }
}

#[test]
fn test_2560x1440_clamps_texture_dimensions_within_limit_with_debug_assertions() {
    assert!(std::hint::black_box(cfg!(debug_assertions)));
    assert_eq!(clamp_fractal_dimensions(2560, 1440, 2048), (2048, 1152));
    assert_eq!(clamp_fractal_dimensions(2560, 1440, 4096), (2048, 1152));
    assert_eq!(clamp_fractal_dimensions(2560, 1440, 1024), (1024, 576));

    for max_side in [2048_usize, 4096_usize, 1024_usize] {
        let ctx = egui::Context::default();
        let mut viewer = FractalViewer::default();
        for _ in 0..4 {
            let _ = ctx.run(raw_input_for_size(2560.0, 1440.0, Some(max_side)), |c| {
                viewer.show(c);
            });
            #[cfg(not(target_arch = "wasm32"))]
            std::thread::sleep(Duration::from_millis(5));
        }
        let tex = viewer
            .texture
            .as_ref()
            .expect("FractalViewer texture must be initialized");
        let [tw, th] = tex.size();
        let expected_cap = max_side.min(MAX_FRACTAL_TEXTURE_SIDE);
        assert!(
            tw <= expected_cap && th <= expected_cap,
            "Texture [{tw}, {th}] exceeded cap {expected_cap} for max_texture_side={max_side}"
        );
        assert!(
            viewer.requested_size.0 <= expected_cap && viewer.requested_size.1 <= expected_cap,
            "Requested worker size {:?} exceeded cap {expected_cap}",
            viewer.requested_size
        );
    }
}

#[test]
fn test_scroll_100_notches_in_and_out_clamps_zoom_and_roundtrips_state() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut viewer = FractalViewer::default();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0));

    let out0 = ctx.run(raw_input_for_size(800.0, 600.0, Some(2048)), |c| {
        viewer.show(c);
    });
    let hover_pos = find_image_rect(&out0, screen).center();

    for _ in 0..100 {
        let mut input = raw_input_for_size(800.0, 600.0, Some(2048));
        input.events.push(egui::Event::PointerMoved(hover_pos));
        input.events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, 24.0),
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(input, |c| viewer.show(c));
        assert!((ZOOM_MIN..=ZOOM_MAX).contains(&viewer.zoom));
    }
    assert!((viewer.zoom - ZOOM_MAX).abs() < 1e-6);

    let saved_in = viewer.save_state().expect("save_state at max zoom");
    let mut roundtrip = FractalViewer::default();
    roundtrip.try_load_state(&saved_in).expect("load max zoom");
    assert!((roundtrip.zoom - ZOOM_MAX).abs() < 1e-6);

    for _ in 0..150 {
        let mut input = raw_input_for_size(800.0, 600.0, Some(2048));
        input.events.push(egui::Event::PointerMoved(hover_pos));
        input.events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(0.0, -24.0),
            modifiers: egui::Modifiers::NONE,
        });
        let _ = ctx.run(input, |c| viewer.show(c));
        assert!((ZOOM_MIN..=ZOOM_MAX).contains(&viewer.zoom));
    }
    assert!((viewer.zoom - ZOOM_MIN).abs() < 1e-6);

    let saved_out = viewer.save_state().expect("save_state at min zoom");
    roundtrip.try_load_state(&saved_out).expect("load min zoom");
    assert!((roundtrip.zoom - ZOOM_MIN).abs() < 1e-6);
}

#[cfg(not(target_arch = "wasm32"))]
fn write_pass_strip_png(path: &str, panel_w: usize, panel_h: usize, frames: &[Vec<egui::Color32>]) {
    let out_w = panel_w * frames.len();
    let mut rgba = Vec::with_capacity(out_w * panel_h * 4);
    for y in 0..panel_h {
        for (idx, frame) in frames.iter().enumerate() {
            for x in 0..panel_w {
                let c = if x == 0 && idx > 0 {
                    egui::Color32::from_rgb(220, 220, 230)
                } else {
                    frame[math_commons::math_kernel::types::flatten_2d_index(x, y, panel_w)]
                };
                rgba.extend_from_slice(&[c.r(), c.g(), c.b(), 255]);
            }
        }
    }
    if let Ok(file) = std::fs::File::create(path) {
        let mut encoder = png::Encoder::new(file, out_w as u32, panel_h as u32);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        if let Ok(mut writer) = encoder.write_header() {
            let _ = writer.write_image_data(&rgba);
        }
    }
}

#[test]
fn test_generate_progressive_fractal_verification_png() {
    let (panel_w, panel_h) = (240_usize, 180_usize);
    let req = FractalRenderRequest {
        generation: 1,
        width: panel_w,
        height: panel_h,
        center: Complex::new(-0.75, 0.0),
        zoom: 1.15,
        max_iter: 200,
        mode: FractalMode::Mandelbrot,
        julia_c: Complex::new(-0.7, 0.27015),
        high_contrast: false,
    };
    let seed = FractalRenderRequest {
        generation: 0,
        width: 16,
        height: 12,
        ..req
    };
    let shared = Arc::new(RwLock::new(seed));
    let latest = Arc::new(AtomicU64::new(0));
    let mut runner = FractalRunner::new(Arc::clone(&shared), Arc::clone(&latest));
    *shared.write().unwrap() = req;
    latest.store(1, Ordering::Release);

    let mut pass_frames: Vec<Vec<egui::Color32>> = Vec::new();
    let mut last_stride = usize::MAX;
    for _ in 0..512 {
        runner.step();
        let snap = runner.get_snapshot();
        let meta = snap
            .structured_data
            .as_ref()
            .and_then(|d| d.downcast_ref::<FractalPassMeta>())
            .copied()
            .unwrap();
        if meta.stride != last_stride && (meta.is_final || runner.next_sample_y == 0) {
            last_stride = meta.stride;
            pass_frames.push(snap.pixels.read().unwrap().clone());
        }
        if meta.is_final {
            break;
        }
    }
    assert_eq!(pass_frames.len(), 4);
    #[cfg(not(target_arch = "wasm32"))]
    write_pass_strip_png(
        "/tmp/fractal_progressive_rendering.png",
        panel_w,
        panel_h,
        &pass_frames,
    );
}
