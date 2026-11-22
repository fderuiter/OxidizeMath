#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use crate::framework::InteractiveTool;
use crate::url_state::{encode_share_fragment, parse_share_location, validate_url_preset_json};
use eframe::egui;

fn frame_input(events: Vec<egui::Event>) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(1024.0, 768.0),
        )),
        max_texture_side: Some(2048),
        focused: true,
        events,
        ..Default::default()
    }
}

fn key_event(key: egui::Key) -> Vec<egui::Event> {
    vec![egui::Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }]
}

fn find_canvas_rect(out: &egui::FullOutput) -> egui::Rect {
    let ak = out.platform_output.accesskit_update.as_ref().unwrap();
    for (_, node) in &ak.nodes {
        if node.role() == egui::accesskit::Role::Image {
            if let Some(b) = node.bounds() {
                return egui::Rect::from_min_max(
                    egui::pos2(b.x0 as f32, b.y0 as f32),
                    egui::pos2(b.x1 as f32, b.y1 as f32),
                );
            }
        }
    }
    panic!("FractalViewer Role::Image node not found");
}

#[test]
fn test_mandelbrot_hover_click_arrow_enter_and_tab_exit_updates_inset() {
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut viewer = FractalViewer::default();

    let out = ctx.run(frame_input(Vec::new()), |c| viewer.show(c));
    let rect = find_canvas_rect(&out);
    assert_eq!(
        viewer.julia_inset_image.as_ref().map(|i| i.size),
        Some([160, 120])
    );

    let hover_pt = egui::pos2(rect.left() + rect.width() * 0.62, rect.center().y - 30.0);
    let _ = ctx.run(
        frame_input(vec![egui::Event::PointerMoved(hover_pt)]),
        |c| viewer.show(c),
    );
    assert!(!viewer.c_pinned);
    let expected_c = screen_to_complex(hover_pt, rect, viewer.center, viewer.zoom);
    assert!((viewer.julia_c - expected_c).norm() < 1e-9);

    let click_pt = complex_to_screen(Complex::new(0.24, 0.0), rect, viewer.center, viewer.zoom);
    let _ = ctx.run(
        frame_input(vec![
            egui::Event::PointerMoved(click_pt),
            egui::Event::PointerButton {
                pos: click_pt,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ]),
        |c| viewer.show(c),
    );
    let _ = ctx.run(
        frame_input(vec![egui::Event::PointerButton {
            pos: click_pt,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }]),
        |c| viewer.show(c),
    );
    assert!(viewer.c_pinned);
    assert!((viewer.julia_c.re - 0.24).abs() < 1e-2);

    let pinned_re = viewer.julia_c.re;
    for _ in 0..3 {
        let _ = ctx.run(frame_input(key_event(egui::Key::ArrowRight)), |c| {
            viewer.show(c);
        });
    }
    assert!((viewer.julia_c.re - (pinned_re + 0.03)).abs() < 1e-6);

    let _ = ctx.run(frame_input(key_event(egui::Key::Enter)), |c| viewer.show(c));
    assert!(!viewer.c_pinned);

    let pre_focus = ctx.memory(|m| m.focused());
    assert!(pre_focus.is_some());
    let _ = ctx.run(frame_input(key_event(egui::Key::Tab)), |c| viewer.show(c));
    assert_ne!(ctx.memory(|m| m.focused()), pre_focus);
}

#[test]
fn test_mandelbrot_cusp_transition_0_24_connected_vs_0_27_cantor_dust() {
    let ctx = egui::Context::default();
    let mut viewer = FractalViewer::default();
    viewer
        .import_preset_json(STARTER_CUSP_CONNECTED_JSON)
        .unwrap();
    let _ = ctx.run(frame_input(Vec::new()), |c| viewer.show(c));

    let orbit_connected = compute_critical_orbit(viewer.julia_c, CRITICAL_ORBIT_STEPS);
    assert_eq!(orbit_connected.len(), CRITICAL_ORBIT_STEPS + 1);
    assert!(orbit_connected.last().unwrap().norm() < 2.0);
    let connected_img = viewer.julia_inset_image.clone().expect("160x120 inset");
    assert_eq!(connected_img.size, [160, 120]);
    let connected_interior = connected_img
        .pixels
        .iter()
        .filter(|&&px| px == egui::Color32::BLACK)
        .count();
    assert!(connected_interior > 1500);

    viewer.import_preset_json(STARTER_CUSP_DUST_JSON).unwrap();
    let _ = ctx.run(frame_input(Vec::new()), |c| viewer.show(c));
    let orbit_dust = compute_critical_orbit(viewer.julia_c, CRITICAL_ORBIT_STEPS);
    assert!(orbit_dust.len() < CRITICAL_ORBIT_STEPS + 1);
    let dust_img = viewer.julia_inset_image.clone().expect("160x120 inset");
    let dust_interior = dust_img
        .pixels
        .iter()
        .filter(|&&px| px == egui::Color32::BLACK)
        .count();
    assert!(
        dust_interior * 5 < connected_interior,
        "Expected Cantor dust J_c at c=0.27 ({dust_interior}) << connected J_c at c=0.24 ({connected_interior})"
    );
}

#[test]
fn test_mode_toggle_preserves_independent_mandelbrot_and_julia_viewports() {
    let mut viewer = FractalViewer {
        center: Complex::new(0.25, 0.01),
        zoom: 6.5,
        ..Default::default()
    };

    viewer.switch_mode(FractalMode::Julia);
    assert_eq!(viewer.mode, FractalMode::Julia);
    assert!((viewer.center - DEFAULT_JULIA_CENTER).norm() < 1e-12);
    assert!((viewer.zoom - 1.0).abs() < 1e-12);

    viewer.center = Complex::new(-0.2, 0.3);
    viewer.zoom = 2.75;

    viewer.switch_mode(FractalMode::Mandelbrot);
    assert_eq!(viewer.mode, FractalMode::Mandelbrot);
    assert!((viewer.center - Complex::new(0.25, 0.01)).norm() < 1e-12);
    assert!((viewer.zoom - 6.5).abs() < 1e-12);

    let saved = viewer.save_state().unwrap();
    let mut restored = FractalViewer::default();
    restored.try_load_state(&saved).unwrap();
    restored.switch_mode(FractalMode::Julia);
    assert!((restored.center - Complex::new(-0.2, 0.3)).norm() < 1e-12);
    assert!((restored.zoom - 2.75).abs() < 1e-12);
}

#[test]
fn test_fractal_legacy_and_cusp_presets_url_roundtrip_within_bounds() {
    for raw_preset in [
        STARTER_PRESET_JSON,
        STARTER_CUSP_CONNECTED_JSON,
        STARTER_CUSP_DUST_JSON,
    ] {
        validate_url_preset_json(raw_preset).unwrap();
        let mut viewer = FractalViewer::default();
        viewer.import_preset_json(raw_preset).unwrap();
        let envelope = viewer.export_preset_json().unwrap();
        validate_url_preset_json(&envelope).unwrap();

        let frag = encode_share_fragment("chaos", Some("FractalViewer"), Some(&envelope)).unwrap();
        let parsed = parse_share_location(&frag, "").unwrap().unwrap();
        let mut restored = FractalViewer::default();
        restored
            .import_preset_json(parsed.state_json.as_deref().unwrap())
            .unwrap();
        assert_eq!(restored.save_state(), viewer.save_state());
    }
}
