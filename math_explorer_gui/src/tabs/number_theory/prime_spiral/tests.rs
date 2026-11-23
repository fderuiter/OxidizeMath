//! Integration, buffer reuse, keyboard navigation, URL round-trip, and sonification tests
//! for [`super::PrimeSpiralWidget`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;

use super::playhead::{eval_ulam_quadratic, UlamPlayheadMode};
use super::PrimeSpiralWidget;
use crate::framework::InteractiveTool;
use crate::tabs::number_theory::NumberTheoryTab;
use crate::tabs::ExplorerTab;

#[test]
fn test_prime_spiral_widget_buffer_reuse() {
    let ctx = egui::Context::default();
    let mut widget = PrimeSpiralWidget::default();

    assert_eq!(widget.grid_size, 200);
    assert!(widget.texture.is_none());
    assert!(widget.pixels_buffer.is_empty());
    assert!(widget.is_prime_lookup.is_empty());

    widget.regenerate_texture(&ctx);
    assert!(widget.texture.is_some());
    assert_eq!(widget.pixels_buffer.len(), 200 * 200);

    let initial_pixels_cap = widget.pixels_buffer.capacity();
    let initial_prime_cap = widget.is_prime_lookup.capacity();
    assert!(initial_pixels_cap >= 200 * 200);
    assert!(initial_prime_cap > 200 * 200);

    widget.grid_size = 300;
    widget.regenerate_texture(&ctx);
    assert_eq!(widget.pixels_buffer.len(), 300 * 300);

    let larger_pixels_cap = widget.pixels_buffer.capacity();
    let larger_prime_cap = widget.is_prime_lookup.capacity();
    assert!(larger_pixels_cap >= 300 * 300);

    widget.grid_size = 200;
    widget.regenerate_texture(&ctx);
    assert_eq!(widget.pixels_buffer.len(), 200 * 200);
    assert!(widget.pixels_buffer.capacity() >= larger_pixels_cap);
    assert!(widget.is_prime_lookup.capacity() >= larger_prime_cap);

    let center = 100;
    let center_idx = flatten_2d_index(center, center, 200);
    assert_eq!(
        widget.pixels_buffer[center_idx],
        crate::app::ThemeMode::Dark.palette().axis_x
    );
}

#[test]
fn test_ulam_playhead_euler_vs_composite_sonification_and_url_roundtrip() {
    assert_eq!(eval_ulam_quadratic(0, 2, 41), 41);
    assert_eq!(eval_ulam_quadratic(1, 2, 41), 47);
    assert_eq!(eval_ulam_quadratic(2, 2, 41), 61);

    let mut euler = PrimeSpiralWidget::default();
    let euler_frame = euler.current_sonification_frame();
    assert!(euler_frame.is_audible());
    assert!(euler_frame.active_voices >= 4);
    assert!(euler_frame.voices[0].transient_impulse > 0.5);

    euler.ray_b = 4;
    euler.ray_c = 4;
    let comp_frame = euler.current_sonification_frame();
    assert_eq!(comp_frame.active_voices, 1);
    assert!(euler_frame.master_gain > comp_frame.master_gain * 2.0);

    let mut tab = NumberTheoryTab::default();
    assert!(tab.select_tool_by_name("PrimeSpiralWidget"));
    if let Some(fw) = tab.framework_mut() {
        let tool = fw
            .active_tool
            .as_mut()
            .and_then(|t| t.as_any_mut())
            .and_then(|a| a.downcast_mut::<PrimeSpiralWidget>())
            .unwrap();
        tool.playhead_mode = UlamPlayheadMode::SpiralSweep;
        tool.ray_b = -2;
        tool.ray_c = 43;
        tool.cursor_step = 18;
        tool.tempo_bpm = 168.0;
    }

    let url = tab.framework().unwrap().encode_share_url().unwrap();
    let hash_pos = url.find('#').unwrap();
    let params = crate::url_state::parse_share_location(&url[hash_pos..], "")
        .unwrap()
        .unwrap();
    let mut restored_tab = NumberTheoryTab::default();
    restored_tab
        .import_url_preset_json(params.tool.as_deref(), params.state_json.as_deref())
        .unwrap();
    let restored = restored_tab
        .framework()
        .and_then(|fw| fw.active_tool.as_ref())
        .and_then(|t| t.as_any())
        .and_then(|a| a.downcast_ref::<PrimeSpiralWidget>())
        .unwrap();
    assert_eq!(restored.playhead_mode, UlamPlayheadMode::SpiralSweep);
    assert_eq!(restored.ray_b, -2);
    assert_eq!(restored.ray_c, 43);
    assert_eq!(restored.cursor_step, 18);
    assert!((restored.tempo_bpm - 168.0).abs() < 1e-9);
}

fn key_input(screen: Option<egui::Rect>, keys: &[egui::Key]) -> egui::RawInput {
    let mut raw = egui::RawInput {
        screen_rect: screen,
        ..Default::default()
    };
    for &key in keys {
        raw.events.push(egui::Event::Key {
            key,
            physical_key: Some(key),
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        });
    }
    raw
}

#[test]
fn test_ulam_keyboard_navigation_and_role_meter_presence() {
    let mut tool = PrimeSpiralWidget::default();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();

    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(1280.0, 800.0),
    ));
    let out = ctx.run(key_input(screen, &[]), |ctx| tool.show(ctx));
    let ak = out
        .platform_output
        .accesskit_update
        .expect("AccessKit tree");
    assert!(ak
        .nodes
        .iter()
        .any(|(_, n)| n.role() == egui::accesskit::Role::Meter));

    let canvas_id = ak
        .nodes
        .iter()
        .find_map(|(id, n)| {
            (n.role() == egui::accesskit::Role::GraphicsDocument
                && n.label().is_some_and(|l| l.contains("Ulam Prime Spiral")))
            .then_some(*id)
        })
        .expect("Ulam Prime Spiral GraphicsDocument node present");

    let _ = ctx.run(key_input(screen, &[egui::Key::Period]), |ctx| {
        tool.show(ctx);
    });
    assert_eq!(tool.cursor_step, 1);

    let reached = (0..80).any(|_| {
        ctx.run(key_input(screen, &[egui::Key::Tab]), |ctx| tool.show(ctx))
            .platform_output
            .accesskit_update
            .is_some_and(|u| u.focus == canvas_id)
    });
    assert!(reached, "Ulam canvas must be reachable via Tab");
    assert!(tool.focused_canvas_id().is_some());

    let _ = ctx.run(key_input(screen, &[egui::Key::ArrowRight]), |ctx| {
        tool.show(ctx);
    });
    assert_eq!(tool.cursor_step, 2);

    let _ = ctx.run(key_input(screen, &[egui::Key::ArrowUp]), |ctx| {
        tool.show(ctx);
    });
    assert_eq!(tool.ray_c, 42);

    let _ = ctx.run(key_input(screen, &[egui::Key::Tab]), |ctx| tool.show(ctx));
    assert!(tool.focused_canvas_id().is_none());
    crate::audio::export_current_tool_wav("prime_spiral");
}
