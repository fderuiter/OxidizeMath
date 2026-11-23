#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::brush_codec::{
    apply_sparse_potential, bresenham_line, decode_rle_mask, encode_rle_mask,
    encode_sparse_potential, rasterize_disk_stroke,
};
use super::spatial_brush::{BrushMode, SpatialBrushKit};
use crate::accessibility::with_live_region;
use crate::app::ThemeMode;
use crate::url_state::MAX_DECODED_URL_STATE_BYTES;
use eframe::egui::{self, Event, Key, Modifiers};
use egui_plot::{contrast_ratio, focus_ring_stroke, Line, Plot, PlotPoints};

#[test]
fn test_bresenham_fast_diagonal_leaves_zero_skipped_cells() {
    let pts = bresenham_line(2, 3, 47, 39);
    assert_eq!(pts.first().copied(), Some((2, 3)));
    assert_eq!(pts.last().copied(), Some((47, 39)));
    for pair in pts.windows(2) {
        let dx = (pair[1].0 - pair[0].0).abs();
        let dy = (pair[1].1 - pair[0].1).abs();
        assert!(
            dx <= 1 && dy <= 1 && (dx + dy) >= 1,
            "Bresenham step jumped from {:?} to {:?}",
            pair[0],
            pair[1]
        );
    }

    let mut mask = vec![0_u8; 60 * 50];
    rasterize_disk_stroke((2, 3), (47, 39), 1, (60, 50), |x, y| {
        mask[y * 60 + x] = 1;
    });
    for (x, y) in pts {
        assert_eq!(mask[y as usize * 60 + x as usize], 1);
    }
}

#[test]
fn test_rle_and_sparse_codecs_roundtrip_within_url_bounds() {
    let mut grid = vec![0_u8; 100 * 100];
    for y in 20..80 {
        grid[y * 100 + 33] = 1;
        grid[y * 100 + 66] = 1;
    }
    grid[30 * 100 + 15] = 2;
    grid[70 * 100 + 15] = 3;
    grid[50 * 100 + 85] = 4;

    let runs = encode_rle_mask(&grid);
    assert!(!runs.is_empty());
    let json = serde_json::to_string(&runs).expect("serialize RLE");
    assert!(
        json.len() < MAX_DECODED_URL_STATE_BYTES,
        "RLE maze JSON {} bytes exceeded MAX_DECODED_URL_STATE_BYTES {}",
        json.len(),
        MAX_DECODED_URL_STATE_BYTES
    );
    let restored = decode_rle_mask(&runs, grid.len()).expect("decode RLE");
    assert_eq!(restored, grid);

    let base = vec![0.0_f64; 100];
    let mut sculpted = base.clone();
    for v in &mut sculpted[42..46] {
        *v = 5.5;
    }
    for v in &mut sculpted[54..58] {
        *v = 5.5;
    }
    let sparse = encode_sparse_potential(&base, &sculpted, 1e-6);
    let sparse_json = serde_json::to_string(&sparse).expect("serialize sparse V(x)");
    assert!(sparse_json.len() < MAX_DECODED_URL_STATE_BYTES);
    let mut target = base;
    apply_sparse_potential(&mut target, &sparse).expect("apply sparse");
    assert_eq!(target, sculpted);
}

#[test]
fn test_focus_ring_stroke_meets_wcag_3_to_1_across_all_themes() {
    for theme in [ThemeMode::Dark, ThemeMode::Light, ThemeMode::HighContrast] {
        let visuals = theme.visuals();
        for bg in [visuals.extreme_bg_color, visuals.panel_fill] {
            let stroke = focus_ring_stroke(&visuals, bg);
            let ratio = contrast_ratio(stroke.color, bg);
            assert!(
                ratio >= 3.0,
                "{theme:?} focus stroke contrast {ratio:.2}:1 < 3.0:1 against {bg:?}"
            );
            assert!(stroke.width >= 2.0);
        }
    }
}

fn run_brush_plot_frame(
    ctx: &egui::Context,
    raw: egui::RawInput,
    kit: &mut SpatialBrushKit,
    strokes: &mut Vec<super::spatial_brush::BrushStroke>,
) -> egui::Id {
    let mut plot_id = egui::Id::NULL;
    let _ = ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = ui.button("Before");
            let resp = Plot::new("kb_brush_test_plot")
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(
                        "ref",
                        PlotPoints::from_iter([[0.0, 0.0], [100.0, 50.0]]),
                    ));
                    strokes.extend(kit.sample_plot(plot_ui, [0.0, 0.0], [100.0, 50.0], None));
                });
            plot_id = resp.response.id;
            let _ = ui.button("After");
        });
    });
    plot_id
}

fn key_event(key: Key, modifiers: Modifiers) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers,
    }
}

#[test]
fn test_keyboard_cursor_stamps_announces_and_preserves_ctrl_z() {
    let ctx = egui::Context::default();
    let mut kit = SpatialBrushKit::new(BrushMode::Solid, 2.0)
        .with_radius_bounds(1.0, 8.0, 1.0)
        .with_cursor_step(3.0, 4.0);
    let mut strokes = Vec::new();
    let plot_id = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);
    ctx.memory_mut(|m| m.request_focus(plot_id));

    let mut raw = egui::RawInput::default();
    raw.events.push(key_event(Key::ArrowRight, Modifiers::NONE));
    raw.events
        .push(key_event(Key::CloseBracket, Modifiers::NONE));
    raw.events.push(key_event(Key::Space, Modifiers::NONE));
    let _ = run_brush_plot_frame(&ctx, raw, &mut kit, &mut strokes);

    assert_eq!(kit.cursor_pos, [53.0, 25.0]);
    assert_eq!(kit.radius, 3.0);
    assert_eq!(strokes.len(), 1);
    assert_eq!(strokes[0].end, [53.0, 25.0]);
    assert_eq!(strokes[0].radius, 3.0);

    let live_text = with_live_region(|r| r.slots[r.active_slot].clone());
    assert!(live_text.contains("Stamped Solid Wall"));

    let mut ctrl_z_remained = false;
    let mut ctrl_input = egui::RawInput {
        modifiers: Modifiers::CTRL,
        events: vec![key_event(Key::Z, Modifiers::CTRL)],
        ..Default::default()
    };
    let _ = ctx.run(std::mem::take(&mut ctrl_input), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("kb_brush_test_plot")
                .allow_drag(false)
                .show(ui, |plot_ui| {
                    let _ = kit.sample_plot(plot_ui, [0.0, 0.0], [100.0, 50.0], None);
                });
            ctrl_z_remained = ctx.input(|i| i.key_pressed(Key::Z) && i.modifiers.ctrl);
        });
    });
    assert!(ctrl_z_remained, "SpatialBrushKit must never consume Ctrl+Z");
}

#[test]
fn test_keyboard_cursor_never_traps_tab_or_shift_tab_focus() {
    let ctx = egui::Context::default();
    let mut kit = SpatialBrushKit::new(BrushMode::Solid, 2.0);
    let mut strokes = Vec::new();
    let plot_id = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);

    let tab_input = egui::RawInput {
        events: vec![key_event(Key::Tab, Modifiers::NONE)],
        ..Default::default()
    };
    let _ = run_brush_plot_frame(&ctx, tab_input, &mut kit, &mut strokes);
    assert_ne!(
        ctx.memory(|m| m.focused()),
        Some(plot_id),
        "Tab must advance focus out of the plot canvas without trapping keyboard users"
    );

    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);
    let shift_tab = egui::RawInput {
        modifiers: Modifiers::SHIFT,
        events: vec![key_event(Key::Tab, Modifiers::SHIFT)],
        ..Default::default()
    };
    let _ = run_brush_plot_frame(&ctx, shift_tab, &mut kit, &mut strokes);
    let _ = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);
    assert_ne!(
        ctx.memory(|m| m.focused()),
        Some(plot_id),
        "Shift+Tab must reverse focus out of the plot canvas without trapping keyboard users"
    );
}

#[test]
fn test_keyboard_cursor_shift_arrow_coarse_step_and_enter_stamp() {
    let ctx = egui::Context::default();
    let mut kit = SpatialBrushKit::new(BrushMode::PotentialBarrier, 0.40)
        .with_radius_bounds(0.10, 1.20, 0.10)
        .with_cursor_step(2.0, 4.0);
    let mut strokes = Vec::new();
    let plot_id = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);
    ctx.memory_mut(|m| m.request_focus(plot_id));
    let _ = run_brush_plot_frame(&ctx, egui::RawInput::default(), &mut kit, &mut strokes);

    let shift_arrow = egui::RawInput {
        modifiers: Modifiers::SHIFT,
        events: vec![
            key_event(Key::ArrowLeft, Modifiers::SHIFT),
            key_event(Key::ArrowUp, Modifiers::SHIFT),
        ],
        ..Default::default()
    };
    let _ = run_brush_plot_frame(&ctx, shift_arrow, &mut kit, &mut strokes);
    assert_eq!(ctx.memory(|m| m.focused()), Some(plot_id));
    assert_eq!(kit.cursor_pos, [44.0, 37.0]);

    let stamp_frame = egui::RawInput {
        modifiers: Modifiers::NONE,
        events: vec![
            key_event(Key::OpenBracket, Modifiers::NONE),
            key_event(Key::Enter, Modifiers::NONE),
        ],
        ..Default::default()
    };
    let _ = run_brush_plot_frame(&ctx, stamp_frame, &mut kit, &mut strokes);

    assert!((kit.radius - 0.30).abs() < 1e-9);
    assert_eq!(strokes.len(), 1);
    assert_eq!(strokes[0].end, [44.0, 37.0]);
}
