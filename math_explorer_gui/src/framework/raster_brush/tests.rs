use super::*;
use crate::framework::state_bounds::ToolStateBounds;
use crate::url_state::{decode_base64url, encode_base64url, URL_JSON_BOUNDS};
use eframe::egui::{
    self, vec2, Context, Event, Key, Modifiers, PointerButton, Pos2, RawInput, Rect,
};

#[test]
fn test_floor_cell_mapping_eliminates_half_cell_round_offset() {
    let rect = Rect::from_min_size(Pos2::new(10.0, 20.0), vec2(200.0, 100.0));
    // 100x50 grid -> each cell is 2.0 x 2.0 screen pixels.
    // Near right/bottom edge of cell (0, 0) at local (1.9, 1.9): round() would yield (1, 1), floor() yields (0, 0).
    let near_edge = Pos2::new(11.9, 21.9);
    assert_eq!(
        map_pointer_to_cell(rect, near_edge, 100, 50, false),
        Some((0, 0))
    );
    assert_eq!(
        map_pointer_to_cell(rect, near_edge, 100, 50, true),
        Some((0, 49))
    );

    // Just inside cell (1, 1) at local (2.05, 2.05):
    let next_cell = Pos2::new(12.05, 22.05);
    assert_eq!(
        map_pointer_to_cell(rect, next_cell, 100, 50, false),
        Some((1, 1))
    );
    assert_eq!(
        map_pointer_to_cell(rect, next_cell, 100, 50, true),
        Some((1, 48))
    );

    // Outside rect returns None:
    assert_eq!(
        map_pointer_to_cell(rect, Pos2::new(9.9, 25.0), 100, 50, false),
        None
    );
}

#[test]
fn test_bresenham_line_is_gapless_across_fast_diagonal_and_steep_drags() {
    for ((x0, y0), (x1, y1)) in [
        ((0, 0), (99, 49)),
        ((99, 49), (0, 0)),
        ((12, 3), (15, 88)),
        ((50, 50), (50, 50)),
    ] {
        let pts = bresenham_line(x0, y0, x1, y1);
        assert_eq!(pts.first().copied(), Some((x0, y0)));
        assert_eq!(pts.last().copied(), Some((x1, y1)));
        for pair in pts.windows(2) {
            let dx = (pair[1].0 - pair[0].0).abs();
            let dy = (pair[1].1 - pair[0].1).abs();
            assert!(dx <= 1 && dy <= 1 && (dx + dy) >= 1);
        }
    }
}

#[test]
fn test_stencils_and_symmetry_modes_produce_valid_distinct_cells() {
    let airfoil = generate_brush_cells(
        BrushStencil::Airfoil,
        SymmetryMode::None,
        (30, 25),
        (30, 25),
        3,
        100,
        50,
    );
    assert!(airfoil.len() > 25);
    assert!(airfoil.contains(&(30, 25)));

    let venturi = generate_brush_cells(
        BrushStencil::VenturiNozzle,
        SymmetryMode::None,
        (50, 25),
        (50, 25),
        2,
        100,
        50,
    );
    assert!(venturi.len() > 30);
    // Center of throat remains open fluid!
    assert!(!venturi.contains(&(50, 25)));

    let bilateral_x = generate_brush_cells(
        BrushStencil::Circle,
        SymmetryMode::BilateralX,
        (10, 20),
        (10, 20),
        1,
        100,
        100,
    );
    assert!(bilateral_x.contains(&(10, 20)));
    assert!(bilateral_x.contains(&(89, 20)));

    let radial = generate_brush_cells(
        BrushStencil::Circle,
        SymmetryMode::Radial,
        (20, 30),
        (20, 30),
        0,
        100,
        100,
    );
    assert_eq!(radial.len(), 4);
    assert!(radial.contains(&(20, 30)));
    assert!(radial.contains(&(79, 69)));
}

#[test]
fn test_rle_codec_roundtrip_and_bounds_compliance() {
    let mut mask = vec![false; 100 * 50];
    for (x, y) in generate_brush_cells(
        BrushStencil::Airfoil,
        SymmetryMode::None,
        (35, 25),
        (35, 25),
        3,
        100,
        50,
    ) {
        mask[y * 100 + x] = true;
    }
    let encoded = encode_bool_grid(100, 50, &mask).expect("encode airfoil RLE");
    let json = serde_json::json!({ "viscosity": 0.02, "obstacles_rle": encoded }).to_string();
    ToolStateBounds::DEFAULT
        .validate_json(&json)
        .expect("fits ToolStateBounds");
    URL_JSON_BOUNDS
        .validate_json(&json)
        .expect("fits URL_JSON_BOUNDS");

    let decoded = decode_bool_grid(&encoded, 100, 50).expect("decode airfoil RLE");
    assert_eq!(decoded, mask);
}

#[test]
fn test_rle_rejects_malformed_oversized_and_mismatched_before_grid_alloc() {
    // 1. Header claiming 500x500 when caller expects 10x10 must reject prior to grid allocation:
    let mut forged = vec![b'R', b'M', 1];
    forged.extend_from_slice(&500u16.to_le_bytes());
    forged.extend_from_slice(&500u16.to_le_bytes());
    forged.extend_from_slice(&[1, 0xFF, 0xFF, 0x0F]);
    let forged_b64 = encode_base64url(&forged);

    // Pre-decode bytes outside run_verified so we strictly verify verify_rle_stream_without_grid_alloc allocates 0 bytes!
    let raw_bytes = decode_base64url(&forged_b64, MAX_RLE_ENCODED_LEN).unwrap();
    let res = {
        let _alloc_guard = crate::VERIFIED_ALLOC_TEST_MUTEX
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (res, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            // Any Vec allocation inside verification would trigger VerifiedAllocator panic!
            super::rle::verify_rle_stream_without_grid_alloc(&raw_bytes, 10, 10, 100, 1)
        });
        res
    };
    assert!(res.is_err());
    assert!(decode_bool_grid(&forged_b64, 10, 10).is_err());
    assert!(decode_bool_grid(&forged_b64, 500, 500).is_err());
    assert!(!raw_bytes.is_empty());

    // 2. Oversized string > MAX_RLE_ENCODED_LEN rejected immediately:
    let huge = "A".repeat(MAX_RLE_ENCODED_LEN + 10);
    assert!(decode_u8_grid(&huge, 10, 10, 2).is_err());

    // 3. Out-of-range symbol rejected:
    let valid_u8 = encode_u8_grid(4, 4, &[2u8; 16], 2).unwrap();
    assert!(decode_u8_grid(&valid_u8, 4, 4, 1).is_err());
}

fn key_press(key: Key) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    }
}

fn step_keyboard_canvas(
    ctx: &Context,
    events: Vec<Event>,
    state: &mut RasterBrushState,
    strokes: &mut Vec<RasterBrushStroke>,
) {
    let raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(400.0, 400.0))),
        events,
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = ui.button("Before");
            let (_, stroke) = RasterBrushCanvas::new("test_grid", 20, 20, "Test Raster Grid")
                .role(egui::accesskit::Role::GraphicsDocument)
                .show_custom(ui, state, |painter, rect, palette| {
                    painter.rect_filled(rect, 0.0, palette.surface_fill);
                });
            if let Some(s) = stroke {
                strokes.push(s);
            }
            let _ = ui.button("After");
        });
    });
}

#[test]
fn test_raster_brush_canvas_keyboard_reticle_space_and_tab_traversal() {
    let ctx = Context::default();
    ctx.enable_accesskit();
    let mut state = RasterBrushState::default();
    let mut strokes = Vec::new();

    step_keyboard_canvas(&ctx, vec![], &mut state, &mut strokes);
    step_keyboard_canvas(&ctx, vec![key_press(Key::Tab)], &mut state, &mut strokes);
    step_keyboard_canvas(&ctx, vec![key_press(Key::Tab)], &mut state, &mut strokes);
    step_keyboard_canvas(&ctx, vec![], &mut state, &mut strokes);
    assert!(state.focused_id.is_some(), "Canvas must receive Tab focus");
    assert_eq!(state.reticle, (10, 10));

    step_keyboard_canvas(
        &ctx,
        vec![key_press(Key::ArrowRight), key_press(Key::ArrowDown)],
        &mut state,
        &mut strokes,
    );
    assert_eq!(state.reticle, (11, 11));

    step_keyboard_canvas(&ctx, vec![key_press(Key::Space)], &mut state, &mut strokes);
    assert_eq!(strokes.len(), 1);
    assert!(strokes[0].via_keyboard);
    assert_eq!(strokes[0].focus_cell, (11, 11));
    assert!(strokes[0].cells.contains(&(11, 11)));

    let live_msg = crate::accessibility::with_live_region(|r| r.slots[r.active_slot].clone());
    assert!(live_msg.contains("(11, 11)"));

    step_keyboard_canvas(&ctx, vec![key_press(Key::Tab)], &mut state, &mut strokes);
    assert!(
        state.focused_id.is_none(),
        "Tab must advance focus past canvas without trapping"
    );
}

#[test]
fn test_raster_brush_canvas_continuous_pointer_drag_bresenham() {
    let ctx = Context::default();
    let mut state = RasterBrushState::with_radius(0);
    let mut touched = Vec::new();

    let mut step_ui = |events: Vec<Event>| -> Rect {
        let mut rect_out = Rect::NOTHING;
        let raw = RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(200.0, 200.0))),
            events,
            ..Default::default()
        };
        let _ = ctx.run(raw, |ctx| {
            egui::CentralPanel::default()
                .frame(egui::Frame::NONE)
                .show(ctx, |ui| {
                    let (resp, stroke) = RasterBrushCanvas::new("drag_grid", 20, 20, "Drag Grid")
                        .show_custom(ui, &mut state, |_painter, rect, _palette| {
                            rect_out = rect;
                        });
                    let _ = resp;
                    if let Some(s) = stroke {
                        touched.extend(s.cells);
                    }
                });
        });
        rect_out
    };

    let canvas_rect = step_ui(vec![]);
    let p0 = canvas_rect.min + vec2(5.0, 5.0); // cell (0, 0)
    let p1 = canvas_rect.min + vec2(155.0, 75.0); // cell (15, 7) — fast jump!
    let _ = step_ui(vec![
        Event::PointerMoved(p0),
        Event::PointerButton {
            pos: p0,
            button: PointerButton::Primary,
            pressed: true,
            modifiers: Modifiers::NONE,
        },
    ]);
    let _ = step_ui(vec![Event::PointerMoved(p1)]);
    let _ = step_ui(vec![Event::PointerButton {
        pos: p1,
        button: PointerButton::Primary,
        pressed: false,
        modifiers: Modifiers::NONE,
    }]);

    for expected in bresenham_line(0, 0, 15, 7) {
        assert!(
            touched.contains(&(expected.0 as usize, expected.1 as usize)),
            "Fast drag missed cell {expected:?}"
        );
    }
}
