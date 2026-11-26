use super::*;
use eframe::egui::{self, vec2, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};
use math_commons::math_kernel::types::flatten_2d_index;
use wasm_bindgen_test::wasm_bindgen_test;

fn wasm_key_press(key: Key) -> Event {
    Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    }
}

fn step_wasm_canvas(
    ctx: &Context,
    events: Vec<Event>,
    state: &mut RasterBrushState,
    stamped: &mut Vec<(usize, usize)>,
) {
    let raw = RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(320.0, 200.0))),
        events,
        ..Default::default()
    };
    let _ = ctx.run(raw, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let (_, stroke) = RasterBrushCanvas::new("wasm_lbm_grid", 40, 20, "WASM LBM Grid")
                .flip_y(true)
                .show_custom(ui, state, |painter, rect, palette| {
                    painter.rect_filled(rect, 0.0, palette.surface_fill);
                });
            if let Some(s) = stroke {
                stamped.extend(s.cells);
            }
        });
    });
}

#[wasm_bindgen_test]
fn wasm_raster_brush_reticle_updates_dom_live_region_and_rle_url_fragment() {
    crate::accessibility::init_accessibility_bridge();
    let doc = web_sys::window()
        .and_then(|w| w.document())
        .expect("browser document should exist");

    let ctx = Context::default();
    let mut state = RasterBrushState::with_stencil_and_radius(BrushStencil::Airfoil, 2);
    let mut stamped = Vec::new();

    step_wasm_canvas(&ctx, vec![], &mut state, &mut stamped);
    step_wasm_canvas(
        &ctx,
        vec![wasm_key_press(Key::Tab)],
        &mut state,
        &mut stamped,
    );
    step_wasm_canvas(&ctx, vec![], &mut state, &mut stamped);
    step_wasm_canvas(
        &ctx,
        vec![wasm_key_press(Key::ArrowRight), wasm_key_press(Key::Space)],
        &mut state,
        &mut stamped,
    );

    assert!(!stamped.is_empty(), "WASM Space key should stamp airfoil");
    let slot0 = doc
        .get_element_by_id("aria-live-polite-slot-0")
        .and_then(|e| e.text_content())
        .unwrap_or_default();
    let slot1 = doc
        .get_element_by_id("aria-live-polite-slot-1")
        .and_then(|e| e.text_content())
        .unwrap_or_default();
    let combined = format!("{slot0}{slot1}");
    assert!(
        combined.contains("Airfoil"),
        "Expected DOM live region to announce Airfoil stamp, got: {combined}"
    );

    let mut grid = vec![false; 40 * 20];
    for (x, y) in stamped {
        grid[flatten_2d_index(x, y, 40)] = true;
    }
    let rle = encode_bool_grid(40, 20, &grid).expect("encode WASM airfoil RLE");
    let decoded = decode_bool_grid(&rle, 40, 20).expect("decode WASM airfoil RLE");
    assert_eq!(decoded, grid);
}
