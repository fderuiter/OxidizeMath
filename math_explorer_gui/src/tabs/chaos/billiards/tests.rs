//! Unit, accessibility, and export tests for [`DynamicalBilliardsTool`].

#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::preset::{
    PRESET_BUNIMOVICH_STADIUM, PRESET_ELLIPTICAL_CAUSTICS, PRESET_SINAI_DISPERSING,
};
use super::{BilliardGeometryKind, DynamicalBilliardsTool, MAX_BIRKHOFF_POINTS, MAX_RAYS};
use crate::framework::InteractiveTool;
use eframe::egui::{self, Context, Event, Key, Modifiers, Pos2, RawInput, Rect};

fn raw_viewport() -> RawInput {
    RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, egui::vec2(1280.0, 800.0))),
        ..Default::default()
    }
}

fn push_key_event(key: Key) -> RawInput {
    let mut raw = raw_viewport();
    raw.events.push(Event::Key {
        key,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: Modifiers::NONE,
    });
    raw
}

#[test]
fn test_billiards_canvas_keyboard_focus_and_out_of_domain_rejection() {
    let mut tool = DynamicalBilliardsTool::default();
    tool.switch_geometry(BilliardGeometryKind::Sinai);
    tool.clear_rays();

    let ctx = Context::default();
    ctx.enable_accesskit();
    let _ = ctx.run(raw_viewport(), |ctx| tool.show(ctx));

    let mut focused = false;
    for _ in 0..60 {
        let _ = ctx.run(push_key_event(Key::Tab), |ctx| tool.show(ctx));
        if tool.focused_canvas_id().is_some() {
            focused = true;
            break;
        }
    }
    assert!(
        focused,
        "Euclidean Billiard Table canvas must receive Tab focus"
    );

    let _ = ctx.run(push_key_event(Key::Enter), |ctx| tool.show(ctx));
    assert_eq!(tool.rays.len(), 1, "Enter must launch a billiard ray");

    // Launching inside the Sinai central scatterer (0, 0) must be rejected cleanly
    // without mutating existing rays and must emit an inline rejection notice.
    let prev_bounces = tool.rays[0].bounce_count;
    tool.launch_and_announce(&ctx, [0.0, 0.0]);
    assert_eq!(tool.rays.len(), 1);
    assert_eq!(tool.rays[0].bounce_count, prev_bounces);
    assert!(
        tool.last_status_note
            .as_deref()
            .unwrap_or("")
            .starts_with("Cannot launch"),
        "Expected inline rejection notice, got {:?}",
        tool.last_status_note
    );
}

#[test]
fn test_billiards_presets_exports_and_state_bounds() {
    let mut tool = DynamicalBilliardsTool::default();
    for preset in [
        PRESET_ELLIPTICAL_CAUSTICS,
        PRESET_BUNIMOVICH_STADIUM,
        PRESET_SINAI_DISPERSING,
    ] {
        tool.import_preset_json(preset)
            .expect("billiard starter preset must load");
        tool.advance_bounces(200);
        let datasets = tool.export_datasets();
        assert!(!datasets.is_empty());
        let csv_text = egui_plot::export::format_csv(&datasets);
        assert!(csv_text.contains("Birkhoff Ray 1"));

        let prims = tool.export_plot_primitives();
        let svg = egui_plot::export_primitives_svg(&prims, &egui_plot::ExportOptions::default())
            .expect("billiard SVG export");
        assert!(svg.contains("<svg") && svg.contains("</svg>"));
    }

    let saved = tool.save_state().expect("billiards save_state");
    assert!(saved.len() <= crate::url_state::MAX_DECODED_URL_STATE_BYTES);
    tool.state_bounds()
        .validate_json(&saved)
        .expect("ToolStateBounds must accept billiard snapshot");

    for _ in 0..24 {
        let _ = tool.seed_single_ray_at([0.75, 0.15]);
    }
    assert_eq!(tool.rays.len(), MAX_RAYS);
    tool.rays[0]
        .birkhoff_points
        .resize(MAX_BIRKHOFF_POINTS, [0.5, 0.0]);
    tool.advance_bounces(50);
    assert!(tool.rays[0].birkhoff_points.len() <= MAX_BIRKHOFF_POINTS);
}
