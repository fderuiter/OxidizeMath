#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::{ElementType, PotentialFlowTool};
use crate::accessibility::init_accessibility_bridge;
use crate::framework::InteractiveTool;
use eframe::egui;
use wasm_bindgen_test::wasm_bindgen_test;

#[wasm_bindgen_test]
fn wasm_potential_flow_presets_contours_and_state_roundtrip() {
    init_accessibility_bridge();
    let mut tool = PotentialFlowTool::default();
    tool.import_preset_json(crate::starter_gallery::PRESET_LIFTING_CYLINDER)
        .expect("Lifting cylinder preset should load in browser");
    assert_eq!(tool.elements.len(), 3);

    let frames = tool.sample_animation_frames(3);
    assert_eq!(frames.len(), 3);
    assert!(frames[0].iter().any(|(name, _)| name == "Vector Field"));
    assert!(frames[0]
        .iter()
        .any(|(name, _)| name.starts_with("Dye Streak")));

    tool.selected_element_type = ElementType::Sink;
    let saved = tool.save_state().expect("state serializes");
    let mut restored = PotentialFlowTool::default();
    restored.try_load_state(&saved).expect("state restores");
    assert_eq!(restored.selected_element_type, ElementType::Sink);
}

#[wasm_bindgen_test]
fn wasm_potential_flow_live_reticle_announcements() {
    init_accessibility_bridge();
    let mut tool = PotentialFlowTool::default();
    let ctx = egui::Context::default();
    let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1024.0, 768.0));
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(rect),
            ..Default::default()
        },
        |c| tool.show(c),
    );
    let canvas_id = tool.focused_canvas_id().expect("canvas id initialized");
    ctx.memory_mut(|m| m.request_focus(canvas_id));
    let key_ev = |pressed| egui::Event::Key {
        key: egui::Key::ArrowUp,
        physical_key: None,
        pressed,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    };
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: Some(rect),
            events: vec![key_ev(true), key_ev(false)],
            ..Default::default()
        },
        |c| tool.show(c),
    );
    let doc = web_sys::window()
        .and_then(|w| w.document())
        .expect("browser document must exist");
    let text = |id| {
        doc.get_element_by_id(id)
            .and_then(|e| e.text_content())
            .unwrap_or_default()
    };
    assert!(
        text("aria-live-polite-slot-0").contains("Reticle")
            || text("aria-live-polite-slot-1").contains("Reticle")
    );
}
