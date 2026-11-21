#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CountingWidgetText<'a>(&'a str, &'a AtomicUsize);

impl From<CountingWidgetText<'_>> for WidgetText {
    fn from(value: CountingWidgetText<'_>) -> Self {
        value.1.fetch_add(1, Ordering::SeqCst);
        WidgetText::from(value.0)
    }
}

#[test]
fn test_twin_slot_alternation() {
    let polite_toggle = std::sync::atomic::AtomicBool::new(false);
    let assertive_toggle = std::sync::atomic::AtomicBool::new(false);

    assert_eq!(next_live_slot(&polite_toggle), 0);
    assert_eq!(next_live_slot(&polite_toggle), 1);
    assert_eq!(next_live_slot(&polite_toggle), 0);
    assert_eq!(next_live_slot(&polite_toggle), 1);

    assert_eq!(next_live_slot(&assertive_toggle), 0);
    assert_eq!(next_live_slot(&assertive_toggle), 1);
    assert_eq!(next_live_slot(&assertive_toggle), 0);

    let mut region = TwinSlotLiveRegion::new();
    assert_eq!(region.active_slot, 0);
    assert_eq!(region.dispatch("First"), 1);
    assert_eq!(region.slots, ["".to_string(), "First".to_string()]);
    assert_eq!(region.dispatch("First"), 0);
    assert_eq!(region.slots, ["First".to_string(), "".to_string()]);
}

#[test]
fn test_announcement_dispatch_on_native() {
    clear_test_announcements();
    assert!(all_announcements().is_empty());
    assert_eq!(with_live_region(|r| r.active_slot), 0);

    announce_status("Simulation paused");
    assert_eq!(with_live_region(|r| r.active_slot), 1);

    announce_status_with_priority("Syntax Error: missing operand", "assertive");
    assert_eq!(with_live_region(|r| r.active_slot), 0);

    let announcements = all_announcements();
    assert_eq!(announcements.len(), 2);
    assert_eq!(
        announcements[0],
        ("Simulation paused".to_string(), "polite".to_string())
    );
    assert_eq!(
        announcements[1],
        (
            "Syntax Error: missing operand".to_string(),
            "assertive".to_string()
        )
    );

    clear_test_announcements();
    assert!(all_announcements().is_empty());
    assert_eq!(with_live_region(|r| r.active_slot), 0);
}

#[test]
fn test_accessible_hover_text_zero_conversion_on_inactive_frame() {
    let ctx = egui::Context::default();
    let conversions = AtomicUsize::new(0);
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(800.0, 600.0),
    ));
    clear_test_announcements();

    let _ = ctx.run(
        egui::RawInput {
            screen_rect,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.button("Action")
                    .accessible_hover_text(CountingWidgetText("Helpful guidance", &conversions));
            });
        },
    );
    assert_eq!(
        conversions.load(Ordering::SeqCst),
        0,
        "Standard inactive frame updates must not convert or heap-allocate tooltip text"
    );
    assert!(all_announcements().is_empty());
}

fn run_action_frame(
    ctx: &egui::Context,
    focused: bool,
    events: Vec<egui::Event>,
) -> (egui::Id, bool) {
    let mut out = (egui::Id::NULL, false);
    let input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::vec2(800.0, 600.0),
        )),
        focused,
        events,
        ..Default::default()
    };
    let _ = ctx.run(input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = ui
                .button("Action")
                .accessible_hover_text("Helpful guidance");
            out = (resp.id, resp.has_focus());
        });
    });
    out
}

#[test]
fn test_accessible_hover_text_focus_escape_preserves_focus() {
    let ctx = egui::Context::default();
    clear_test_announcements();

    let (btn_id, _) = run_action_frame(&ctx, false, Vec::new());
    ctx.memory_mut(|m| m.request_focus(btn_id));
    let (_, focused_on_tab) = run_action_frame(&ctx, true, Vec::new());
    assert!(focused_on_tab);
    assert!(all_announcements()
        .iter()
        .any(|(msg, prio)| msg == "Helpful guidance" && prio == "polite"));

    clear_test_announcements();
    let esc = vec![egui::Event::Key {
        key: egui::Key::Escape,
        physical_key: None,
        pressed: true,
        repeat: false,
        modifiers: egui::Modifiers::NONE,
    }];
    let (_, focused_after_escape) = run_action_frame(&ctx, true, esc);
    assert!(
        focused_after_escape,
        "Escape dismissal must preserve keyboard focus"
    );
    assert!(ctx
        .data(|d| d.get_temp::<bool>(btn_id.with("tooltip_dismissed")))
        .unwrap_or(false));
    assert!(all_announcements().is_empty());
}

#[cfg(target_arch = "wasm32")]
#[path = "accessibility_wasm_tests.rs"]
mod wasm_tests;
