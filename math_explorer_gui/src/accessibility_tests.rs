#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;
use eframe::egui::{self, WidgetText};
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
    super::live_region::reset_startup_ready_flag();
    announce_startup_ready_once();
    announce_startup_ready_once();
    assert_eq!(
        all_announcements(),
        vec![(READY_ANNOUNCEMENT.to_string(), "polite".to_string())]
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

#[test]
fn test_speech_narration_opt_in_volume_and_cancel() {
    super::speech::reset_test_speech_state();
    assert!(!is_speech_enabled());
    speak_if_enabled("Silent while opt-in is false");
    assert!(super::speech::test_speech_log().0.is_empty());

    sync_speech_settings(true, 0.65);
    assert!(is_speech_enabled());
    assert!((speech_volume() - 0.65).abs() < 1e-6);

    speak_if_enabled("Active narration");
    let (utterances, cancels) = super::speech::test_speech_log();
    assert_eq!(utterances.len(), 1);
    assert_eq!(utterances[0].0, "Active narration");
    assert!((utterances[0].1 - 0.65).abs() < 1e-6);
    assert_eq!(cancels, 0);

    stop_speech();
    assert_eq!(super::speech::test_speech_log().1, 1);
    assert!(super::speech::take_frame_speech_suppression());
    assert!(!super::speech::take_frame_speech_suppression());

    sync_speech_settings(false, 0.65);
    assert!(!is_speech_enabled());
    assert_eq!(super::speech::test_speech_log().1, 2);
}

fn summarize_projected_store(
    store: &AccessKitTreeStore,
    screen: egui::Rect,
) -> (bool, bool, bool, bool) {
    let (mut landmark, mut heading, mut control, mut vis) = (false, false, false, false);
    for nid in store.node_ids() {
        let Some(spec) = store.build_dom_spec(nid, screen, 1.0) else {
            continue;
        };
        landmark |= matches!(spec.role, Some("region" | "navigation" | "main"));
        heading |= spec.role == Some("heading");
        control |= matches!(spec.tag, "button" | "input" | "textarea")
            || matches!(
                spec.role,
                Some("checkbox" | "radio" | "separator" | "slider" | "spinbutton")
            );
        vis |= matches!(spec.role, Some("graphics-document" | "img")) || spec.tag == "figure";
    }
    (landmark, heading, control, vis)
}

#[test]
fn test_accesskit_tree_store_projects_all_tabs_and_tools_landmarks_headings_and_controls() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    let ctx = egui::Context::default();
    ctx.enable_accesskit();
    let mut app = crate::app::MathExplorerApp::default();
    let mut frame = eframe::Frame::_new_kittest();
    let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(1280.0, 900.0));
    let mut store = AccessKitTreeStore::new();
    let mut total_visualizations = 0usize;

    for tab_idx in 0..app.tabs.len() {
        app.selected_tab = tab_idx;
        if !app.pane_tabs.is_empty() {
            app.pane_tabs[0] = tab_idx;
        }
        let tool_ids: Vec<&'static str> = app.tabs[tab_idx]
            .framework()
            .map(|fw| fw.available_tools.iter().map(|m| m.id).collect())
            .unwrap_or_else(|| vec![""]);

        for tool_id in tool_ids {
            if !tool_id.is_empty() {
                let _ = app.tabs[tab_idx].select_tool_by_name(tool_id);
            }
            let out = ctx.run(
                egui::RawInput {
                    screen_rect: Some(screen),
                    ..Default::default()
                },
                |c| eframe::App::update(&mut app, c, &mut frame),
            );
            let ak = out.platform_output.accesskit_update.as_ref().unwrap();
            store.apply_update(ak);
            assert!(store.root_id().is_some());
            let (has_landmark, has_heading, has_control, has_vis) =
                summarize_projected_store(&store, screen);
            total_visualizations += usize::from(has_vis);
            assert!(
                has_landmark && has_heading && has_control,
                "Tab {} tool '{tool_id}' missing landmarks/headings/controls",
                app.tabs[tab_idx].name()
            );
        }
    }
    assert!(total_visualizations >= 20);
    store.clear();
    assert!(store.is_empty());
}

struct DummyTheory;

impl math_explorer::scientific_metadata::theory::TheoryDescribable for DummyTheory {
    fn theory_description(&self) -> String {
        "Logistic map r*x*(1-x)".to_string()
    }

    fn phonetic_description(&self) -> String {
        "Logistic map r times x times 1 minus x".to_string()
    }

    fn theory_citation(&self) -> String {
        "May (1976)".to_string()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[test]
fn test_speech_state_roundtrip_and_theory_hover_silence_by_default() {
    let _guard = crate::reflective_ui::tests::lock_test_mutex();
    super::speech::reset_test_speech_state();
    let mut app = crate::app::MathExplorerApp::default();
    assert!(!app.speech_enabled);
    assert!((app.speech_volume - 1.0).abs() < 1e-6);
    assert!(!is_speech_enabled());

    let ctx = egui::Context::default();
    let screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(800.0, 600.0),
    ));
    let mut btn_id = egui::Id::NULL;
    for focus_now in [false, true] {
        if focus_now {
            ctx.memory_mut(|m| m.request_focus(btn_id));
        }
        let _ = ctx.run(
            egui::RawInput {
                screen_rect,
                focused: focus_now,
                ..Default::default()
            },
            |c| {
                assert!(!c.options(|o| o.screen_reader));
                egui::CentralPanel::default().show(c, |ui| {
                    btn_id = ui
                        .button("📖 Theory")
                        .accessible_theory_hover(&DummyTheory)
                        .id;
                });
            },
        );
    }
    assert!(
        super::speech::test_speech_log().0.is_empty(),
        "Theory descriptions must stay silent when Speech Narration is disabled"
    );

    app.speech_enabled = true;
    app.speech_volume = 0.42;
    let saved = app.get_app_state();
    let mut restored = crate::app::MathExplorerApp::default();
    saved.apply(&mut restored);
    assert!(restored.speech_enabled);
    assert!((restored.speech_volume - 0.42).abs() < 1e-6);
    assert!(is_speech_enabled());
    assert!((speech_volume() - 0.42).abs() < 1e-6);
    super::speech::reset_test_speech_state();
}

#[cfg(target_arch = "wasm32")]
mod wasm_tests {
    use super::*;
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn wasm_accessibility_root_reexports_operate_cleanly() {
        clear_accesskit_dom_root();
        let runner = eframe::WebRunner::new();
        attach_web_runner_accesskit(&runner);
        runner.destroy();
    }
}
