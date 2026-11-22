use super::playback::{PlaybackAction, PlaybackTransport};
use super::InputMode;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::async_sim::{SimWorkerState, TimelineMetadata};
use eframe::egui;

fn key_input(key: egui::Key) -> egui::RawInput {
    egui::RawInput {
        events: vec![egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    }
}

#[test]
fn test_comma_step_back_and_open_bracket_keyframe_jump_shortcuts() {
    clear_test_announcements();
    let ctx = egui::Context::default();
    let meta = TimelineMetadata {
        current_step: 40,
        min_step: 0,
        max_step: 64,
        stride: 16,
    };

    let mut comma_action = None;
    let _ = ctx.run(key_input(egui::Key::Comma), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            comma_action = PlaybackTransport::from_worker_state(
                "morpho",
                "Morphogenesis",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert_eq!(comma_action, Some(PlaybackAction::StepBack));

    let mut bracket_action = None;
    let _ = ctx.run(key_input(egui::Key::OpenBracket), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bracket_action = PlaybackTransport::from_worker_state(
                "morpho",
                "Morphogenesis",
                SimWorkerState::Ready,
            )
            .timeline(Some(meta))
            .show(ui);
        });
    });
    assert_eq!(bracket_action, Some(PlaybackAction::ScrubTo(32)));

    let on_kf = TimelineMetadata {
        current_step: 32,
        min_step: 16,
        max_step: 64,
        stride: 16,
    };
    assert_eq!(on_kf.previous_keyframe_step(), 16);
    let near_min = TimelineMetadata {
        current_step: 19,
        min_step: 16,
        max_step: 64,
        stride: 16,
    };
    assert_eq!(near_min.previous_keyframe_step(), 16);
}

#[test]
fn test_non_opted_in_tools_hide_scrubber_and_disable_step_back() {
    let ctx = egui::Context::default();
    let mut base_id = egui::Id::NULL;
    let mut comma_action = None;
    let mut bracket_action = None;

    let _ = ctx.run(key_input(egui::Key::Comma), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            base_id = ui.make_persistent_id("spin");
            comma_action = PlaybackTransport::in_thread("spin", "Spin Visualizer", false).show(ui);
        });
    });
    assert_eq!(comma_action, None);

    let _ = ctx.run(key_input(egui::Key::OpenBracket), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            bracket_action =
                PlaybackTransport::in_thread("spin", "Spin Visualizer", false).show(ui);
        });
    });
    assert_eq!(bracket_action, None);

    let scrubber_info =
        ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("timeline_scrubber_info")));
    assert!(
        scrubber_info.is_none(),
        "Timeline scrubber must remain hidden when timeline metadata is None"
    );
    let step_back_info = ctx
        .data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("step_back_button_info")))
        .expect("Step back WidgetInfo recorded");
    assert!(
        !step_back_info.enabled,
        "Step back button must be disabled for non-opted-in tools"
    );
}

#[test]
fn test_opted_in_scrubber_touch_targets_commands_and_aria_announcements() {
    use egui_plot::commands::{CommandRegistryData, CommandTrigger};

    clear_test_announcements();
    let ctx = egui::Context::default();
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new("INPUT_MODE"), InputMode::Touch);
    });
    let meta = TimelineMetadata {
        current_step: 12,
        min_step: 0,
        max_step: 30,
        stride: 1,
    };

    let mut base_id = egui::Id::NULL;
    let mut total_rect = egui::Rect::NOTHING;
    let _ = ctx.run(key_input(egui::Key::Comma), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let scoped = ui.scope(|ui| {
                base_id = ui.make_persistent_id("hh");
                let act = PlaybackTransport::from_worker_state(
                    "hh",
                    "Hodgkin-Huxley Model",
                    SimWorkerState::Ready,
                )
                .timeline(Some(meta))
                .show(ui);
                assert_eq!(act, Some(PlaybackAction::StepBack));
            });
            total_rect = scoped.response.rect;
        });
    });

    assert!(
        total_rect.height() >= 88.0,
        "Expected touch transport + scrubber height >= 88px (2 x 44px rows), got {}",
        total_rect.height()
    );
    let scrubber_info =
        ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("timeline_scrubber_info")));
    assert!(scrubber_info.is_some());

    let cmds = ctx
        .data(|d| d.get_temp::<CommandRegistryData>(egui::Id::new("CMD_REGISTRY")))
        .unwrap_or_default()
        .commands;
    assert!(cmds
        .iter()
        .any(|c| { c.name == "Step Back" && c.trigger == CommandTrigger::Key(egui::Key::Comma) }));
    assert!(cmds.iter().any(|c| {
        c.name == "Previous Keyframe" && c.trigger == CommandTrigger::Key(egui::Key::OpenBracket)
    }));

    assert!(all_announcements().iter().any(|(msg, prio)| {
        msg.contains("Hodgkin-Huxley Model") && msg.contains("step 11") && prio == "polite"
    }));
}
