use super::playback::{PlaybackAction, PlaybackTransport};
use super::InputMode;
use crate::accessibility::{all_announcements, clear_test_announcements};
use crate::async_sim::SimWorkerState;
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
fn test_space_and_period_shortcuts_when_paused_and_running() {
    clear_test_announcements();
    let ctx = egui::Context::default();

    let mut action = None;
    let _ = ctx.run(key_input(egui::Key::Period), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            action = PlaybackTransport::from_worker_state(
                "wave",
                "Wave Simulator",
                SimWorkerState::Ready,
            )
            .show(ui);
        });
    });
    assert_eq!(action, Some(PlaybackAction::Step));

    let mut running_step = None;
    let _ = ctx.run(key_input(egui::Key::Period), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            running_step = PlaybackTransport::from_worker_state(
                "wave",
                "Wave Simulator",
                SimWorkerState::Running,
            )
            .show(ui);
        });
    });
    assert_eq!(running_step, None);

    let mut toggle = None;
    let _ = ctx.run(key_input(egui::Key::Space), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            toggle = PlaybackTransport::from_worker_state(
                "wave",
                "Wave Simulator",
                SimWorkerState::Running,
            )
            .show(ui);
        });
    });
    assert_eq!(toggle, Some(PlaybackAction::Pause));
}

#[test]
fn test_shortcuts_suppressed_when_wants_keyboard_input() {
    let ctx = egui::Context::default();
    let mut text = String::from("x^2");

    // Frame 1: focus the text input
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::in_thread("spin", "Spin Visualizer", false).show(ui);
            let resp = ui.add(egui::TextEdit::singleline(&mut text));
            resp.request_focus();
        });
    });

    // Frame 2: press Space and Period while text input has focus
    let mut space_action = None;
    let _ = ctx.run(key_input(egui::Key::Space), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            assert!(ctx.wants_keyboard_input());
            space_action = PlaybackTransport::in_thread("spin", "Spin Visualizer", false).show(ui);
            let _ = ui.add(egui::TextEdit::singleline(&mut text));
        });
    });
    assert_eq!(space_action, None);

    let mut period_action = None;
    let _ = ctx.run(key_input(egui::Key::Period), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            assert!(ctx.wants_keyboard_input());
            period_action = PlaybackTransport::in_thread("spin", "Spin Visualizer", false).show(ui);
            let _ = ui.add(egui::TextEdit::singleline(&mut text));
        });
    });
    assert_eq!(period_action, None);
}

#[test]
fn test_step_disabled_in_invalid_worker_states() {
    let ctx = egui::Context::default();
    for invalid_state in [
        SimWorkerState::Running,
        SimWorkerState::Stepping,
        SimWorkerState::Faulted,
        SimWorkerState::Terminated,
        SimWorkerState::Uninitialized,
    ] {
        let mut action = None;
        let _ = ctx.run(key_input(egui::Key::Period), |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                action =
                    PlaybackTransport::from_worker_state("sim", "Test Sim", invalid_state).show(ui);
            });
        });
        assert_eq!(
            action, None,
            "Step must be disabled in worker state {invalid_state}"
        );
    }
}

#[test]
fn test_touch_mode_enforces_44px_minimum_button_targets() {
    let ctx = egui::Context::default();
    ctx.data_mut(|d| {
        d.insert_temp(egui::Id::new("INPUT_MODE"), InputMode::Touch);
    });

    let mut transport_rect = egui::Rect::NOTHING;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let scoped = ui.scope(|ui| {
                let _ = PlaybackTransport::in_thread("touch_sim", "Touch Sim", false).show(ui);
            });
            transport_rect = scoped.response.rect;
        });
    });

    assert!(
        transport_rect.height() >= 44.0,
        "Expected >= 44.0 logical px touch height, got {}",
        transport_rect.height()
    );
    assert!(
        transport_rect.width() >= 44.0 * 3.0,
        "Expected >= 132.0 logical px combined width for 3 touch buttons, got {}",
        transport_rect.width()
    );
}

#[test]
fn test_accessible_hover_escape_dismisses_tooltip() {
    let ctx = egui::Context::default();
    let screen = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::vec2(800.0, 600.0),
    ));

    // Frame 1: initial render + Tab to focus the first transport button (Play)
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = PlaybackTransport::in_thread("esc_sim", "Escape Sim", false).show(ui);
            });
        },
    );
    clear_test_announcements();
    let mut tab_input = key_input(egui::Key::Tab);
    tab_input.screen_rect = screen;
    let _ = ctx.run(tab_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::in_thread("esc_sim", "Escape Sim", false).show(ui);
        });
    });
    // Settle focus frame so has_focus() is active during render
    clear_test_announcements();
    let _ = ctx.run(
        egui::RawInput {
            screen_rect: screen,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let _ = PlaybackTransport::in_thread("esc_sim", "Escape Sim", false).show(ui);
            });
        },
    );
    assert!(
        all_announcements()
            .iter()
            .any(|(msg, _)| msg.contains("Play Escape Sim")),
        "Focused/hovered play button must announce its tooltip"
    );

    // Next frame: press Escape while focused -> tooltip dismissed and not re-announced
    clear_test_announcements();
    let mut esc_input = key_input(egui::Key::Escape);
    esc_input.screen_rect = screen;
    let _ = ctx.run(esc_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::in_thread("esc_sim", "Escape Sim", false).show(ui);
        });
    });
    assert!(
        all_announcements().is_empty(),
        "Pressing Escape while active must dismiss tooltip announcements"
    );
}

#[test]
fn test_widget_info_and_polite_announcements() {
    clear_test_announcements();
    let ctx = egui::Context::default();
    let mut base_id = egui::Id::NULL;

    let _ = ctx.run(key_input(egui::Key::Period), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            base_id = ui.make_persistent_id("a11y_sim");
            let action =
                PlaybackTransport::in_thread("a11y_sim", "Spin Visualizer", false).show(ui);
            assert_eq!(action, Some(PlaybackAction::Step));
        });
    });

    let play_info = ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("play_button_info")));
    let step_info = ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("step_button_info")));
    let reset_info =
        ctx.data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("reset_button_info")));
    assert!(play_info.is_some());
    assert!(step_info.is_some());
    assert!(reset_info.is_some());

    let announcements = all_announcements();
    assert!(announcements.iter().any(|(msg, prio)| {
        msg.contains("Spin Visualizer") && msg.contains("step completed") && prio == "polite"
    }));
}

#[test]
fn test_playback_transport_registers_frame_commands() {
    use egui_plot::commands::{CommandRegistryData, CommandTrigger};

    let ready_ctx = egui::Context::default();
    let _ = ready_ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::from_worker_state(
                "cmd_sim",
                "Wave Simulator",
                SimWorkerState::Ready,
            )
            .show(ui);
        });
    });
    let ready_cmds = ready_ctx
        .data(|d| d.get_temp::<CommandRegistryData>(egui::Id::new("CMD_REGISTRY")))
        .unwrap_or_default()
        .commands;
    assert!(ready_cmds.iter().any(|c| {
        c.context == "Simulation Playback"
            && c.name == "Play / Pause"
            && c.trigger == CommandTrigger::Key(egui::Key::Space)
    }));
    assert!(ready_cmds.iter().any(|c| {
        c.context == "Simulation Playback"
            && c.name == "Step Forward"
            && c.trigger == CommandTrigger::Key(egui::Key::Period)
    }));

    let running_ctx = egui::Context::default();
    let _ = running_ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::from_worker_state(
                "cmd_sim",
                "Wave Simulator",
                SimWorkerState::Running,
            )
            .show(ui);
        });
    });
    let running_cmds = running_ctx
        .data(|d| d.get_temp::<CommandRegistryData>(egui::Id::new("CMD_REGISTRY")))
        .unwrap_or_default()
        .commands;
    assert!(running_cmds
        .iter()
        .any(|c| c.context == "Simulation Playback" && c.name == "Play / Pause"));
    assert!(!running_cmds
        .iter()
        .any(|c| c.context == "Simulation Playback" && c.name == "Step Forward"));
}

#[test]
fn test_opt_in_step_back_comma_shortcut_and_scrubber_metadata() {
    use egui_plot::commands::{CommandRegistryData, CommandTrigger};

    clear_test_announcements();
    let ctx = egui::Context::default();
    let mut base_id = egui::Id::NULL;

    // Non-opted-in transport ignores Comma and omits step_back / scrubber WidgetInfo
    let mut default_action = None;
    let _ = ctx.run(key_input(egui::Key::Comma), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            base_id = ui.make_persistent_id("legacy_sim");
            default_action =
                PlaybackTransport::in_thread("legacy_sim", "Legacy Sim", false).show(ui);
        });
    });
    assert_eq!(default_action, None);
    assert!(ctx
        .data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("step_back_button_info")))
        .is_none());
    assert!(ctx
        .data(|d| d.get_temp::<egui::WidgetInfo>(base_id.with("timeline_scrubber_info")))
        .is_none());

    // Opted-in transport responds to Comma, registers CommandRegistryData, and populates WidgetInfo
    let mut back_action = None;
    let mut twin_id = egui::Id::NULL;
    let _ = ctx.run(key_input(egui::Key::Comma), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            twin_id = ui.make_persistent_id("twin_sim");
            back_action = PlaybackTransport::from_worker_state(
                "twin_sim",
                "Attractor Plotter",
                SimWorkerState::Ready,
            )
            .can_step_back(true)
            .timeline_scrubber(42, 120)
            .show(ui);
        });
    });
    assert_eq!(back_action, Some(PlaybackAction::StepBack));
    assert!(ctx
        .data(|d| d.get_temp::<egui::WidgetInfo>(twin_id.with("step_back_button_info")))
        .is_some());
    assert!(ctx
        .data(|d| d.get_temp::<egui::WidgetInfo>(twin_id.with("timeline_scrubber_info")))
        .is_some());

    let cmds = ctx
        .data(|d| d.get_temp::<CommandRegistryData>(egui::Id::new("CMD_REGISTRY")))
        .unwrap_or_default()
        .commands;
    assert!(cmds.iter().any(|c| {
        c.context == "Simulation Playback"
            && c.name == "Step Backward"
            && c.trigger == CommandTrigger::Key(egui::Key::Comma)
    }));
    assert!(all_announcements().iter().any(|(msg, prio)| {
        msg.contains("Stepping backward Attractor Plotter") && prio == "polite"
    }));
}

#[test]
fn test_comma_shortcut_suppressed_when_text_edit_focused() {
    let ctx = egui::Context::default();
    let mut formula = String::from("1e-9 * cos(x0)");

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = PlaybackTransport::in_thread("twin", "Attractor Plotter", false)
                .can_step_back(true)
                .show(ui);
            ui.add(egui::TextEdit::singleline(&mut formula))
                .request_focus();
        });
    });

    let mut action = None;
    let _ = ctx.run(key_input(egui::Key::Comma), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            assert!(ctx.wants_keyboard_input());
            action = PlaybackTransport::in_thread("twin", "Attractor Plotter", false)
                .can_step_back(true)
                .show(ui);
            let _ = ui.add(egui::TextEdit::singleline(&mut formula));
        });
    });
    assert_eq!(action, None);
}
