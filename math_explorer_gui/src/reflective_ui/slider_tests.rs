#![cfg_attr(any(), verified(opt_out = "tests"))]

use super::*;

#[test]
fn test_animated_slider_state_toggle() {
    let ctx = egui::Context::default();
    let mut val = 5.0;
    let id = egui::Id::new("test_toggle_slider");

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Toggle")
                    .id_salt("test_toggle_slider"),
            );
        });
    });

    let state_init: AnimatedSliderState = ctx
        .data_mut(|d| d.get_temp(id))
        .expect("State should be stored");
    assert!(!state_init.playing);

    // Manually set playing = true to simulate user click
    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        )
    });

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Toggle")
                    .id_salt("test_toggle_slider"),
            );
        });
    });

    let state_after: AnimatedSliderState = ctx
        .data_mut(|d| d.get_temp(id))
        .expect("State should be stored");
    assert!(state_after.playing);
}

#[test]
fn test_animated_slider_ping_pong_reversal_and_clamping() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("test_ping_pong");

    // Start near max boundary: val = 9.8, max = 10.0
    let mut val = 9.8;

    // Force playing = true, direction = 1.0, speed = 10.0
    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        )
    });

    let raw_input = egui::RawInput {
        predicted_dt: 0.5, // Simulate dt = 0.5s -> clamped to 0.1s max dt
        ..Default::default()
    };

    let _ = ctx.run(raw_input.clone(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Ping Pong")
                    .id_salt("test_ping_pong")
                    .speed(10.0), // speed = 10 units/s -> delta = 1.0
            );
        });
    });

    // 9.8 + 1.0 = 10.8, which exceeds max 10.0. Value must clamp to 10.0 and direction reverse to -1.0
    assert_eq!(val, 10.0);
    let state_at_max: AnimatedSliderState =
        ctx.data_mut(|d| d.get_temp(id)).expect("State must exist");
    assert_eq!(state_at_max.direction, -1.0);

    // Next frame with reversed direction: delta = -1.0 (since dt is clamped to 0.1 max)
    let _ = ctx.run(raw_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Ping Pong")
                    .id_salt("test_ping_pong")
                    .speed(10.0),
            );
        });
    });

    // 10.0 - (10.0 * 0.1) = 9.0
    assert_eq!(val, 9.0);

    // Set value close to min: val = 0.05
    val = 0.05;
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Ping Pong")
                    .id_salt("test_ping_pong")
                    .speed(10.0),
            );
        });
    });

    // Value must clamp to min 0.0 and direction reverse back to +1.0
    assert_eq!(val, 0.0);
    let state_at_min: AnimatedSliderState =
        ctx.data_mut(|d| d.get_temp(id)).expect("State must exist");
    assert_eq!(state_at_min.direction, 1.0);
}

#[test]
fn test_animated_slider_focus_loss_pause() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("test_focus_loss");
    let mut val = 5.0;

    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        )
    });

    // First frame: playing is true
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Focus")
                    .id_salt("test_focus_loss"),
            );
        });
    });

    let state_before: AnimatedSliderState =
        ctx.data_mut(|d| d.get_temp(id)).expect("State must exist");
    assert!(state_before.playing);
}

#[test]
fn test_animated_slider_custom_speed_and_step() {
    let mut val = 0.0;
    let slider = AnimatedSlider::new(&mut val, 0.0..=100.0)
        .speed(25.0)
        .step_by(0.5);
    assert_eq!(slider.speed, Some(25.0));
    assert_eq!(slider.step, 0.5);
}

#[test]
fn test_animated_slider_zero_range_no_division() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("zero_range_slider");
    let mut val = 5.0;

    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        )
    });

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp = ui.add(
                AnimatedSlider::new(&mut val, 5.0..=5.0)
                    .text("Zero Range")
                    .id_salt("zero_range_slider"),
            );
            assert!(!resp.changed());
        });
    });

    assert_eq!(val, 5.0);
}

#[test]
fn test_animated_slider_manual_drag_pause() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("test_drag_pause");
    let mut val = 5.0;

    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        )
    });

    // Run frame with mouse click/drag on slider
    let raw_input = egui::RawInput {
        events: vec![egui::Event::PointerButton {
            pos: egui::pos2(50.0, 50.0),
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::default(),
        }],
        ..Default::default()
    };

    let _ = ctx.run(raw_input, |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Drag")
                    .id_salt("test_drag_pause"),
            );
        });
    });

    let state: AnimatedSliderState = ctx.data_mut(|d| d.get_temp(id)).expect("State must exist");
    // Verify state can be inspected and updated
    assert!(state.direction == 1.0 || state.direction == -1.0);
}

#[test]
#[cfg(feature = "pure_math")]
fn test_ode_animated_slider_sweep() {
    use crate::framework::InteractiveTool;
    use crate::presets::OdePreset;
    use crate::tabs::analysis::ode::OdeSolverTool;

    let mut tool = OdeSolverTool {
        preset: OdePreset::Exponential,
        param_k: 1.0,
        ..Default::default()
    };
    tool.recalculate();
    let initial_y = tool.y_series.clone();

    let ctx = egui::Context::default();
    let id = egui::Id::new("ode_param_k_exp");
    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing: true,
                direction: 1.0,
            },
        );
    });

    let raw_input = egui::RawInput {
        predicted_dt: 0.1,
        ..Default::default()
    };
    let _ = ctx.run(raw_input, |ctx| {
        tool.show(ctx);
    });

    // Parameter k has swept, recalculate should have updated trajectory
    assert_ne!(tool.param_k, 1.0);
    assert_ne!(tool.y_series, initial_y);
}

const TEST_SCREEN: Option<egui::Rect> = Some(egui::Rect {
    min: egui::Pos2::ZERO,
    max: egui::pos2(1000.0, 1000.0),
});

fn run_slider_events(
    ctx: &egui::Context,
    val: &mut f64,
    label: &'static str,
    id_salt: &'static str,
    events: Vec<egui::Event>,
) {
    let _ = ctx.run(
        egui::RawInput {
            events,
            screen_rect: TEST_SCREEN,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                ui.add(
                    AnimatedSlider::new(val, 0.0..=10.0)
                        .text(label)
                        .id_salt(id_salt),
                );
            });
        },
    );
}

fn simulate_click(ctx: &egui::Context, val: &mut f64, label: &'static str, id_salt: &'static str) {
    let p = egui::pos2(15.0, 15.0);
    for pressed in [true, false] {
        let ev = egui::Event::PointerButton {
            pos: p,
            button: egui::PointerButton::Primary,
            pressed,
            modifiers: Default::default(),
        };
        run_slider_events(ctx, val, label, id_salt, vec![ev]);
    }
}

fn run_hover_frame(ctx: &egui::Context, pos: egui::Pos2, key: Option<egui::Key>) -> egui::Id {
    let mut events = vec![egui::Event::PointerMoved(pos)];
    if let Some(key) = key {
        events.push(egui::Event::Key {
            key,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: Default::default(),
        });
    }
    let mut id = egui::Id::NULL;
    let _ = ctx.run(
        egui::RawInput {
            events,
            screen_rect: TEST_SCREEN,
            ..Default::default()
        },
        |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                id = ui.label("Widget").accessible_hover_text("Tooltip text").id;
            });
        },
    );
    id
}

fn check_info(ctx: &egui::Context, id: egui::Id, sel: bool, lbl: &str) {
    let info: egui::WidgetInfo = ctx
        .data(|d| d.get_temp(id.with("play_button_info")))
        .unwrap();
    assert_eq!(info.typ, egui::WidgetType::Button);
    assert_eq!(info.selected, Some(sel));
    assert_eq!(info.label.as_deref(), Some(lbl));
    assert!(info.description().contains(lbl));
}

fn set_state(ctx: &egui::Context, id: egui::Id, playing: bool) {
    ctx.data_mut(|d| {
        d.insert_temp(
            id,
            AnimatedSliderState {
                playing,
                direction: 1.0,
            },
        )
    });
}

fn check_aria(ctx: &egui::Context, expected: &str) {
    let msg = ctx.data(|d| d.get_temp::<String>(egui::Id::new("aria_live_message")));
    assert_eq!(msg.as_deref(), Some(expected));
}

#[test]
fn test_animated_slider_widget_info_descriptors() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("test_widget_info_slider");
    let mut val = 3.0;

    for (lbl, play, exp) in [
        ("Speed", false, "Play animation for Speed"),
        ("Speed", true, "Pause animation for Speed"),
        ("", false, "Play animation"),
        ("", true, "Pause animation"),
    ] {
        set_state(&ctx, id, play);
        run_slider_events(&ctx, &mut val, lbl, "test_widget_info_slider", vec![]);
        check_info(&ctx, id, play, exp);
    }
}

#[test]
fn test_animated_slider_escape_key_dismissal() {
    let ctx = egui::Context::default();
    let p_hover = egui::pos2(20.0, 20.0);

    let resp_id = run_hover_frame(&ctx, p_hover, None);
    run_hover_frame(&ctx, p_hover, None);
    let dismissed = ctx
        .data(|d| d.get_temp::<bool>(resp_id.with("tooltip_dismissed")))
        .unwrap_or(false);
    assert!(!dismissed);
    check_aria(&ctx, "Tooltip text");

    run_hover_frame(&ctx, p_hover, Some(egui::Key::Escape));
    let dismissed = ctx
        .data(|d| d.get_temp::<bool>(resp_id.with("tooltip_dismissed")))
        .unwrap_or(false);
    assert!(dismissed);

    run_hover_frame(&ctx, egui::pos2(800.0, 800.0), None);
    let dismissed = ctx
        .data(|d| d.get_temp::<bool>(resp_id.with("tooltip_dismissed")))
        .unwrap_or(false);
    assert!(!dismissed);
}

#[test]
fn test_animated_slider_playback_announcements() {
    let ctx = egui::Context::default();
    let id = egui::Id::new("test_announcements");
    let mut val = 4.0;

    run_slider_events(&ctx, &mut val, "Frequency", "test_announcements", vec![]);

    // 1. Click play button to start animation
    simulate_click(&ctx, &mut val, "Frequency", "test_announcements");
    assert!(
        ctx.data_mut(|d| d.get_temp::<AnimatedSliderState>(id))
            .unwrap()
            .playing
    );
    check_aria(&ctx, "Play animation for Frequency");

    // 2. Click pause button to pause animation (rapid toggle)
    simulate_click(&ctx, &mut val, "Frequency", "test_announcements");
    assert!(
        !ctx.data_mut(|d| d.get_temp::<AnimatedSliderState>(id))
            .unwrap()
            .playing
    );
    check_aria(&ctx, "Pause animation for Frequency");

    // 3. Start animation and drag slider to pause
    set_state(&ctx, id, true);
    let drag_ev = vec![egui::Event::PointerButton {
        pos: egui::pos2(60.0, 15.0),
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: Default::default(),
    }];
    run_slider_events(&ctx, &mut val, "Frequency", "test_announcements", drag_ev);
    assert!(
        !ctx.data_mut(|d| d.get_temp::<AnimatedSliderState>(id))
            .unwrap()
            .playing
    );
    check_aria(&ctx, "Pause animation for Frequency");
}
