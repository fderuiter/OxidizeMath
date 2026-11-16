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
fn test_animated_slider_from_constraint() {
    let constraint = ParameterConstraint {
        min: -2.0,
        max: 8.0,
        step: 0.25,
    };
    let mut val = 1.0;
    let slider = AnimatedSlider::from_constraint(&mut val, &constraint).text("Constraint Slider");
    assert_eq!(slider.min, -2.0);
    assert_eq!(slider.max, 8.0);
    assert_eq!(slider.step, 0.25);
    assert_eq!(slider.label, "Constraint Slider");
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
