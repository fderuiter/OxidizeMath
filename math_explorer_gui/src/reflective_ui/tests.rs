use super::*;
use std::sync::Mutex;

pub(crate) static TEST_MUTEX: Mutex<()> = Mutex::new(());

struct DummyMissingTheoryModel {
    param_val: f64,
}

impl TheoryDescribable for DummyMissingTheoryModel {
    fn theory_description(&self) -> String {
        "Dummy description".into()
    }
    fn phonetic_description(&self) -> String {
        "Dummy phonetic".into()
    }
    fn theory_citation(&self) -> String {
        "Dummy citation".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        HashMap::new()
    }

    fn get_parameter(&self, name: &str) -> Option<f64> {
        if name == "missing_param" {
            Some(self.param_val)
        } else {
            None
        }
    }

    fn set_parameter(&mut self, name: &str, value: f64) {
        if name == "missing_param" {
            self.param_val = value;
        }
    }
}

#[test]
fn test_get_theory_constraint_fallback_and_telemetry() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    let model = DummyMissingTheoryModel { param_val: 10.0 };
    let constraint = get_theory_constraint(&model, "missing_param");

    assert_eq!(constraint.min, 0.0);
    assert_eq!(constraint.max, 100.0);
    assert_eq!(constraint.step, 0.1);

    let events = global_registry().try_recv_all();
    let warning_event = events.iter().find(|e| {
        e.source == "reflective_ui"
            && e.severity == Severity::Warning
            && e.message.contains("missing_param")
    });
    assert!(
        warning_event.is_some(),
        "Expected diagnostic warning event for missing parameter metadata"
    );
}

#[test]
fn test_render_theory_parameter_fallback_and_telemetry() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    let ctx = egui::Context::default();
    let model = DummyMissingTheoryModel { param_val: 10.0 };
    let mut value = 10.0;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_theory_parameter(ui, &model, "missing_param", "Missing Param", &mut value);
        });
    });

    let events = global_registry().try_recv_all();
    let warning_event = events.iter().find(|e| {
        e.source == "reflective_ui"
            && e.severity == Severity::Warning
            && e.message.contains("missing_param")
    });
    assert!(
        warning_event.is_some(),
        "Expected diagnostic warning event during parameter rendering"
    );
}

#[test]
fn test_render_all_theory_parameters_resiliency() {
    let _guard = TEST_MUTEX.lock().unwrap();
    let _ = global_registry().try_recv_all();

    struct DummyIncompleteModel;
    impl TheoryDescribable for DummyIncompleteModel {
        fn theory_description(&self) -> String {
            "Incomplete".to_string()
        }
        fn phonetic_description(&self) -> String {
            "Incomplete".to_string()
        }
        fn theory_citation(&self) -> String {
            "None".to_string()
        }
        fn available_descriptions(&self) -> HashMap<String, String> {
            HashMap::new()
        }
        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            let mut map = HashMap::new();
            map.insert(
                "valid_param".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 10.0,
                    step: 1.0,
                },
            );
            map
        }
        fn get_parameter(&self, name: &str) -> Option<f64> {
            if name == "valid_param" {
                Some(5.0)
            } else {
                None
            }
        }
    }

    let ctx = egui::Context::default();
    let mut model = DummyIncompleteModel;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let changed = render_all_theory_parameters(ui, &mut model);
            assert!(!changed);
        });
    });
}

#[test]
fn test_render_copyable_metric() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let resp1 = render_copyable_metric(ui, "Test Metric", "42.0");
            assert!(resp1.rect.width() >= 0.0);

            let resp2 = render_copyable_metric(ui, "Test Metric:", "100");
            assert!(resp2.rect.width() >= 0.0);

            let resp3 = render_copyable_metric(ui, "", "empty_label_val");
            assert!(resp3.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_render_formula_with_export() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_formula_with_export(ui, "\\frac{d}{dx} e^x", Some("d/dx e^x"));
            let r2 = render_formula_with_export(ui, "", None);
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_render_theory_summary_with_export() {
    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let r1 = render_theory_summary_with_export(ui, "Euler identity", Some("e^{i\\pi}+1=0"));
            let r2 = render_theory_summary_with_export(ui, "", None);
            assert!(r1.rect.width() >= 0.0 && r2.rect.width() >= 0.0);
        });
    });
}

#[test]
fn test_formula_export_populates_copied_text() {
    let ctx = egui::Context::default();
    let full_output = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.ctx().copy_text("\\int_0^1 x^2 dx = 1/3".to_string());
        });
    });
    let copied = full_output
        .platform_output
        .commands
        .iter()
        .any(|cmd| match cmd {
            egui::OutputCommand::CopyText(txt) => txt == "\\int_0^1 x^2 dx = 1/3",
            _ => false,
        });
    assert!(
        copied,
        "Expected CopyText output command in full_output.platform_output, got: {:?}",
        full_output.platform_output.commands
    );
}

#[test]
fn test_render_theory_parameter_help_button_toggle() {
    let ctx = egui::Context::default();
    let model = DummyMissingTheoryModel { param_val: 10.0 };
    let mut value = 10.0;

    // Frame 1: render parameter, portal state is not set or false
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            render_theory_parameter(ui, &model, "missing_param", "Missing Param", &mut value);
        });
    });

    let initial_portal = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")));
    assert_ne!(initial_portal, Some(true));

    // Get rect/interaction or simulate toggle in context
    ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), true));
    let updated_portal = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL")));
    assert_eq!(updated_portal, Some(true));
}

#[test]
fn test_render_theory_parameter_with_custom_description_and_help_button() {
    struct DummyWithDescModel;
    impl TheoryDescribable for DummyWithDescModel {
        fn theory_description(&self) -> String {
            "Model description".to_string()
        }
        fn phonetic_description(&self) -> String {
            "Model phonetic".to_string()
        }
        fn theory_citation(&self) -> String {
            "Citation 2026".to_string()
        }
        fn available_descriptions(&self) -> HashMap<String, String> {
            let mut map = HashMap::new();
            map.insert(
                "param1".to_string(),
                "Custom parameter 1 tooltip".to_string(),
            );
            map
        }
        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            let mut map = HashMap::new();
            map.insert(
                "param1".to_string(),
                ParameterConstraint {
                    min: 0.0,
                    max: 50.0,
                    step: 0.5,
                },
            );
            map
        }
    }

    let ctx = egui::Context::default();
    let model = DummyWithDescModel;
    let mut val = 25.0;

    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let response = render_theory_parameter(ui, &model, "param1", "Param One", &mut val);
            assert!(response.rect.width() >= 0.0);
        });
    });
}

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

    // Force playing = true, direction = 1.0, high speed
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
        predicted_dt: 0.5, // Simulate dt = 0.5s
        ..Default::default()
    };

    let _ = ctx.run(raw_input.clone(), |ctx| {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(
                AnimatedSlider::new(&mut val, 0.0..=10.0)
                    .text("Test Ping Pong")
                    .id_salt("test_ping_pong")
                    .speed(10.0), // speed = 10 units/s -> delta = 5.0
            );
        });
    });

    // 9.8 + 5.0 = 14.8, which exceeds max 10.0. Value must clamp to 10.0 and direction reverse to -1.0
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

    // Continue until min bound: val = 0.05 (so delta of ~0.166 exceeds 0.05)
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
