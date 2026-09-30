use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use federated_registry::{global_registry, Severity, TelemetryEvent};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

fn default_fallback_constraint() -> ParameterConstraint {
    ParameterConstraint {
        min: 0.0,
        max: 100.0,
        step: 0.1,
    }
}

fn log_missing_metadata_warning(param_name: &str) {
    let mut metadata = HashMap::new();
    metadata.insert("parameter".to_string(), param_name.to_string());
    global_registry().emit(TelemetryEvent {
        source: "reflective_ui".to_string(),
        severity: Severity::Warning,
        message: format!(
            "Parameter '{}' missing from theory metadata; using fallback constraints (0.0 to 100.0)",
            param_name
        ),
        metadata,
        thread_name: std::thread::current().name().map(|s| s.to_string()),
    });
}

/// Renders a UI parameter directly from the theoretical constraints.
pub fn render_theory_parameter<T: TheoryDescribable>(
    ui: &mut egui::Ui,
    model: &T,
    param_name: &str,
    label: &str,
    value: &mut f64,
) -> egui::Response {
    let params = model.theory_parameters();
    let fallback = default_fallback_constraint();
    let constraint = if let Some(c) = params.get(param_name) {
        c
    } else {
        log_missing_metadata_warning(param_name);
        &fallback
    };

    let slider = egui::Slider::new(value, constraint.min..=constraint.max)
        .step_by(constraint.step)
        .text(label);

    let response = ui.add(slider);
    let available_descs = model.available_descriptions();

    if let Some(param_desc) = available_descs.get(param_name) {
        response.accessible_hover_text(param_desc)
    } else {
        let tooltip = format!(
            "{}\n\nCitation: {}",
            model.theory_description(),
            model.theory_citation()
        );

        response.accessible_hover_text(tooltip)
    }
}

/// Fallback mechanism for complex parameters that require a custom UI layout
/// but still need to be validated against theoretical limits.
pub fn get_theory_constraint<T: TheoryDescribable>(
    model: &T,
    param_name: &str,
) -> ParameterConstraint {
    if let Some(constraint) = model.theory_parameters().get(param_name) {
        constraint.clone()
    } else {
        log_missing_metadata_warning(param_name);
        default_fallback_constraint()
    }
}

/// Dynamically renders UI sliders for all properties defined in the model metadata.
pub fn render_all_theory_parameters<T: TheoryDescribable>(
    ui: &mut egui::Ui,
    model: &mut T,
) -> bool {
    let mut changed = false;

    let params = model.theory_parameters();
    let mut param_names: Vec<String> = params.keys().cloned().collect();
    param_names.sort();

    for param_name in param_names {
        if let Some(mut value) = model.get_parameter(&param_name) {
            let fallback = default_fallback_constraint();
            let constraint = if let Some(c) = params.get(&param_name) {
                c
            } else {
                log_missing_metadata_warning(&param_name);
                &fallback
            };
            let slider = egui::Slider::new(&mut value, constraint.min..=constraint.max)
                .step_by(constraint.step)
                .text(&param_name);

            let mut response = ui.add(slider);
            let available_descs = model.available_descriptions();

            if let Some(param_desc) = available_descs.get(&param_name) {
                response = response.accessible_hover_text(param_desc);
            } else {
                let tooltip = format!(
                    "{}\n\nCitation: {}",
                    model.theory_description(),
                    model.theory_citation()
                );
                response = response.accessible_hover_text(tooltip);
            }

            if response.changed() {
                model.set_parameter(&param_name, value);
                changed = true;
            }
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static TEST_MUTEX: Mutex<()> = Mutex::new(());

    struct DummyMissingTheoryModel {
        param_val: f64,
    }

    impl TheoryDescribable for DummyMissingTheoryModel {
        fn theory_description(&self) -> String {
            "Dummy description".to_string()
        }

        fn phonetic_description(&self) -> String {
            "Dummy phonetic".to_string()
        }

        fn theory_citation(&self) -> String {
            "Dummy citation".to_string()
        }

        fn available_descriptions(&self) -> HashMap<String, String> {
            HashMap::new()
        }

        fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
            // Intentionally empty parameter map
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
}
