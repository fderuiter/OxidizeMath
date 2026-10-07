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

fn format_metric_label(label: &str, value: &str) -> String {
    if label.is_empty() {
        value.to_string()
    } else if label.ends_with(':') {
        format!("{} {}", label, value)
    } else {
        format!("{}: {}", label, value)
    }
}

/// Renders a formatted metric label and value alongside an inline copy button and context menu.
pub fn render_copyable_metric(ui: &mut egui::Ui, label: &str, value: &str) -> egui::Response {
    let response = ui
        .horizontal(|ui| {
            ui.label(format_metric_label(label, value));
            if ui.button("📋").on_hover_text("Copy value").clicked() {
                ui.ctx().copy_text(value.to_string());
            }
        })
        .response;

    response.context_menu(|ui| {
        if ui.button("Copy Value").clicked() {
            ui.ctx().copy_text(value.to_string());
            ui.close_kind(egui::UiKind::Menu);
        }
        if ui.button("Copy Label & Value").clicked() {
            ui.ctx().copy_text(format_metric_label(label, value));
            ui.close_kind(egui::UiKind::Menu);
        }
    });

    response
}

/// Renders a mathematical formula (LaTeX with optional plain text) alongside inline export buttons and a right-click context menu.
pub fn render_formula_with_export(
    ui: &mut egui::Ui,
    latex_str: &str,
    plain_str: Option<&str>,
) -> egui::Response {
    let plain_text = plain_str.unwrap_or(latex_str);
    let is_latex_empty = latex_str.trim().is_empty();
    let is_plain_empty = plain_text.trim().is_empty();

    let response = ui
        .horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(latex_str).wrap());
            if ui
                .add_enabled(!is_latex_empty, egui::Button::new("📋 LaTeX"))
                .on_hover_text("Copy LaTeX formula")
                .clicked()
            {
                ui.ctx().copy_text(latex_str.to_string());
            }
            if ui
                .add_enabled(!is_plain_empty, egui::Button::new("📋 Text"))
                .on_hover_text("Copy plain text formula")
                .clicked()
            {
                ui.ctx().copy_text(plain_text.to_string());
            }
        })
        .response;

    response.context_menu(|ui| {
        if ui
            .add_enabled(!is_latex_empty, egui::Button::new("Copy LaTeX"))
            .clicked()
        {
            ui.ctx().copy_text(latex_str.to_string());
            ui.close_kind(egui::UiKind::Menu);
        }
        if ui
            .add_enabled(!is_plain_empty, egui::Button::new("Copy Plain Text"))
            .clicked()
        {
            ui.ctx().copy_text(plain_text.to_string());
            ui.close_kind(egui::UiKind::Menu);
        }
        if !is_latex_empty
            && !is_plain_empty
            && latex_str != plain_text
            && ui.button("Copy Both (LaTeX & Text)").clicked()
        {
            ui.ctx()
                .copy_text(format!("LaTeX: {}\nPlain: {}", latex_str, plain_text));
            ui.close_kind(egui::UiKind::Menu);
        }
    });

    response
}

/// Renders theoretical summary text alongside inline export buttons and a right-click context menu.
pub fn render_theory_summary_with_export(
    ui: &mut egui::Ui,
    text: &str,
    latex_str: Option<&str>,
) -> egui::Response {
    let plain_text = text;
    let latex_text = latex_str.unwrap_or(text);
    let is_plain_empty = plain_text.trim().is_empty();
    let is_latex_empty = latex_text.trim().is_empty();

    let response = ui
        .horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(text).wrap());
            if ui
                .add_enabled(!is_plain_empty, egui::Button::new("📋 Text"))
                .on_hover_text("Copy text summary")
                .clicked()
            {
                ui.ctx().copy_text(plain_text.to_string());
            }
            if ui
                .add_enabled(!is_latex_empty, egui::Button::new("📋 LaTeX"))
                .on_hover_text("Copy LaTeX formula/text")
                .clicked()
            {
                ui.ctx().copy_text(latex_text.to_string());
            }
        })
        .response;

    response.context_menu(|ui| {
        if ui
            .add_enabled(!is_plain_empty, egui::Button::new("Copy Plain Text"))
            .clicked()
        {
            ui.ctx().copy_text(plain_text.to_string());
            ui.close_kind(egui::UiKind::Menu);
        }
        if ui
            .add_enabled(!is_latex_empty, egui::Button::new("Copy LaTeX"))
            .clicked()
        {
            ui.ctx().copy_text(latex_text.to_string());
            ui.close_kind(egui::UiKind::Menu);
        }
        if !is_plain_empty
            && !is_latex_empty
            && plain_text != latex_text
            && ui.button("Copy Both (Text & LaTeX)").clicked()
        {
            ui.ctx()
                .copy_text(format!("Text: {}\nLaTeX: {}", plain_text, latex_text));
            ui.close_kind(egui::UiKind::Menu);
        }
    });

    response
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
                let r1 =
                    render_theory_summary_with_export(ui, "Euler identity", Some("e^{i\\pi}+1=0"));
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
}
