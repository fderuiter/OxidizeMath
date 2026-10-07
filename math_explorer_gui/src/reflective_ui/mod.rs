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

/// Persistent animated slider state per widget ID.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnimatedSliderState {
    /// Whether parameter animation is currently playing.
    pub playing: bool,
    /// Sweep direction (+1.0 for forward, -1.0 for reverse).
    pub direction: f64,
}

impl Default for AnimatedSliderState {
    fn default() -> Self {
        Self {
            playing: false,
            direction: 1.0,
        }
    }
}

/// An interactive parameter slider coupled with play/pause animation controls.
pub struct AnimatedSlider<'a> {
    value: &'a mut f64,
    min: f64,
    max: f64,
    step: f64,
    label: String,
    id_salt: Option<egui::Id>,
    speed: Option<f64>,
}

impl<'a> AnimatedSlider<'a> {
    /// Creates an `AnimatedSlider` from a value reference and range bounds.
    pub fn new(value: &'a mut f64, range: std::ops::RangeInclusive<f64>) -> Self {
        let min = *range.start();
        let max = *range.end();
        Self {
            value,
            min,
            max,
            step: 0.0,
            label: String::new(),
            id_salt: None,
            speed: None,
        }
    }

    /// Creates an `AnimatedSlider` from a `ParameterConstraint`.
    pub fn from_constraint(value: &'a mut f64, constraint: &ParameterConstraint) -> Self {
        Self {
            value,
            min: constraint.min,
            max: constraint.max,
            step: constraint.step,
            label: String::new(),
            id_salt: None,
            speed: None,
        }
    }

    /// Sets the label text for the slider.
    pub fn text(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Sets the ID salt used for storing animation state in egui memory.
    pub fn id_salt(mut self, salt: impl std::hash::Hash) -> Self {
        self.id_salt = Some(egui::Id::new(salt));
        self
    }

    /// Sets step size for slider.
    pub fn step_by(mut self, step: f64) -> Self {
        self.step = step;
        self
    }

    /// Sets custom animation sweep speed per second.
    pub fn speed(mut self, speed: f64) -> Self {
        self.speed = Some(speed);
        self
    }

    fn advance_animation(&mut self, ui: &mut egui::Ui, state: &mut AnimatedSliderState) -> bool {
        let dt = ui.input(|i| i.stable_dt as f64).min(0.1);
        let range = (self.max - self.min).abs();
        let speed = self
            .speed
            .unwrap_or_else(|| if range > 0.0 { range / 5.0 } else { 1.0 });

        if range > 0.0 && dt > 0.0 {
            *self.value += speed * dt * state.direction;
            if *self.value >= self.max {
                *self.value = self.max;
                state.direction = -1.0;
            } else if *self.value <= self.min {
                *self.value = self.min;
                state.direction = 1.0;
            }
            *self.value = self.value.clamp(self.min, self.max);
            ui.ctx().request_repaint();
            true
        } else {
            false
        }
    }
}

impl<'a> egui::Widget for AnimatedSlider<'a> {
    fn ui(mut self, ui: &mut egui::Ui) -> egui::Response {
        let id = self
            .id_salt
            .unwrap_or_else(|| ui.make_persistent_id(&self.label));
        let mut state: AnimatedSliderState =
            ui.ctx().data_mut(|d| d.get_temp(id)).unwrap_or_default();

        let mut changed = false;

        let response = ui
            .horizontal(|ui| {
                let button_text = if state.playing { "⏸" } else { "▶" };
                let play_btn = ui.button(button_text).on_hover_text(if state.playing {
                    "Pause animation"
                } else {
                    "Play animation"
                });

                if play_btn.clicked() {
                    state.playing = !state.playing;
                    changed = true;
                }

                let mut slider = egui::Slider::new(self.value, self.min..=self.max);
                if !self.label.is_empty() {
                    slider = slider.text(&self.label);
                }
                if self.step > 0.0 {
                    slider = slider.step_by(self.step);
                }

                let slider_resp = ui.add(slider);
                if slider_resp.changed() {
                    changed = true;
                }

                if slider_resp.lost_focus() {
                    state.playing = false;
                }

                slider_resp
            })
            .response;

        if state.playing && self.advance_animation(ui, &mut state) {
            changed = true;
        }

        ui.ctx().data_mut(|d| d.insert_temp(id, state));

        let mut response = response;
        if changed {
            response.mark_changed();
        }

        response
    }
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

    let available_descs = model.available_descriptions();
    let tooltip = if let Some(param_desc) = available_descs.get(param_name) {
        param_desc.clone()
    } else {
        format!(
            "{}\n\nCitation: {}",
            model.theory_description(),
            model.theory_citation()
        )
    };

    let mut slider_response = None;

    ui.horizontal(|ui| {
        let animated_slider = AnimatedSlider::from_constraint(value, constraint)
            .text(label)
            .id_salt(param_name);

        let slider_resp = ui.add(animated_slider).accessible_hover_text(&tooltip);

        let help_btn = ui
            .add(egui::Button::new("❓").small())
            .accessible_hover_text(&tooltip);

        if help_btn.clicked() {
            let current = ui.ctx().data(|d| {
                d.get_temp::<bool>(egui::Id::new("SHOW_THEORY_PORTAL"))
                    .unwrap_or(false)
            });
            ui.ctx()
                .data_mut(|d| d.insert_temp(egui::Id::new("SHOW_THEORY_PORTAL"), !current));
        }

        slider_response = Some(slider_resp);
    });

    slider_response.unwrap_or_else(|| ui.label(label))
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
            let response = render_theory_parameter(ui, model, &param_name, &param_name, &mut value);
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
pub(crate) mod tests;
