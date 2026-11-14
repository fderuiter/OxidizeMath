use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use federated_registry::{global_registry, Severity, TelemetryEvent};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

/// Default fallback constraint used when metadata is missing.
pub static DEFAULT_FALLBACK_CONSTRAINT: ParameterConstraint = ParameterConstraint {
    min: 0.0,
    max: 100.0,
    step: 0.1,
};

/// Returns fallback constraint struct.
pub fn default_fallback_constraint() -> ParameterConstraint {
    DEFAULT_FALLBACK_CONSTRAINT.clone()
}

/// Returns a reference to the static fallback constraint.
pub fn default_fallback_constraint_ref() -> &'static ParameterConstraint {
    &DEFAULT_FALLBACK_CONSTRAINT
}

static THEORY_METADATA_CACHE: LazyLock<
    RwLock<HashMap<&'static str, HashMap<String, ParameterConstraint>>>,
> = LazyLock::new(|| RwLock::new(HashMap::new()));

/// Clears the global parameter metadata cache (primarily used for test isolation).
pub fn clear_metadata_cache() {
    if let Ok(mut cache) = THEORY_METADATA_CACHE.write() {
        cache.clear();
    }
}

/// Returns the cached parameter constraints map for the given model type.
/// If not yet cached, queries `model.theory_parameters()` once and caches the result.
pub fn get_cached_theory_parameters<T: TheoryDescribable>(
    model: &T,
) -> HashMap<String, ParameterConstraint> {
    let key = std::any::type_name::<T>();
    if let Ok(cache) = THEORY_METADATA_CACHE.read() {
        if let Some(params) = cache.get(key) {
            return params.clone();
        }
    }
    let params = model.theory_parameters();
    if let Ok(mut cache) = THEORY_METADATA_CACHE.write() {
        cache.insert(key, params.clone());
    }
    params
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

/// Renders a UI parameter slider using pre-fetched parameter constraint and tooltip.
/// This helper performs zero heap allocations during slider rendering.
pub fn render_parameter_slider(
    ui: &mut egui::Ui,
    label: &str,
    value: &mut f64,
    constraint: &ParameterConstraint,
    tooltip: &str,
) -> egui::Response {
    let mut slider_response = None;

    ui.horizontal(|ui| {
        let slider = egui::Slider::new(value, constraint.min..=constraint.max)
            .step_by(constraint.step)
            .text(label);

        let slider_resp = ui.add(slider).accessible_hover_text(tooltip);

        let help_btn = ui
            .add(egui::Button::new("❓").small())
            .accessible_hover_text(tooltip);

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

/// Renders a UI parameter slider using pre-fetched parameter metadata maps without recreating them.
pub fn render_theory_parameter_with_metadata(
    ui: &mut egui::Ui,
    param_name: &str,
    label: &str,
    value: &mut f64,
    params: &HashMap<String, ParameterConstraint>,
    descs: &HashMap<String, String>,
    default_tooltip: &str,
) -> egui::Response {
    let constraint = lookup_theory_constraint(params, param_name);
    let tooltip = if let Some(desc) = descs.get(param_name) {
        desc.as_str()
    } else {
        default_tooltip
    };

    render_parameter_slider(ui, label, value, constraint, tooltip)
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
    let available_descs = model.available_descriptions();
    let default_tooltip = format!(
        "{}\n\nCitation: {}",
        model.theory_description(),
        model.theory_citation()
    );

    render_theory_parameter_with_metadata(
        ui,
        param_name,
        label,
        value,
        &params,
        &available_descs,
        &default_tooltip,
    )
}

/// Looks up a single parameter constraint from a pre-fetched parameter map
/// without generating full map copies or heap allocations.
pub fn lookup_theory_constraint<'a>(
    params: &'a HashMap<String, ParameterConstraint>,
    param_name: &str,
) -> &'a ParameterConstraint {
    if let Some(constraint) = params.get(param_name) {
        constraint
    } else {
        log_missing_metadata_warning(param_name);
        default_fallback_constraint_ref()
    }
}

/// Queries a single parameter constraint by borrowed reference from a pre-fetched parameter map.
pub fn get_theory_constraint_ref<'a>(
    params: &'a HashMap<String, ParameterConstraint>,
    param_name: &str,
) -> &'a ParameterConstraint {
    lookup_theory_constraint(params, param_name)
}

/// Queries a single parameter constraint using cached theoretical constraints,
/// avoiding allocating or copying the full metadata map on every frame.
pub fn get_cached_theory_constraint<T: TheoryDescribable>(
    model: &T,
    param_name: &str,
) -> ParameterConstraint {
    let key = std::any::type_name::<T>();
    if let Ok(cache) = THEORY_METADATA_CACHE.read() {
        if let Some(params) = cache.get(key) {
            return lookup_theory_constraint(params, param_name).clone();
        }
    }

    let params = model.theory_parameters();
    let result = lookup_theory_constraint(&params, param_name).clone();

    if let Ok(mut cache) = THEORY_METADATA_CACHE.write() {
        cache.insert(key, params);
    }

    result
}

/// Fallback mechanism for complex parameters that require a custom UI layout
/// but still need to be validated against theoretical limits.
/// Caches the parameter metadata per model type to avoid recreating maps on every frame.
pub fn get_theory_constraint<T: TheoryDescribable>(
    model: &T,
    param_name: &str,
) -> ParameterConstraint {
    get_cached_theory_constraint(model, param_name)
}

/// Dynamically renders UI sliders for all properties defined in the model metadata.
/// Queries metadata once per batch render and passes borrowed references to child slider renderers.
pub fn render_all_theory_parameters<T: TheoryDescribable>(
    ui: &mut egui::Ui,
    model: &mut T,
) -> bool {
    let mut changed = false;

    let params = model.theory_parameters();
    let descs = model.available_descriptions();
    let default_tooltip = format!(
        "{}\n\nCitation: {}",
        model.theory_description(),
        model.theory_citation()
    );

    let mut param_names: Vec<&str> = params.keys().map(|s| s.as_str()).collect();
    param_names.sort();

    for param_name in param_names {
        if let Some(mut value) = model.get_parameter(param_name) {
            let response = render_theory_parameter_with_metadata(
                ui,
                param_name,
                param_name,
                &mut value,
                &params,
                &descs,
                &default_tooltip,
            );
            if response.changed() {
                model.set_parameter(param_name, value);
                changed = true;
            }
        }
    }

    changed
}

/// Renders a collapsible parameter sensitivity and perturbation analysis panel for a theory-describable model.
pub fn render_sensitivity_panel<T: TheoryDescribable>(
    ui: &mut egui::Ui,
    model: &mut T,
    engine: &mut crate::sensitivity::SensitivityEngine,
    id_source: &str,
) {
    let params_map = get_cached_theory_parameters(model);
    let mut param_list = Vec::new();
    for key in params_map.keys() {
        if let Some(val) = model.get_parameter(key) {
            param_list.push((key.clone(), val));
        }
    }
    param_list.sort_by(|a, b| a.0.cmp(&b.0));

    engine.compute(&param_list, |p_map| p_map.values().map(|v| v.abs()).sum());

    crate::sensitivity::render_sensitivity_panel_ui(ui, engine, id_source);
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

    let formula_job = egui_plot::parse_latex_to_layout_job(
        latex_str,
        egui::TextStyle::Body,
        ui,
        ui.visuals().text_color(),
    );

    let response = ui
        .horizontal_wrapped(|ui| {
            ui.add(egui::Label::new(formula_job).wrap());
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
