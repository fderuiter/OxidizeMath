use crate::accessibility::AccessibleHoverText;
use eframe::egui;

pub mod slider;
#[allow(unused_imports)]
pub use slider::AnimatedSlider;
#[cfg(test)]
#[allow(unused_imports)]
pub use slider::AnimatedSliderState;

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
            if ui
                .button("📋")
                .accessible_hover_text("Copy value")
                .clicked()
            {
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
            egui_plot::render_math_label(
                ui,
                latex_str,
                egui::TextStyle::Body,
                ui.visuals().text_color(),
            );
            if ui
                .add_enabled(!is_latex_empty, egui::Button::new("📋 LaTeX"))
                .accessible_hover_text("Copy LaTeX formula")
                .clicked()
            {
                ui.ctx().copy_text(latex_str.to_string());
            }
            if ui
                .add_enabled(!is_plain_empty, egui::Button::new("📋 Text"))
                .accessible_hover_text("Copy plain text formula")
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
            if let Some(latex) = latex_str {
                if !plain_text.is_empty() && plain_text != latex {
                    ui.label(plain_text);
                }
                egui_plot::render_math_label(
                    ui,
                    latex,
                    egui::TextStyle::Body,
                    ui.visuals().text_color(),
                );
            } else {
                let summary_job = egui_plot::parse_latex_to_layout_job(
                    latex_text,
                    egui::TextStyle::Body,
                    ui,
                    ui.visuals().text_color(),
                );
                ui.add(egui::Label::new(summary_job).wrap());
            }
            if ui
                .add_enabled(!is_plain_empty, egui::Button::new("📋 Text"))
                .accessible_hover_text("Copy text summary")
                .clicked()
            {
                ui.ctx().copy_text(plain_text.to_string());
            }
            if ui
                .add_enabled(!is_latex_empty, egui::Button::new("📋 LaTeX"))
                .accessible_hover_text("Copy LaTeX formula/text")
                .clicked()
            {
                ui.ctx().copy_text(latex_text.to_string());
            }
        })
        .response;

    attach_theory_context_menu(response, plain_text, latex_text)
}

fn attach_theory_context_menu(
    response: egui::Response,
    plain_text: &str,
    latex_text: &str,
) -> egui::Response {
    let is_plain_empty = plain_text.trim().is_empty();
    let is_latex_empty = latex_text.trim().is_empty();
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
