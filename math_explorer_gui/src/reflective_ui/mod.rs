use crate::accessibility::AccessibleHoverText;
use eframe::egui;

pub mod slider;
pub use slider::AnimatedSlider;
#[cfg(test)]
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

fn copy_formula_text(ui: &egui::Ui, text: &str, status: &str) {
    ui.ctx().copy_text(text.to_string());
    #[cfg(not(test))]
    let _ = egui_plot::export::copy_text_to_clipboard(text);
    crate::accessibility::announce_status(status);
}

fn save_formula_tex(latex_str: &str) {
    #[cfg(test)]
    let _ = latex_str;
    #[cfg(not(test))]
    egui_plot::save_file_dialog(
        "formula.tex",
        latex_str.as_bytes(),
        "LaTeX Document",
        &["tex"],
    );
    crate::accessibility::announce_status("Saved formula.tex");
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
                copy_formula_text(ui, latex_str, "Copied LaTeX formula to clipboard");
            }
            if ui
                .add_enabled(!is_plain_empty, egui::Button::new("📋 Text"))
                .accessible_hover_text("Copy plain text formula")
                .clicked()
            {
                copy_formula_text(ui, plain_text, "Copied plain text formula to clipboard");
            }
            if ui
                .add_enabled(!is_latex_empty, egui::Button::new("💾 .tex"))
                .accessible_hover_text("Save LaTeX formula to .tex file")
                .clicked()
            {
                save_formula_tex(latex_str);
            }
        })
        .response;

    attach_formula_context_menu(response, latex_str, plain_text)
}

fn attach_formula_context_menu(
    response: egui::Response,
    latex_str: &str,
    plain_text: &str,
) -> egui::Response {
    let is_latex_empty = latex_str.trim().is_empty();
    let is_plain_empty = plain_text.trim().is_empty();
    response.context_menu(|ui| {
        if ui
            .add_enabled(!is_latex_empty, egui::Button::new("Copy LaTeX"))
            .clicked()
        {
            copy_formula_text(ui, latex_str, "Copied LaTeX formula to clipboard");
            ui.close_kind(egui::UiKind::Menu);
        }
        if ui
            .add_enabled(!is_plain_empty, egui::Button::new("Copy Plain Text"))
            .clicked()
        {
            copy_formula_text(ui, plain_text, "Copied plain text formula to clipboard");
            ui.close_kind(egui::UiKind::Menu);
        }
        if ui
            .add_enabled(!is_latex_empty, egui::Button::new("Save LaTeX (.tex)"))
            .clicked()
        {
            save_formula_tex(latex_str);
            ui.close_kind(egui::UiKind::Menu);
        }
        if !is_latex_empty
            && !is_plain_empty
            && latex_str != plain_text
            && ui.button("Copy Both (LaTeX & Text)").clicked()
        {
            let both = format!("LaTeX: {}\nPlain: {}", latex_str, plain_text);
            copy_formula_text(ui, &both, "Copied LaTeX and plain text formula");
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
