//! Preview formatting and inline `.tex` / `.rs` / `.py` snippet export controls for [`super::MathExpressionInputBar`].

use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use math_commons::ast::Expr;
use std::collections::HashMap;

pub(super) fn update_latex_preview(
    latex_cache: &mut String,
    lhs: Option<&str>,
    default_lhs: &str,
    expr: &Expr,
) {
    let lhs_label = lhs.or((!default_lhs.is_empty()).then_some(default_lhs));
    let expr_latex = expr.to_latex();
    *latex_cache = match lhs_label {
        Some(prefix) => format!("{prefix} = {expr_latex}"),
        None => expr_latex,
    };
}

pub(super) fn render_presets(
    ui: &mut egui::Ui,
    presets: &[(&'static str, &'static str)],
) -> Option<&'static str> {
    if presets.is_empty() {
        return None;
    }
    let mut selected_preset = None;
    ui.horizontal(|ui| {
        ui.label("Presets:");
        egui::ComboBox::from_id_salt(ui.id().with("math_presets"))
            .selected_text("Select Preset...")
            .show_ui(ui, |ui| {
                for (name, expr_str) in presets {
                    if ui.button(*name).clicked() {
                        selected_preset = Some(*expr_str);
                        ui.close();
                    }
                }
            });
    });
    ui.add_space(4.0);
    selected_preset
}

pub(super) fn render_status_row(
    ui: &mut egui::Ui,
    (err_detail, valid_expr, lhs_prefix, params): (
        Option<&str>,
        Option<&Expr>,
        Option<&str>,
        &HashMap<String, f64>,
    ),
    (latex, raw, axes): (&str, &str, &[String]),
    had_error: &mut bool,
    last_announced_error: &mut Option<String>,
) {
    if let Some(detail) = err_detail {
        let err_msg = format!("Syntax Error: {detail}");
        if last_announced_error.as_deref() != Some(&err_msg) {
            crate::accessibility::announce_status_with_priority(&err_msg, "assertive");
            *had_error = true;
            *last_announced_error = Some(err_msg);
        }
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.horizontal(|ui| {
            ui.label(
                egui::RichText::new(format!("⚠️ Syntax Error: {detail}"))
                    .color(palette.error)
                    .small(),
            );
        });
    } else if let Some(expr) = valid_expr {
        if *had_error {
            crate::accessibility::announce_status("Mathematical expression is valid");
            *had_error = false;
            *last_announced_error = None;
        }
        let calibrated = build_calibrated_preview(expr, lhs_prefix, latex, raw, params);
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("Preview:").weak());
            egui_plot::render_math_label(
                ui,
                &calibrated,
                egui::TextStyle::Body,
                ui.visuals().text_color(),
            );
            render_expr_export_actions(ui, expr, &calibrated, params, axes);
        });
    }
}

fn build_calibrated_preview(
    expr: &Expr,
    lhs_prefix: Option<&str>,
    symbolic_latex: &str,
    raw: &str,
    params: &HashMap<String, f64>,
) -> String {
    if !params.is_empty() {
        if let Ok(cal_rhs) = expr.to_latex_with_params(params) {
            return match lhs_prefix {
                Some(prefix) => format!("{prefix} = {cal_rhs}"),
                None => cal_rhs,
            };
        }
    }
    if !symbolic_latex.is_empty() {
        symbolic_latex.to_string()
    } else {
        raw.to_string()
    }
}

fn render_expr_export_actions(
    ui: &mut egui::Ui,
    expr: &Expr,
    calibrated_latex: &str,
    params: &HashMap<String, f64>,
    axes: &[String],
) {
    let axis_refs: Vec<&str> = axes.iter().map(String::as_str).collect();
    if ui
        .small_button("📋 LaTeX")
        .accessible_hover_text("Copy calibrated LaTeX expression to clipboard")
        .clicked()
    {
        copy_expr_snippet(ui, calibrated_latex, "Copied LaTeX expression to clipboard");
    }
    if ui
        .small_button("📋 Rust")
        .accessible_hover_text("Copy Rust f64 expression to clipboard")
        .clicked()
    {
        let rs_res = if axis_refs.is_empty() {
            expr.to_rust_expr(params)
        } else {
            expr.to_rust_expr_with_vars(&axis_refs, params)
        };
        if let Ok(rs) = rs_res {
            copy_expr_snippet(ui, &rs, "Copied Rust expression to clipboard");
        }
    }
    if ui
        .small_button("📋 Python")
        .accessible_hover_text("Copy Python numpy expression to clipboard")
        .clicked()
    {
        let py_res = if axis_refs.is_empty() {
            expr.to_python_expr(params)
        } else {
            expr.to_python_expr_with_vars(&axis_refs, params)
        };
        if let Ok(py) = py_res {
            copy_expr_snippet(ui, &py, "Copied Python expression to clipboard");
        }
    }
    if ui
        .small_button("💾 .tex")
        .accessible_hover_text("Save LaTeX expression to .tex file")
        .clicked()
    {
        #[cfg(not(test))]
        egui_plot::save_file_dialog(
            "expression.tex",
            calibrated_latex.as_bytes(),
            "LaTeX Document",
            &["tex"],
        );
        crate::accessibility::announce_status("Saved expression.tex");
    }
}

fn copy_expr_snippet(ui: &egui::Ui, text: &str, announcement: &str) {
    ui.ctx().copy_text(text.to_string());
    #[cfg(not(test))]
    let _ = egui_plot::export::copy_text_to_clipboard(text);
    crate::accessibility::announce_status(announcement);
}
