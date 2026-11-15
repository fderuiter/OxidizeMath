//! Reusable egui mathematical expression input bar with live syntax validation and LaTeX preview.

use eframe::egui;
use math_commons::ast::{parse_equation, Expr, ParseError};

/// Reusable input bar component for dynamic mathematical equations.
///
/// Features:
/// - Real-time Pratt-parser syntax validation.
/// - Live LaTeX preview rendering via `egui_plot::render_math_label`.
/// - Clear inline syntax error highlighting in red.
/// - Preset equations selection dropdown.
#[derive(Clone, Debug)]
pub struct MathExpressionInputBar {
    /// Raw expression string typed by the user.
    pub raw_text: String,
    /// Default left-hand side equation label (e.g. "y'" or "w").
    pub default_lhs: String,
    /// Extracted LHS name if explicitly typed with '='.
    pub lhs_name: Option<String>,
    /// Compiled AST expression or syntax error.
    pub parsed_expr: Result<Expr, ParseError>,
    /// Formatted LaTeX string for preview rendering.
    pub latex_cache: String,
    /// Available presets as (Name, Equation String) pairs.
    pub presets: Vec<(&'static str, &'static str)>,
}

impl MathExpressionInputBar {
    /// Creates a new mathematical expression input bar.
    #[must_use]
    pub fn new(
        default_lhs: &str,
        initial_expr: &str,
        presets: Vec<(&'static str, &'static str)>,
    ) -> Self {
        let mut bar = Self {
            raw_text: initial_expr.to_string(),
            default_lhs: default_lhs.to_string(),
            lhs_name: None,
            parsed_expr: Err(ParseError::EmptyExpression),
            latex_cache: String::new(),
            presets,
        };
        bar.reparse();
        bar
    }

    /// Updates the expression text and reparses the AST.
    pub fn set_expression(&mut self, text: &str) {
        self.raw_text = text.to_string();
        self.reparse();
    }

    /// Returns a reference to the parsed AST if valid.
    #[must_use]
    pub fn expr(&self) -> Option<&Expr> {
        self.parsed_expr.as_ref().ok()
    }

    /// Returns `true` if current input represents a valid mathematical expression.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.parsed_expr.is_ok()
    }

    /// Returns the current syntax error message, if any.
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        self.parsed_expr.as_ref().err().map(|e| e.to_string())
    }

    /// Internal method to reparse raw text into AST and update LaTeX cache.
    pub fn reparse(&mut self) {
        let trimmed = self.raw_text.trim();
        if trimmed.is_empty() {
            self.parsed_expr = Err(ParseError::EmptyExpression);
            self.latex_cache.clear();
            self.lhs_name = None;
            return;
        }

        match parse_equation(trimmed) {
            Ok((lhs, expr)) => {
                let lhs_label = lhs.as_deref().or_else(|| {
                    if self.default_lhs.is_empty() {
                        None
                    } else {
                        Some(&self.default_lhs)
                    }
                });

                let expr_latex = expr.to_latex();
                self.latex_cache = match lhs_label {
                    Some(prefix) => format!("{prefix} = {expr_latex}"),
                    None => expr_latex,
                };
                self.lhs_name = lhs;
                self.parsed_expr = Ok(expr);
            }
            Err(err) => {
                self.parsed_expr = Err(err);
                self.latex_cache.clear();
            }
        }
    }

    /// Renders the math expression input bar UI.
    ///
    /// Returns `true` if the expression was modified by the user or preset selection.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;

        ui.group(|ui| {
            ui.vertical(|ui| {
                if let Some(preset_str) = render_presets(ui, &self.presets) {
                    self.set_expression(preset_str);
                    changed = true;
                }

                ui.horizontal(|ui| {
                    if !self.default_lhs.is_empty() && !self.raw_text.contains('=') {
                        ui.label(format!("{} =", self.default_lhs));
                    }
                    let res = ui.add(
                        egui::TextEdit::singleline(&mut self.raw_text)
                            .hint_text("Enter mathematical expression...")
                            .desired_width(ui.available_width() - 8.0),
                    );
                    if res.changed() {
                        self.reparse();
                        changed = true;
                    }
                });

                ui.add_space(4.0);
                render_status_row(ui, &self.parsed_expr, &self.latex_cache, &self.raw_text);
            });
        });

        changed
    }
}

fn render_presets(
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

fn render_status_row(ui: &mut egui::Ui, parsed: &Result<Expr, ParseError>, latex: &str, raw: &str) {
    match parsed {
        Ok(_) => {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Preview:").weak());
                let preview_str = if !latex.is_empty() { latex } else { raw };
                egui_plot::render_math_label(
                    ui,
                    preview_str,
                    egui::TextStyle::Body,
                    ui.visuals().text_color(),
                );
            });
        }
        Err(err) => {
            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new(format!("⚠️ Syntax Error: {err}"))
                        .color(egui::Color32::from_rgb(220, 50, 50))
                        .small(),
                );
            });
        }
    }
}
