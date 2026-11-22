//! Reusable egui mathematical expression input bar with live syntax validation, LaTeX preview,
//! coordinate scope validation, `CompiledExpr` bytecode caching, and `AnimatedSlider` synthesis.

pub mod params;

#[cfg(test)]
mod tests;

pub use params::{ParameterSliderBank, MAX_SYNTHESIZED_PARAMS};

use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use math_commons::ast::{parse_equation, CompiledExpr, Expr, ParseError};
use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;

/// Reusable input bar component for dynamic mathematical equations.
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
    /// Tracks whether an error was previously present so correction can be announced.
    pub had_error: bool,
    /// Last announced error message to prevent duplicate announcements on static frames.
    pub last_announced_error: Option<String>,
    /// Coordinate axis names bound to leading bytecode slots (e.g. `["x"]` or `["x", "y"]`).
    axes: Vec<String>,
    /// Whether coordinate scope & parameter slider synthesis are enabled.
    scope_enabled: bool,
    /// Whether `i` / `I` compiles as the imaginary unit (`Complex64::i()`).
    complex_mode: bool,
    /// Whether inline `AnimatedSlider` controls should be rendered inside [`Self::ui`].
    render_inline_sliders: bool,
    /// Semantic/arity/parameter-limit error message when syntax parsed but compilation failed.
    validation_error: Option<String>,
    /// Currently valid compiled bytecode (`None` when current text has an error).
    compiled_expr: Option<CompiledExpr>,
    /// Last valid compiled bytecode preserved across transient syntax edits.
    last_valid_compiled: Option<CompiledExpr>,
    /// Auto-synthesized free parameter sliders synchronized with `last_valid_compiled`.
    slider_bank: ParameterSliderBank,
}

impl MathExpressionInputBar {
    /// Creates a new mathematical expression input bar (legacy constructor compatible with ODE tools).
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
            had_error: false,
            last_announced_error: if initial_expr.trim().is_empty() {
                Some(format!(
                    "Syntax Error: {}",
                    ParseError::EmptyExpression.user_message()
                ))
            } else {
                None
            },
            axes: Vec::new(),
            scope_enabled: false,
            complex_mode: false,
            render_inline_sliders: true,
            validation_error: None,
            compiled_expr: None,
            last_valid_compiled: None,
            slider_bank: ParameterSliderBank::new(),
        };
        bar.reparse();
        bar
    }

    /// Configures coordinate axis slots (e.g., `&["x"]`, `&["x", "y"]`, `&["x", "w", "b"]`) and
    /// activates `CompiledExpr` bytecode compilation + free parameter `AnimatedSlider` discovery.
    #[must_use]
    pub fn with_axes(mut self, axes: &[&str]) -> Self {
        self.axes = axes.iter().map(|&s| s.to_string()).collect();
        self.scope_enabled = true;
        self.reparse();
        self
    }

    /// Configures complex evaluation mode (`i` / `I` treated as imaginary unit rather than parameter).
    #[must_use]
    pub fn with_complex_mode(mut self, complex: bool) -> Self {
        self.complex_mode = complex;
        self.scope_enabled = true;
        self.reparse();
        self
    }

    /// Configures whether discovered free parameter sliders render inline inside [`Self::ui`].
    #[must_use]
    pub fn with_inline_sliders(mut self, inline: bool) -> Self {
        self.render_inline_sliders = inline;
        self
    }

    /// Configures the range and default value for synthesized parameter sliders.
    #[must_use]
    pub fn with_param_range(mut self, range: RangeInclusive<f64>, default_value: f64) -> Self {
        self.slider_bank = self.slider_bank.with_range(range, default_value);
        self
    }

    /// Updates the expression text and reparses/recompiles the AST.
    pub fn set_expression(&mut self, text: &str) {
        self.raw_text = text.to_string();
        self.reparse();
    }

    /// Returns a reference to the parsed AST if currently valid.
    #[must_use]
    pub fn expr(&self) -> Option<&Expr> {
        if self.validation_error.is_some() {
            None
        } else {
            self.parsed_expr.as_ref().ok()
        }
    }

    /// Returns the currently valid compiled bytecode (`None` if current input has an error).
    #[must_use]
    pub const fn compiled(&self) -> Option<&CompiledExpr> {
        self.compiled_expr.as_ref()
    }

    /// Returns the active compiled bytecode, falling back to the last valid formula during edits.
    #[must_use]
    pub fn active_compiled(&self) -> Option<&CompiledExpr> {
        self.compiled_expr
            .as_ref()
            .or(self.last_valid_compiled.as_ref())
    }

    /// Returns the inline slice of parameter values synchronized with [`Self::active_compiled`].
    #[must_use]
    pub fn param_slice(&self) -> &[f64] {
        self.slider_bank.param_slice()
    }

    /// Exports active parameter values for JSON preset and `#state=` URL serialization.
    #[must_use]
    pub fn params_map(&self) -> BTreeMap<String, f64> {
        self.slider_bank.export_active_map()
    }

    /// Restores parameter values from a deserialized map.
    pub fn set_params_map(&mut self, map: &BTreeMap<String, f64>) {
        self.slider_bank.set_params_map(map);
    }

    /// Returns `true` if current input represents a valid, compilable mathematical expression.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.parsed_expr.is_ok() && self.validation_error.is_none()
    }

    /// Returns the current syntax or scope/arity error message, if any.
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        if let Some(err) = &self.validation_error {
            return Some(err.clone());
        }
        self.parsed_expr
            .as_ref()
            .err()
            .map(|e| e.user_message_for_input(self.raw_text.trim()))
    }

    /// Reparses `raw_text`, compiles bytecode, validates arity/scope, and syncs parameter sliders.
    pub fn reparse(&mut self) {
        let trimmed = self.raw_text.trim();
        if trimmed.is_empty() {
            self.parsed_expr = Err(ParseError::EmptyExpression);
            self.validation_error = None;
            self.compiled_expr = None;
            self.latex_cache.clear();
            self.lhs_name = None;
            return;
        }

        match parse_equation(trimmed) {
            Ok((lhs, expr)) => self.finish_valid_parse(lhs, expr),
            Err(err) => {
                self.parsed_expr = Err(err);
                self.validation_error = None;
                self.compiled_expr = None;
                self.latex_cache.clear();
            }
        }
    }

    fn finish_valid_parse(&mut self, lhs: Option<String>, expr: Expr) {
        let axis_refs: Vec<&str> = self.axes.iter().map(String::as_str).collect();
        let compile_res = if self.complex_mode {
            CompiledExpr::compile_complex(&expr, &axis_refs)
        } else {
            CompiledExpr::compile_with_vars(&expr, &axis_refs)
        };

        match compile_res {
            Ok(compiled) => {
                if self.scope_enabled && compiled.param_names().len() > MAX_SYNTHESIZED_PARAMS {
                    self.validation_error = Some(format!(
                        "Formula defines {} free parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
                        compiled.param_names().len()
                    ));
                    self.compiled_expr = None;
                    self.parsed_expr = Ok(expr);
                    return;
                }
                if self.scope_enabled {
                    self.slider_bank.sync_names(compiled.param_names());
                }
                update_latex_preview(
                    &mut self.latex_cache,
                    lhs.as_deref(),
                    &self.default_lhs,
                    &expr,
                );
                self.lhs_name = lhs;
                self.validation_error = None;
                self.last_valid_compiled = Some(compiled.clone());
                self.compiled_expr = Some(compiled);
                self.parsed_expr = Ok(expr);
            }
            Err(eval_err) => {
                self.validation_error = Some(eval_err.to_string());
                self.compiled_expr = None;
                self.latex_cache.clear();
                self.parsed_expr = Ok(expr);
            }
        }
    }

    /// Renders the math expression input bar UI and optional synthesized `AnimatedSlider`s.
    ///
    /// Returns `true` if the expression or any synthesized parameter slider changed.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let slider_params: HashMap<String, f64> = if self.scope_enabled {
            self.slider_bank.export_active_map().into_iter().collect()
        } else {
            HashMap::new()
        };
        self.ui_with_params(ui, &slider_params)
    }

    /// Renders the math expression input bar UI with live parameter-calibrated LaTeX preview and code copy/save actions.
    ///
    /// Returns `true` if the expression or any synthesized parameter slider changed.
    pub fn ui_with_params(&mut self, ui: &mut egui::Ui, params: &HashMap<String, f64>) -> bool {
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
                            .desired_width((ui.available_width() - 8.0).max(48.0)),
                    );
                    if res.changed() {
                        self.reparse();
                        changed = true;
                    }
                });

                ui.add_space(4.0);
                let default_prefix =
                    (!self.default_lhs.is_empty()).then_some(self.default_lhs.as_str());
                let lhs_prefix = self.lhs_name.as_deref().or(default_prefix);
                let err_detail = self.error_message();
                let valid_expr = if self.validation_error.is_none() {
                    self.parsed_expr.as_ref().ok()
                } else {
                    None
                };
                render_status_row(
                    ui,
                    (err_detail.as_deref(), valid_expr, lhs_prefix, params),
                    (&self.latex_cache, &self.raw_text),
                    &mut self.had_error,
                    &mut self.last_announced_error,
                );

                if self.scope_enabled && self.render_inline_sliders && self.slider_bank.ui(ui) {
                    changed = true;
                }
            });
        });

        changed
    }
}

fn update_latex_preview(
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

fn render_status_row(
    ui: &mut egui::Ui,
    (err_detail, valid_expr, lhs_prefix, params): (
        Option<&str>,
        Option<&Expr>,
        Option<&str>,
        &HashMap<String, f64>,
    ),
    (latex, raw): (&str, &str),
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
            render_expr_export_actions(ui, expr, &calibrated, params);
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
) {
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
        if let Ok(rs) = expr.to_rust_expr(params) {
            copy_expr_snippet(ui, &rs, "Copied Rust expression to clipboard");
        }
    }
    if ui
        .small_button("📋 Python")
        .accessible_hover_text("Copy Python numpy expression to clipboard")
        .clicked()
    {
        if let Ok(py) = expr.to_python_expr(params) {
            copy_expr_snippet(ui, &py, "Copied Python expression to clipboard");
        }
    }
    if ui
        .small_button("💾 .tex")
        .accessible_hover_text("Save LaTeX expression to .tex file")
        .clicked()
    {
        #[cfg(not(test))]
        egui_plot::file_io::save_file_dialog(
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
