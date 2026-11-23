//! Coupled multi-equation expression bank sharing a single [`ParameterSliderBank`]
//! and synchronized [`CompiledExpr`] parameter indices across $N$ equations.

use super::{MathExpressionInputBar, ParameterSliderBank, MAX_SYNTHESIZED_PARAMS};
use eframe::egui;
use math_commons::ast::{is_universal_constant, parse, CompiledExpr, Expr};
use std::collections::BTreeMap;
use std::ops::RangeInclusive;

/// Multi-equation input bank where every equation shares the same coordinate axes,
/// one deduplicated [`ParameterSliderBank`], and aligned [`CompiledExpr`] parameter slots.
///
/// During transient syntax edits or parameter-cap violations on any bar, [`Self::active_compiled`]
/// and [`Self::param_slice`] retain the last valid synchronized bytecode and parameter values.
#[derive(Clone, Debug)]
pub struct CoupledExpressionBank {
    bars: Vec<MathExpressionInputBar>,
    axes: Vec<String>,
    slider_bank: ParameterSliderBank,
    active_compiled: Vec<CompiledExpr>,
    bank_error: Option<String>,
    system_presets: Vec<(&'static str, Vec<&'static str>)>,
}

impl CoupledExpressionBank {
    /// Creates a coupled bank over `axes` from `(lhs_label, initial_expr)` equation definitions.
    #[must_use]
    pub fn new(axes: &[&str], equations: &[(&str, &str)]) -> Self {
        let bars: Vec<MathExpressionInputBar> = equations
            .iter()
            .map(|(lhs, expr)| {
                MathExpressionInputBar::new(lhs, expr, Vec::new())
                    .with_axes(axes)
                    .with_inline_sliders(false)
            })
            .collect();
        let fallback = build_zero_fallbacks(axes, equations.len());
        let mut bank = Self {
            bars,
            axes: axes.iter().map(|&s| s.to_string()).collect(),
            slider_bank: ParameterSliderBank::new(),
            active_compiled: fallback,
            bank_error: None,
            system_presets: Vec::new(),
        };
        bank.recompile();
        bank
    }

    /// Configures range bounds and default value for newly discovered shared parameters.
    #[must_use]
    pub fn with_param_range(mut self, range: RangeInclusive<f64>, default_value: f64) -> Self {
        self.slider_bank = self.slider_bank.with_range(range, default_value);
        self
    }

    /// Attaches multi-equation system presets `(preset_label, &[expr_0, expr_1, ...])`.
    #[must_use]
    pub fn with_system_presets(mut self, presets: Vec<(&'static str, Vec<&'static str>)>) -> Self {
        self.system_presets = presets;
        self
    }

    /// Seeds initial parameter values without overwriting expressions.
    #[must_use]
    pub fn with_initial_params(mut self, defaults: &[(&str, f64)]) -> Self {
        let mut map = self.slider_bank.export_active_map();
        for &(k, v) in defaults {
            map.insert(k.to_string(), v);
        }
        self.slider_bank.set_params_map(&map);
        self
    }

    /// Number of coupled equations in this bank.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Returns `true` if the bank holds zero equations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }

    /// Immutable access to the individual equation input bars.
    #[must_use]
    pub fn bars(&self) -> &[MathExpressionInputBar] {
        &self.bars
    }

    /// Returns raw expression strings across all bars in order.
    #[must_use]
    pub fn expressions(&self) -> Vec<String> {
        self.bars.iter().map(|b| b.raw_text.clone()).collect()
    }

    /// Returns the raw expression string at slot `idx` (or `""` if out of bounds).
    #[must_use]
    pub fn raw_text(&self, idx: usize) -> &str {
        self.bars.get(idx).map_or("", |b| b.raw_text.as_str())
    }

    /// Updates a single equation slot (`idx`) and recompiles synchronized bytecode.
    pub fn set_expression(&mut self, idx: usize, expr: &str) {
        if let Some(bar) = self.bars.get_mut(idx) {
            bar.set_expression(expr);
            self.recompile();
        }
    }

    /// Updates all equation slots simultaneously and recompiles synchronized bytecode.
    pub fn set_expressions<S: AsRef<str>>(&mut self, exprs: &[S]) {
        for (bar, expr) in self.bars.iter_mut().zip(exprs.iter()) {
            bar.set_expression(expr.as_ref());
        }
        self.recompile();
    }

    /// Restores expressions and shared parameter values from a preset or `#state=` payload.
    pub fn load_state<S: AsRef<str>>(&mut self, exprs: &[S], params: &BTreeMap<String, f64>) {
        for (bar, expr) in self.bars.iter_mut().zip(exprs.iter()) {
            bar.set_expression(expr.as_ref());
        }
        self.slider_bank.set_params_map(params);
        self.recompile();
        self.slider_bank.set_params_map(params);
    }

    /// Returns `true` when every bar has valid syntax and the bank is within parameter limits.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.bank_error.is_none() && self.bars.iter().all(MathExpressionInputBar::is_valid)
    }

    /// Returns the bank-level validation error (e.g., exceeding 6 shared free parameters), if any.
    #[must_use]
    pub fn bank_error(&self) -> Option<&str> {
        self.bank_error.as_deref()
    }

    /// Returns the first syntax or bank-level validation error across all slots, if any.
    #[must_use]
    pub fn first_error(&self) -> Option<String> {
        if let Some(ref err) = self.bank_error {
            return Some(err.clone());
        }
        self.bars
            .iter()
            .find_map(MathExpressionInputBar::error_message)
    }

    /// Synchronized last-valid [`CompiledExpr`] slice aligned with [`Self::param_slice`].
    #[must_use]
    pub fn active_compiled(&self) -> &[CompiledExpr] {
        &self.active_compiled
    }

    /// Active shared parameter values aligned with [`Self::active_compiled`].
    #[must_use]
    pub fn param_slice(&self) -> &[f64] {
        self.slider_bank.param_slice()
    }

    /// Copies active parameter values into a fixed stack array `(values, len)` for worker jobs.
    #[must_use]
    pub fn inline_params(&self) -> ([f64; MAX_SYNTHESIZED_PARAMS], usize) {
        let slice = self.slider_bank.param_slice();
        let mut out = [0.0; MAX_SYNTHESIZED_PARAMS];
        let len = slice.len().min(MAX_SYNTHESIZED_PARAMS);
        out[..len].copy_from_slice(&slice[..len]);
        (out, len)
    }

    /// Ordered names of discovered free parameters shared across the bank.
    #[must_use]
    pub fn param_names(&self) -> &[String] {
        self.slider_bank.active_names()
    }

    /// Exports active parameter values for compact JSON / `#state=` serialization.
    #[must_use]
    pub fn params_map(&self) -> BTreeMap<String, f64> {
        self.slider_bank.export_active_map()
    }

    /// Replaces parameter slider values from a deserialized map.
    pub fn set_params_map(&mut self, map: &BTreeMap<String, f64>) {
        self.slider_bank.set_params_map(map);
    }

    /// Updates a single named parameter value in the shared slider bank.
    pub fn set_param(&mut self, name: &str, value: f64) {
        let mut map = self.slider_bank.export_active_map();
        map.insert(name.to_string(), value);
        self.slider_bank.set_params_map(&map);
    }

    /// Recompiles all bars against their deduplicated union of free parameters.
    ///
    /// Preserves `active_compiled` and `slider_bank` unchanged if any bar has a syntax error
    /// or if the union of free parameters exceeds [`MAX_SYNTHESIZED_PARAMS`].
    pub fn recompile(&mut self) {
        let mut parsed: Vec<&Expr> = Vec::with_capacity(self.bars.len());
        for bar in &self.bars {
            match &bar.parsed_expr {
                Ok(expr) => parsed.push(expr),
                Err(_) => return,
            }
        }

        let discovered = collect_union_params(&parsed, &self.axes);
        if discovered.len() > MAX_SYNTHESIZED_PARAMS {
            self.bank_error = Some(format!(
                "Coupled equations define {} free parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
                discovered.len()
            ));
            return;
        }

        let axis_refs: Vec<&str> = self.axes.iter().map(String::as_str).collect();
        let param_refs: Vec<&str> = discovered.iter().map(String::as_str).collect();
        let mut next_compiled = Vec::with_capacity(parsed.len());
        for expr in parsed {
            match CompiledExpr::compile_with_vars_and_params(expr, &axis_refs, &param_refs) {
                Ok(c) => next_compiled.push(c),
                Err(err) => {
                    self.bank_error = Some(err.to_string());
                    return;
                }
            }
        }

        self.bank_error = None;
        self.slider_bank.sync_names(&discovered);
        self.active_compiled = next_compiled;
    }

    /// Renders system preset selector (if any), all equation input bars, and bank-level errors
    /// under `id_salt` without rendering the shared parameter sliders.
    pub fn ui_inputs(&mut self, ui: &mut egui::Ui, id_salt: &str) -> bool {
        ui.push_id(id_salt, |ui| {
            let mut changed = false;
            if let Some(exprs) = render_system_presets(ui, &self.system_presets) {
                self.set_expressions(&exprs);
                changed = true;
            }

            let shared_params: std::collections::HashMap<String, f64> =
                self.slider_bank.export_active_map().into_iter().collect();
            let mut bar_edited = false;
            for (idx, bar) in self.bars.iter_mut().enumerate() {
                ui.push_id(("coupled_expr_slot", idx), |ui| {
                    if bar.ui_with_params(ui, &shared_params) {
                        bar_edited = true;
                    }
                });
            }
            if bar_edited {
                self.recompile();
                changed = true;
            }

            if let Some(ref err) = self.bank_error {
                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                ui.colored_label(palette.error, format!("⚠️ {err}"));
            }
            changed
        })
        .inner
    }

    /// Renders only the shared [`ParameterSliderBank`] under `id_salt`.
    pub fn sliders_ui(&mut self, ui: &mut egui::Ui, id_salt: &str) -> bool {
        ui.push_id(id_salt, |ui| self.slider_bank.ui(ui)).inner
    }

    /// Renders system preset selector (if any), all equation input bars, bank errors,
    /// and the shared [`ParameterSliderBank`]. Returns `true` on any change.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = self.ui_inputs(ui, "coupled_bank_inputs");
        changed |= self.sliders_ui(ui, "coupled_bank_sliders");
        changed
    }
}

fn collect_union_params(exprs: &[&Expr], axes: &[String]) -> Vec<String> {
    let mut discovered = Vec::new();
    for expr in exprs {
        for var in expr.free_variables() {
            let is_axis = axes.iter().any(|ax| ax.eq_ignore_ascii_case(&var));
            if !is_axis
                && !is_universal_constant(&var)
                && !discovered.iter().any(|existing| existing == &var)
            {
                discovered.push(var);
            }
        }
    }
    discovered
}

fn build_zero_fallbacks(axes: &[&str], count: usize) -> Vec<CompiledExpr> {
    let zero = parse("0").expect("constant 0 parses");
    let c0 = CompiledExpr::compile_with_vars(&zero, axes).expect("constant 0 compiles");
    vec![c0; count]
}

fn render_system_presets(
    ui: &mut egui::Ui,
    presets: &[(&'static str, Vec<&'static str>)],
) -> Option<Vec<&'static str>> {
    if presets.is_empty() {
        return None;
    }
    let mut chosen = None;
    ui.horizontal(|ui| {
        ui.label("System Preset:");
        egui::ComboBox::from_id_salt(ui.id().with("coupled_system_presets"))
            .selected_text("Select Coupled System...")
            .show_ui(ui, |ui| {
                for (name, exprs) in presets {
                    if ui.button(*name).clicked() {
                        chosen = Some(exprs.clone());
                        ui.close();
                    }
                }
            });
    });
    ui.add_space(4.0);
    chosen
}
