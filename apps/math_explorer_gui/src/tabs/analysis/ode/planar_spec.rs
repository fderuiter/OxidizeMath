// [cite:pure_math]
// [cite:math_commons]

//! Coordinate-scoped `(t, x, y)` planar input bars and shared [`PlanarCustomBank`]
//! bytecode/slider compilation for `OdePreset::PlanarCustom`.

use super::custom_spec::CustomOdeSpec;
use super::default_scalar_ode_input;
use super::phase_plane::{DEFAULT_PLANAR_X_EXPR, DEFAULT_PLANAR_Y_EXPR, PLANAR_COUPLED_PRESETS};
use crate::framework::math_input::{
    MathExpressionInputBar, ParameterSliderBank, MAX_COUPLED_BANK_INSTRUCTIONS,
    MAX_SYNTHESIZED_PARAMS,
};
use eframe::egui;
use math_commons::ast::{is_universal_constant, parse, CompiledExpr, Expr};
use std::collections::{BTreeMap, HashMap};
use std::ops::RangeInclusive;

const PLANAR_AXES: &[&str] = &["t", "x", "y"];

const PLANAR_DISALLOWED_VARS: &[(&str, &str)] = &[
    ("z", "allowed variables: t, x, y"),
    (
        "v",
        "allowed variables: t, x, y; use y for the second planar state or Coupled (y', v') mode",
    ),
];

fn configure_planar_bar(bar: MathExpressionInputBar) -> MathExpressionInputBar {
    bar.with_axes(PLANAR_AXES)
        .with_inline_sliders(false)
        .with_param_range(-5.0..=5.0, 1.0)
        .with_disallowed_vars(PLANAR_DISALLOWED_VARS)
        .with_param_spec("k", "k (Coupling / Damping)", -5.0..=5.0, 1.0)
        .with_param_spec("r", "r (Growth / Forcing)", -5.0..=5.0, 1.0)
        .with_param_spec("cap_k", "K (Carrying Capacity)", 1.0..=100.0, 10.0)
        .with_param_spec("K", "K (Carrying Capacity)", 1.0..=100.0, 10.0)
}

fn build_planar_slider_bank() -> ParameterSliderBank {
    let mut bank = ParameterSliderBank::new().with_range(-5.0..=5.0, 1.0);
    bank.register_param_spec("k", "k (Coupling / Damping)", -5.0..=5.0, 1.0);
    bank.register_param_spec("r", "r (Growth / Forcing)", -5.0..=5.0, 1.0);
    bank.register_param_spec("cap_k", "K (Carrying Capacity)", 1.0..=100.0, 10.0);
    bank.register_param_spec("K", "K (Carrying Capacity)", 1.0..=100.0, 10.0);
    bank
}

/// Constructs default coordinate-scoped input bars `(scalar_y, planar_x, planar_y)` for `OdeSolverTool`.
#[must_use]
pub(crate) fn default_ode_math_inputs() -> (
    MathExpressionInputBar,
    MathExpressionInputBar,
    MathExpressionInputBar,
) {
    let scalar = default_scalar_ode_input();
    let planar_x = configure_planar_bar(MathExpressionInputBar::new(
        "x'",
        DEFAULT_PLANAR_X_EXPR,
        vec![
            ("x' = y (Second-Order Velocity)", "y"),
            ("x' = r*x - k*x*y (Lotka-Volterra)", "r * x - k * x * y"),
            ("x' = y - 0.2*x (Damped Rotation)", "y - 0.2 * x"),
        ],
    ));
    let planar_y = configure_planar_bar(MathExpressionInputBar::new(
        "y'",
        DEFAULT_PLANAR_Y_EXPR,
        vec![
            (
                "y' = k*(1 - x^2)*y - x (Van der Pol)",
                "k * (1 - x^2) * y - x",
            ),
            ("y' = k*x*y - y (Lotka-Volterra)", "k * x * y - y"),
            (
                "y' = -k*sin(x) - 0.25*y (Pendulum)",
                "-k * sin(x) - 0.25 * y",
            ),
            ("y' = x - x^3 - 0.3*k*y (Duffing)", "x - x^3 - 0.3 * k * y"),
        ],
    ));
    (scalar, planar_x, planar_y)
}

fn collect_planar_params(ex: &Expr, ey: &Expr) -> Vec<String> {
    let mut out = Vec::new();
    for expr in [ex, ey] {
        for var in expr.free_variables() {
            let is_axis = PLANAR_AXES.iter().any(|ax| ax.eq_ignore_ascii_case(&var));
            if !is_axis && !is_universal_constant(&var) && !out.iter().any(|p| p == &var) {
                out.push(var);
            }
        }
    }
    out
}

fn compile_planar_candidate(
    ex: &Expr,
    ey: &Expr,
) -> Result<(CompiledExpr, CompiledExpr, Vec<String>), String> {
    let discovered = collect_planar_params(ex, ey);
    if discovered.len() > MAX_SYNTHESIZED_PARAMS {
        return Err(format!(
            "Planar equations define {} free parameters (maximum {MAX_SYNTHESIZED_PARAMS})",
            discovered.len()
        ));
    }
    let param_refs: Vec<&str> = discovered.iter().map(String::as_str).collect();
    let cx = CompiledExpr::compile_with_vars_and_params(ex, PLANAR_AXES, &param_refs)
        .map_err(|e| e.to_string())?;
    let cy = CompiledExpr::compile_with_vars_and_params(ey, PLANAR_AXES, &param_refs)
        .map_err(|e| e.to_string())?;
    let total = cx.instruction_count() + cy.instruction_count();
    if total > MAX_COUPLED_BANK_INSTRUCTIONS {
        return Err(format!(
            "Planar equations compile to {total} total instructions (maximum {MAX_COUPLED_BANK_INSTRUCTIONS}). Simplify one or more equations"
        ));
    }
    Ok((cx, cy, discovered))
}

fn zero_planar_expr() -> CompiledExpr {
    let zero = parse("0").expect("constant 0 parses");
    CompiledExpr::compile_with_vars(&zero, PLANAR_AXES).expect("constant 0 compiles")
}

/// Shared free-parameter [`ParameterSliderBank`] and synchronized `(t, x, y)` bytecode compiler
/// for `OdePreset::PlanarCustom`.
#[derive(Clone, Debug)]
pub(crate) struct PlanarCustomBank {
    slider_bank: ParameterSliderBank,
    active_dx: CompiledExpr,
    active_dy: CompiledExpr,
    synced_texts: (String, String),
    bank_error: Option<String>,
    had_bank_error: bool,
    last_announced_bank_error: Option<String>,
    remembered_values: BTreeMap<String, f64>,
}

impl PlanarCustomBank {
    #[must_use]
    pub(crate) fn new(px: &MathExpressionInputBar, py: &MathExpressionInputBar) -> Self {
        let z = zero_planar_expr();
        let mut bank = Self {
            slider_bank: build_planar_slider_bank(),
            active_dx: z.clone(),
            active_dy: z,
            synced_texts: (String::new(), String::new()),
            bank_error: None,
            had_bank_error: false,
            last_announced_bank_error: None,
            remembered_values: BTreeMap::from([
                ("k".to_string(), 1.0),
                ("r".to_string(), 1.0),
                ("cap_k".to_string(), 10.0),
                ("K".to_string(), 10.0),
            ]),
        };
        bank.recompile(px, py);
        bank
    }

    #[must_use]
    pub(crate) fn is_synced(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
    ) -> bool {
        self.synced_texts.0 == px.raw_text && self.synced_texts.1 == py.raw_text
    }

    pub(crate) fn recompile(&mut self, px: &MathExpressionInputBar, py: &MathExpressionInputBar) {
        self.synced_texts = (px.raw_text.clone(), py.raw_text.clone());
        let (Some(ex), Some(ey)) = (px.expr(), py.expr()) else {
            self.bank_error = None;
            return;
        };
        match compile_planar_candidate(ex, ey) {
            Ok((cx, cy, discovered)) => {
                self.bank_error = None;
                self.slider_bank.sync_names(&discovered);
                for (k, v) in self.slider_bank.export_active_map() {
                    self.remembered_values.insert(k, v);
                }
                self.active_dx = cx;
                self.active_dy = cy;
            }
            Err(err) => {
                self.bank_error = Some(err);
            }
        }
    }

    #[must_use]
    pub(crate) fn is_valid(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
    ) -> bool {
        if !px.is_valid() || !py.is_valid() {
            return false;
        }
        if self.is_synced(px, py) {
            self.bank_error.is_none()
        } else {
            match (px.expr(), py.expr()) {
                (Some(ex), Some(ey)) => compile_planar_candidate(ex, ey).is_ok(),
                _ => false,
            }
        }
    }

    #[must_use]
    pub(crate) fn first_error(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
    ) -> Option<String> {
        if let Some(err) = px.error_message().or_else(|| py.error_message()) {
            return Some(err);
        }
        if self.is_synced(px, py) {
            self.bank_error.clone()
        } else {
            let (ex, ey) = (px.expr()?, py.expr()?);
            compile_planar_candidate(ex, ey).err()
        }
    }

    #[must_use]
    pub(crate) fn total_instruction_count(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
    ) -> usize {
        let (dx, dy) = self.resolved_compiled(px, py);
        dx.instruction_count() + dy.instruction_count()
    }

    #[must_use]
    pub(crate) fn param_names(&self) -> &[String] {
        self.slider_bank.active_names()
    }

    #[must_use]
    pub(crate) fn param_slice(&self) -> &[f64] {
        self.slider_bank.param_slice()
    }

    #[must_use]
    pub(crate) fn active_value(&self, name: &str) -> Option<f64> {
        self.slider_bank.active_value(name)
    }

    #[must_use]
    pub(crate) fn params_map(&self) -> BTreeMap<String, f64> {
        self.slider_bank.export_active_map()
    }

    pub(crate) fn set_param(&mut self, name: &str, value: f64) {
        self.slider_bank.set_param(name, value);
        if value.is_finite() {
            self.remembered_values
                .insert(name.to_string(), value.clamp(-1e4, 1e4));
        }
    }

    pub(crate) fn set_params_map(&mut self, map: &BTreeMap<String, f64>) {
        self.remembered_values.clear();
        for (k, &v) in map {
            if v.is_finite() {
                self.remembered_values.insert(k.clone(), v.clamp(-1e4, 1e4));
            }
        }
        self.slider_bank.set_params_map(map);
    }

    pub(crate) fn load_state(
        &mut self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
        params: &BTreeMap<String, f64>,
    ) {
        self.set_params_map(params);
        self.recompile(px, py);
        self.slider_bank.set_params_map(params);
        for (k, v) in self.slider_bank.export_active_map() {
            self.remembered_values.insert(k, v);
        }
    }

    pub(crate) fn clamp_active_params(&mut self, range: RangeInclusive<f64>) {
        self.slider_bank.clamp_active_values(range);
        for (k, v) in self.slider_bank.export_active_map() {
            self.remembered_values.insert(k, v);
        }
    }

    fn resolved_compiled(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
    ) -> (CompiledExpr, CompiledExpr) {
        if !self.is_synced(px, py) {
            if let (Some(ex), Some(ey)) = (px.expr(), py.expr()) {
                if let Ok((cx, cy, _)) = compile_planar_candidate(ex, ey) {
                    return (cx, cy);
                }
            }
        }
        (self.active_dx.clone(), self.active_dy.clone())
    }

    fn resolve_slot_value(&self, name: &str, legacy: [f64; 3], synced_legacy: [f64; 3]) -> f64 {
        let [param_k, param_r, param_cap_k] = legacy;
        if name == "k" && param_k.is_finite() && param_k.to_bits() != synced_legacy[0].to_bits() {
            return param_k;
        }
        if name == "r" && param_r.is_finite() && param_r.to_bits() != synced_legacy[1].to_bits() {
            return param_r;
        }
        if matches!(name, "cap_k" | "K")
            && param_cap_k.is_finite()
            && param_cap_k.to_bits() != synced_legacy[2].to_bits()
        {
            return param_cap_k;
        }
        self.slider_bank
            .active_value(name)
            .or_else(|| self.remembered_values.get(name).copied())
            .unwrap_or(if matches!(name, "cap_k" | "K") {
                10.0
            } else {
                1.0
            })
    }

    #[must_use]
    pub(crate) fn snapshot_compiled_and_params(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
        legacy: [f64; 3],
        synced_legacy: [f64; 3],
    ) -> (
        CompiledExpr,
        CompiledExpr,
        [f64; MAX_SYNTHESIZED_PARAMS],
        usize,
    ) {
        let (dx, dy) = self.resolved_compiled(px, py);
        let len = dx.param_names().len().min(MAX_SYNTHESIZED_PARAMS);
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        for (idx, name) in dx.param_names().iter().take(len).enumerate() {
            params[idx] = self.resolve_slot_value(name, legacy, synced_legacy);
        }
        (dx, dy, params, len)
    }

    #[must_use]
    pub(crate) fn to_coupled_spec(
        &self,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
        legacy: [f64; 3],
        synced_legacy: [f64; 3],
    ) -> CustomOdeSpec {
        let (dy, dv, params, param_len) =
            self.snapshot_compiled_and_params(px, py, legacy, synced_legacy);
        CustomOdeSpec::Coupled {
            dy,
            dv,
            params,
            param_len,
        }
    }

    pub(crate) fn ui_editor(
        &mut self,
        ui: &mut egui::Ui,
        px: &mut MathExpressionInputBar,
        py: &mut MathExpressionInputBar,
    ) -> bool {
        if !self.is_synced(px, py) {
            self.recompile(px, py);
        }
        let mut changed = false;
        ui.add_space(6.0);
        egui::ComboBox::from_id_salt("ode_planar_coupled_preset")
            .selected_text("Coupled System Templates...")
            .show_ui(ui, |ui| {
                for (title, fx, fy) in PLANAR_COUPLED_PRESETS {
                    if ui.button(*title).clicked() {
                        px.set_expression(fx);
                        py.set_expression(fy);
                        self.recompile(px, py);
                        changed = true;
                        ui.close();
                    }
                }
            });

        let shared_params: HashMap<String, f64> =
            self.slider_bank.export_active_map().into_iter().collect();
        let mut edited = false;
        ui.add_space(4.0);
        ui.push_id("ode_planar_x_bar", |ui| {
            edited |= px.ui_with_params(ui, &shared_params);
        });
        ui.add_space(4.0);
        ui.push_id("ode_planar_y_bar", |ui| {
            edited |= py.ui_with_params(ui, &shared_params);
        });
        if edited {
            self.recompile(px, py);
            changed = true;
        }
        self.render_bank_error(ui, px, py);
        changed
    }

    fn render_bank_error(
        &mut self,
        ui: &mut egui::Ui,
        px: &MathExpressionInputBar,
        py: &MathExpressionInputBar,
    ) {
        if let Some(ref err) = self.bank_error {
            let announced = format!("Syntax Error: {err}");
            if self.last_announced_bank_error.as_deref() != Some(&announced) {
                crate::accessibility::announce_status_with_priority(&announced, "assertive");
                self.had_bank_error = true;
                self.last_announced_bank_error = Some(announced);
            }
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.colored_label(palette.error, format!("⚠️ Syntax Error: {err}"));
        } else if self.had_bank_error && self.is_valid(px, py) {
            crate::accessibility::announce_status("Planar mathematical expressions are valid");
            self.had_bank_error = false;
            self.last_announced_bank_error = None;
        }
    }

    pub(crate) fn sliders_ui(&mut self, ui: &mut egui::Ui, id_salt: &str) -> bool {
        let changed = ui.push_id(id_salt, |ui| self.slider_bank.ui(ui)).inner;
        if changed {
            for (k, v) in self.slider_bank.export_active_map() {
                self.remembered_values.insert(k, v);
            }
        }
        changed
    }
}
