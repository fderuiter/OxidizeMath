#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub mod data;
mod ui;

pub use data::{LossMetric, TargetDataMode};
pub(crate) use data::{NONCONVEX_SINUSOIDAL_PRESET_JSON, STARTER_PRESET_JSON};

use crate::framework::{InteractiveTool, MathExpressionInputBar};
use data::{
    default_preset_targets, default_target_formula_bar, format_target_table, parse_target_table,
    sample_formula_targets, DEFAULT_HUBER_DELTA, DEFAULT_TABLE_TEXT, DEFAULT_TARGET_EXPR,
    DEFAULT_TARGET_SAMPLES, DEFAULT_TARGET_X_BOUNDS, MAX_TABLE_ROWS,
};
use eframe::egui;
use egui::Color32;
use egui_plot::{Plot, Points};
use std::collections::BTreeMap;

pub(crate) const LOSS_PARAM_MIN_BOUND: f64 = -100.0;
pub(crate) const LOSS_PARAM_MAX_BOUND: f64 = 100.0;
pub(super) const LOSS_MIN_SPAN: f64 = 0.1;
const DEFAULT_LOSS_EXPR: &str = "w * sin(b * x)";

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum PredictorMode {
    #[default]
    Linear,
    Custom,
}

impl PredictorMode {
    const fn is_linear(&self) -> bool {
        matches!(self, Self::Linear)
    }
}

pub struct LossLandscapeTool {
    cached_targets: Vec<[f64; 2]>,
    pub(crate) predictor_mode: PredictorMode,
    pub(crate) target_mode: TargetDataMode,
    pub(crate) loss_metric: LossMetric,
    pub(crate) huber_delta: f64,
    pub(crate) target_formula_bar: MathExpressionInputBar,
    pub(crate) target_x_bounds: [f64; 2],
    pub(crate) target_sample_count: usize,
    pub(crate) target_table_text: String,
    pub(crate) table_parse_error: Option<String>,
    pub(crate) w_min: f64,
    pub(crate) w_max: f64,
    pub(crate) b_min: f64,
    pub(crate) b_max: f64,
    pub(crate) resolution: usize,
    pub(crate) math_input: MathExpressionInputBar,
}

impl Default for LossLandscapeTool {
    fn default() -> Self {
        Self {
            cached_targets: default_preset_targets(),
            predictor_mode: PredictorMode::Linear,
            target_mode: TargetDataMode::PresetLinear,
            loss_metric: LossMetric::Mse,
            huber_delta: DEFAULT_HUBER_DELTA,
            target_formula_bar: default_target_formula_bar(),
            target_x_bounds: DEFAULT_TARGET_X_BOUNDS,
            target_sample_count: DEFAULT_TARGET_SAMPLES,
            target_table_text: DEFAULT_TABLE_TEXT.to_string(),
            table_parse_error: None,
            w_min: 0.0,
            w_max: 4.0,
            b_min: -1.0,
            b_max: 3.0,
            resolution: 20,
            math_input: MathExpressionInputBar::new(
                "f(x,w,b)",
                DEFAULT_LOSS_EXPR,
                vec![
                    ("Sinusoidal Non-Convex", "w * sin(b * x) + a"),
                    ("Shifted ReLU Neuron", "w * max(0, x + b)"),
                    ("Saturated Gate", "w * clamp(x + b, -1, 1)"),
                    ("Linear Baseline", "w * x + b"),
                ],
            )
            .with_axes(&["x", "w", "b"]),
        }
    }
}

fn default_huber_delta() -> f64 {
    DEFAULT_HUBER_DELTA
}

fn is_default_huber_delta(v: &f64) -> bool {
    (*v - DEFAULT_HUBER_DELTA).abs() < 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LossLandscapeState {
    w_min: f64,
    w_max: f64,
    b_min: f64,
    b_max: f64,
    resolution: usize,
    #[serde(default, skip_serializing_if = "PredictorMode::is_linear")]
    predictor_mode: PredictorMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "LossMetric::is_mse")]
    loss_metric: LossMetric,
    #[serde(default = "default_huber_delta", skip_serializing_if = "is_default_huber_delta")]
    huber_delta: f64,
    #[serde(default, skip_serializing_if = "TargetDataMode::is_preset_linear")]
    target_mode: TargetDataMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    target_params: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_x_bounds: Option<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_sample_count: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    target_table: Option<Vec<[f64; 2]>>,
}

impl LossLandscapeTool {
    #[cfg(test)]
    pub fn set_target_table_text(&mut self, text: &str) {
        self.target_mode = TargetDataMode::Table;
        self.target_table_text = text.to_string();
        self.refresh_target_samples();
    }

    pub fn refresh_target_samples(&mut self) {
        match self.target_mode {
            TargetDataMode::PresetLinear => {
                self.table_parse_error = None;
                self.cached_targets = default_preset_targets();
            }
            TargetDataMode::Formula => {
                self.table_parse_error = None;
                if let Some(pts) = sample_formula_targets(
                    &self.target_formula_bar,
                    self.target_x_bounds,
                    self.target_sample_count,
                ) {
                    self.cached_targets = pts;
                }
            }
            TargetDataMode::Table => match parse_target_table(&self.target_table_text) {
                Ok(rows) => {
                    self.table_parse_error = None;
                    self.cached_targets = rows;
                }
                Err(err) => {
                    self.table_parse_error = Some(err);
                }
            },
        }
    }

    pub(crate) fn evaluate_loss_at(&self, w: f64, b: f64) -> Option<f64> {
        let n = self.cached_targets.len();
        if n == 0 {
            return None;
        }
        let mut total_loss = 0.0;
        let metric = self.loss_metric;
        let delta = self.huber_delta;
        match self.predictor_mode {
            PredictorMode::Linear => {
                for &[x, y] in &self.cached_targets {
                    let err = (w * x + b - y).clamp(-1e6, 1e6);
                    total_loss += metric.point_loss(err, delta);
                }
            }
            PredictorMode::Custom => {
                let compiled = self.math_input.active_compiled()?;
                let params = self.math_input.param_slice();
                for &[x, y] in &self.cached_targets {
                    let pred = compiled.eval_3d(x, w, b, params);
                    if !pred.is_finite() {
                        return None;
                    }
                    let err = (pred - y).clamp(-1e6, 1e6);
                    total_loss += metric.point_loss(err, delta);
                }
            }
        }
        let loss = total_loss / (n as f64);
        loss.is_finite().then_some(loss)
    }

    #[cfg(test)]
    pub(crate) fn evaluate_mse_at(&self, w: f64, b: f64) -> Option<f64> {
        self.evaluate_loss_at(w, b)
    }

    fn compute_loss_points(&self, low: Color32, high: Color32) -> Vec<Points<'static>> {
        if !self.has_finite_parameters() {
            return Vec::new();
        }
        let w_step = (self.w_max - self.w_min) / (self.resolution as f64 - 1.0);
        let b_step = (self.b_max - self.b_min) / (self.resolution as f64 - 1.0);
        let mut min_loss = f64::MAX;
        let mut max_loss = f64::MIN;
        let mut loss_grid = Vec::with_capacity(self.resolution * self.resolution);

        for i in 0..self.resolution {
            for j in 0..self.resolution {
                let w = self.w_min + i as f64 * w_step;
                let b = self.b_min + j as f64 * b_step;
                if w.is_finite() && b.is_finite() {
                    if let Some(loss) = self.evaluate_loss_at(w, b) {
                        min_loss = min_loss.min(loss);
                        max_loss = max_loss.max(loss);
                        loss_grid.push((w, b, loss));
                    }
                }
            }
        }

        loss_grid
            .into_iter()
            .map(|(w, b, loss)| {
                let normalized = if max_loss > min_loss {
                    ((loss - min_loss) / (max_loss - min_loss)).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let lerp = |a: u8, b: u8| {
                    (f64::from(a) + (f64::from(b) - f64::from(a)) * normalized).round() as u8
                };
                Points::new(format!("Loss: {loss:.4}"), vec![[w, b]])
                    .color(Color32::from_rgb(
                        lerp(low.r(), high.r()),
                        lerp(low.g(), high.g()),
                        lerp(low.b(), high.b()),
                    ))
                    .radius(3.0_f32)
            })
            .collect()
    }
}

impl InteractiveTool for LossLandscapeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Loss Landscape"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn has_finite_parameters(&self) -> bool {
        self.w_min.is_finite()
            && self.w_max.is_finite()
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.w_min)
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.w_max)
            && self.w_min < self.w_max
            && self.b_min.is_finite()
            && self.b_max.is_finite()
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.b_min)
            && (LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&self.b_max)
            && self.b_min < self.b_max
            && (10..=50).contains(&self.resolution)
            && self.huber_delta.is_finite()
            && (0.01..=50.0).contains(&self.huber_delta)
            && !self.cached_targets.is_empty()
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.predictor_mode == PredictorMode::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_LOSS_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        let is_formula_target = self.target_mode == TargetDataMode::Formula;
        let is_table_target = self.target_mode == TargetDataMode::Table;
        serde_json::to_string(&LossLandscapeState {
            w_min: self.w_min,
            w_max: self.w_max,
            b_min: self.b_min,
            b_max: self.b_max,
            resolution: self.resolution,
            predictor_mode: self.predictor_mode,
            custom_expr,
            custom_params,
            loss_metric: self.loss_metric,
            huber_delta: self.huber_delta,
            target_mode: self.target_mode,
            target_expr: is_formula_target.then(|| self.target_formula_bar.raw_text.clone()),
            target_params: if is_formula_target {
                self.target_formula_bar.params_map()
            } else {
                BTreeMap::new()
            },
            target_x_bounds: is_formula_target.then_some(self.target_x_bounds),
            target_sample_count: is_formula_target.then_some(self.target_sample_count),
            target_table: is_table_target.then(|| self.cached_targets.clone()),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: LossLandscapeState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid loss landscape state: {e}"))?;
        validate_loaded_loss_state(&parsed)?;
        self.w_min = parsed.w_min;
        self.w_max = parsed.w_max;
        self.b_min = parsed.b_min;
        self.b_max = parsed.b_max;
        self.resolution = parsed.resolution;
        self.predictor_mode = parsed.predictor_mode;
        self.loss_metric = parsed.loss_metric;
        self.huber_delta = parsed.huber_delta;
        self.target_mode = parsed.target_mode;
        self.math_input
            .set_expression(parsed.custom_expr.as_deref().unwrap_or(DEFAULT_LOSS_EXPR));
        self.math_input.set_params_map(&parsed.custom_params);
        self.target_formula_bar.set_expression(
            parsed
                .target_expr
                .as_deref()
                .unwrap_or(DEFAULT_TARGET_EXPR),
        );
        self.target_formula_bar.set_params_map(&parsed.target_params);
        self.target_x_bounds = parsed.target_x_bounds.unwrap_or(DEFAULT_TARGET_X_BOUNDS);
        self.target_sample_count = parsed.target_sample_count.unwrap_or(DEFAULT_TARGET_SAMPLES);
        if let Some(rows) = parsed.target_table {
            self.target_table_text = format_target_table(&rows);
        }
        self.refresh_target_samples();
        Ok(())
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.draw_controls(parent_ui);
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let points = self.compute_loss_points(palette.axis_z, palette.axis_x);
            Plot::new("loss_landscape_plot")
                .x_axis_label("Weight (w)")
                .y_axis_label("Bias (b)")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    for point in points {
                        plot_ui.points(point);
                    }
                    if self.predictor_mode == PredictorMode::Linear
                        && self.target_mode == TargetDataMode::PresetLinear
                    {
                        plot_ui.points(
                            Points::new("Global Minima (True Solution)", vec![[2.0, 1.0]])
                                .color(palette.success)
                                .radius(6.0_f32)
                                .shape(egui_plot::MarkerShape::Circle),
                        );
                    }
                });
        });
    }
}

fn validate_loaded_loss_state(parsed: &LossLandscapeState) -> Result<(), String> {
    if !parsed.w_min.is_finite()
        || !parsed.w_max.is_finite()
        || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&parsed.w_min)
        || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&parsed.w_max)
        || parsed.w_min >= parsed.w_max
        || !parsed.b_min.is_finite()
        || !parsed.b_max.is_finite()
        || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&parsed.b_min)
        || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&parsed.b_max)
        || parsed.b_min >= parsed.b_max
        || !(10..=50).contains(&parsed.resolution)
        || !parsed.huber_delta.is_finite()
        || !(0.01..=50.0).contains(&parsed.huber_delta)
    {
        return Err("Loss landscape parameter bounds, Huber delta, or resolution invalid".into());
    }
    if let Some([x0, x1]) = parsed.target_x_bounds {
        if !x0.is_finite()
            || !x1.is_finite()
            || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&x0)
            || !(LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND).contains(&x1)
            || x0 >= x1
        {
            return Err("Target x range invalid".into());
        }
    }
    if let Some(count) = parsed.target_sample_count {
        if !(2..=MAX_TABLE_ROWS).contains(&count) {
            return Err("Target sample count out of valid range".into());
        }
    }
    if let Some(ref rows) = parsed.target_table {
        if rows.is_empty()
            || rows.len() > MAX_TABLE_ROWS
            || rows
                .iter()
                .any(|[x, y]| !x.is_finite() || !y.is_finite() || x.abs() > 1e6 || y.abs() > 1e6)
        {
            return Err("Target table rows invalid".into());
        }
    }
    Ok(())
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "LossLandscapeTool",
        display_name: "Loss Landscape",
        domain: "ai",
        tags: &[],
        build: || Box::new(LossLandscapeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for LossLandscapeTool {
    fn theory_description(&self) -> String {
        format!(
            "Empirical risk landscape $L(w, b) = \\frac{{1}}{{n}}\\sum_{{i=1}}^n \\ell(f(x_i, w, b) - y_i)$ using {} objective over {} target dataset.",
            self.loss_metric.label(),
            self.target_mode.label()
        )
    }

    fn phonetic_description(&self) -> String {
        "Loss landscape MSE, MAE, and Huber empirical risk optimization theory".into()
    }

    fn theory_citation(&self) -> String {
        "Visualizing the Loss Landscape of Neural Nets (Li et al., 2018); Huber (1964), Annals of Mathematical Statistics".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "loss_landscape_starter",
        title: "Loss Landscape",
        description: "2D linear regression Mean Squared Error (MSE) loss surface L(w, b) over weight and bias space.",
        category: "Optimization",
        icon: "⛰️",
        domain: "ai",
        tool_name: Some("LossLandscapeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "loss_landscape_nonconvex_sinusoid",
        title: "Non-Convex Sinusoidal Loss",
        description: "f(x, w, b) = w*sin(b*x) + a: Multi-minima non-convex MSE landscape with live offset slider a.",
        category: "Optimization",
        icon: "🌄",
        domain: "ai",
        tool_name: Some("LossLandscapeTool"),
        preset_generator: || NONCONVEX_SINUSOIDAL_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests;
