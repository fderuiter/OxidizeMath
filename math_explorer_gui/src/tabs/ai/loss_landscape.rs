#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{
    bounded_drag_value, enforce_ordered_drag_range, InteractiveTool, MathExpressionInputBar,
};
use eframe::egui;
use egui::Color32;
use egui_plot::{Plot, Points};
use nalgebra::DVector;
use std::collections::BTreeMap;

pub(crate) const LOSS_PARAM_MIN_BOUND: f64 = -100.0;
pub(crate) const LOSS_PARAM_MAX_BOUND: f64 = 100.0;
const LOSS_MIN_SPAN: f64 = 0.1;
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
    x_data: DVector<f64>,
    y_data: DVector<f64>,
    pub(crate) predictor_mode: PredictorMode,
    pub(crate) w_min: f64,
    pub(crate) w_max: f64,
    pub(crate) b_min: f64,
    pub(crate) b_max: f64,
    pub(crate) resolution: usize,
    pub(crate) math_input: MathExpressionInputBar,
}

impl Default for LossLandscapeTool {
    fn default() -> Self {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        Self {
            x_data: DVector::from_vec(x),
            y_data: DVector::from_vec(y),
            predictor_mode: PredictorMode::Linear,
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
}

impl LossLandscapeTool {
    fn draw_ordered_bound_pair(ui: &mut egui::Ui, label: &str, min_v: &mut f64, max_v: &mut f64) {
        ui.label(label);
        let mut min_changed = false;
        let mut max_changed = false;
        ui.horizontal(|ui| {
            min_changed = ui
                .add(
                    bounded_drag_value(min_v, LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND, 0.1)
                        .prefix("Min: "),
                )
                .changed();
            max_changed = ui
                .add(
                    bounded_drag_value(max_v, LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND, 0.1)
                        .prefix("Max: "),
                )
                .changed();
        });
        enforce_ordered_drag_range(
            min_v,
            max_v,
            LOSS_PARAM_MIN_BOUND,
            LOSS_PARAM_MAX_BOUND,
            LOSS_MIN_SPAN,
            min_changed,
            max_changed,
        );
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "loss_landscape_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Parameters");
            ui.separator();
            ui.label("Predictor Formula:");
            ui.horizontal(|ui| {
                ui.radio_value(
                    &mut self.predictor_mode,
                    PredictorMode::Linear,
                    "Linear (w*x+b)",
                );
                ui.radio_value(
                    &mut self.predictor_mode,
                    PredictorMode::Custom,
                    "Custom f(x,w,b)",
                );
            });
            if self.predictor_mode == PredictorMode::Custom {
                ui.add_space(4.0);
                self.math_input.ui(ui);
            }
            ui.separator();
            Self::draw_ordered_bound_pair(
                ui,
                "Weight (Slope) Range",
                &mut self.w_min,
                &mut self.w_max,
            );
            Self::draw_ordered_bound_pair(
                ui,
                "Bias (Intercept) Range",
                &mut self.b_min,
                &mut self.b_max,
            );
            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.resolution, 10..=50)
                    .text("Resolution (Grid Size) - points"),
            );
            ui.separator();
            ui.label("Target Data: y = 2x + 1");
            ui.label("Loss: Mean Squared Error (MSE)");
        });
    }

    pub(crate) fn evaluate_mse_at(&self, w: f64, b: f64) -> Option<f64> {
        let n = self.x_data.len();
        if n == 0 {
            return None;
        }
        let mut sum_sq = 0.0;
        match self.predictor_mode {
            PredictorMode::Linear => {
                for k in 0..n {
                    let err = (w * self.x_data[k] + b) - self.y_data[k];
                    sum_sq += err * err;
                }
            }
            PredictorMode::Custom => {
                let compiled = self.math_input.active_compiled()?;
                let params = self.math_input.param_slice();
                for k in 0..n {
                    let pred = compiled.eval_3d(self.x_data[k], w, b, params);
                    if !pred.is_finite() {
                        return None;
                    }
                    let err = (pred - self.y_data[k]).clamp(-1e6, 1e6);
                    sum_sq += err * err;
                }
            }
        }
        let loss = sum_sq / (n as f64);
        loss.is_finite().then_some(loss)
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
                    if let Some(loss) = self.evaluate_mse_at(w, b) {
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
        serde_json::to_string(&LossLandscapeState {
            w_min: self.w_min,
            w_max: self.w_max,
            b_min: self.b_min,
            b_max: self.b_max,
            resolution: self.resolution,
            predictor_mode: self.predictor_mode,
            custom_expr,
            custom_params,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: LossLandscapeState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid loss landscape state: {e}"))?;
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
        {
            return Err("Loss landscape parameter bounds or grid resolution invalid".into());
        }
        self.w_min = parsed.w_min;
        self.w_max = parsed.w_max;
        self.b_min = parsed.b_min;
        self.b_max = parsed.b_max;
        self.resolution = parsed.resolution;
        self.predictor_mode = parsed.predictor_mode;
        self.math_input
            .set_expression(parsed.custom_expr.as_deref().unwrap_or(DEFAULT_LOSS_EXPR));
        self.math_input.set_params_map(&parsed.custom_params);
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
                    if self.predictor_mode == PredictorMode::Linear {
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
        match self.predictor_mode {
            PredictorMode::Linear => "Mean squared error loss landscape over parameter space $(w, b)$: $L(w, b) = \\frac{1}{n}\\sum_{i=1}^n (w x_i + b - y_i)^2$. The geometric curvature of the landscape is governed by the Hessian matrix $H = \\nabla^2 L$.".into(),
            PredictorMode::Custom => "Non-convex mean squared error loss landscape $L(w, b) = \\frac{1}{n}\\sum_{i=1}^n (f(x_i, w, b) - y_i)^2$ evaluated over compiled custom predictor bytecode.".into(),
        }
    }

    fn phonetic_description(&self) -> String {
        "Loss landscape and mean squared error optimization theory".into()
    }

    fn theory_citation(&self) -> String {
        "Visualizing the Loss Landscape of Neural Nets (Li et al., 2018); Deep Learning (Goodfellow et al., 2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Loss Landscape",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "w_min": -1.0,
    "w_max": 5.0,
    "b_min": -2.0,
    "b_max": 4.0,
    "resolution": 30
  }
}"#;

const NONCONVEX_SINUSOIDAL_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Non-Convex Sinusoidal Loss",
    "domain": "ai",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "w_min": -4.0,
    "w_max": 4.0,
    "b_min": -3.0,
    "b_max": 3.0,
    "resolution": 32,
    "predictor_mode": "Custom",
    "custom_expr": "w * sin(b * x) + a",
    "custom_params": {
      "a": 1.0
    }
  }
}"#;

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
mod tests {
    #![cfg_attr(any(), verified(opt_out = "tests"))]

    use super::*;

    #[test]
    fn test_custom_loss_landscape_mse_and_preset_roundtrip() {
        let mut tool = LossLandscapeTool {
            predictor_mode: PredictorMode::Custom,
            ..Default::default()
        };
        tool.math_input.set_expression("w * x + b");
        // At true solution w = 2, b = 1 for y = 2x + 1, MSE must be 0
        assert!((tool.evaluate_mse_at(2.0, 1.0).unwrap() - 0.0).abs() < 1e-12);

        tool.import_preset_json(NONCONVEX_SINUSOIDAL_PRESET_JSON)
            .expect("non-convex preset should load");
        assert!(tool.evaluate_mse_at(1.5, 0.8).unwrap().is_finite());
        assert!(tool.import_preset_json(STARTER_PRESET_JSON).is_ok());
    }
}
