#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{InteractiveTool, MathExpressionInputBar};
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::ai::deep_learning_theory::calculus::{
    gelu, gelu_prime, relu, relu_prime, sigmoid, sigmoid_prime, tanh, tanh_prime,
};
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;
use std::collections::BTreeMap;

#[path = "activation_functions/ui.rs"]
mod ui;

pub(crate) const ACTIVATION_X_MIN_BOUND: f64 = -100.0;
pub(crate) const ACTIVATION_X_MAX_BOUND: f64 = 100.0;
pub(super) const ACTIVATION_MIN_SPAN: f64 = 0.1;
const GELU_ASYMPTOTIC_CUTOFF: f64 = 20.0;
const CUSTOM_FD_STEP: f64 = 1e-5;
const DEFAULT_ACTIVATION_EXPR: &str = "clamp(a * x, -1, 1)";

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
    Custom,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ActivationFunctionsState {
    selected_function: ActivationFunction,
    x_min: f64,
    x_max: f64,
    points: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    custom_expr: Option<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    custom_params: BTreeMap<String, f64>,
}

pub struct ActivationFunctionsTool {
    pub(crate) selected_function: ActivationFunction,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    pub(crate) points: usize,
    pub(crate) math_input: MathExpressionInputBar,
}

impl Default for ActivationFunctionsTool {
    fn default() -> Self {
        Self {
            selected_function: ActivationFunction::ReLU,
            x_min: -5.0,
            x_max: 5.0,
            points: 200,
            math_input: MathExpressionInputBar::new(
                "f(x)",
                DEFAULT_ACTIVATION_EXPR,
                vec![
                    ("Parameterized HardTanh", "clamp(a * x, -1, 1)"),
                    ("Leaky ReLU", "max(a * x, x)"),
                    ("Swish / SiLU", "x / (1 + exp(-b * x))"),
                    ("Step Threshold", "step(x - a)"),
                ],
            )
            .with_axes(&["x"]),
        }
    }
}

fn evaluate_guarded_gelu(x_vals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut y_out = Vec::with_capacity(x_vals.len());
    let mut dy_out = Vec::with_capacity(x_vals.len());
    for &x in x_vals {
        if !x.is_finite() || x < -GELU_ASYMPTOTIC_CUTOFF {
            y_out.push(0.0);
            dy_out.push(0.0);
        } else if x > GELU_ASYMPTOTIC_CUTOFF {
            y_out.push(x);
            dy_out.push(1.0);
        } else {
            let single = Vector::from_vec(vec![x]);
            let y = gelu(&single)[0];
            let dy = gelu_prime(&single)[0];
            y_out.push(if y.is_finite() { y } else { 0.0 });
            dy_out.push(if dy.is_finite() { dy } else { 0.0 });
        }
    }
    (y_out, dy_out)
}

impl ActivationFunctionsTool {
    fn evaluate_custom_curves(&self, x_vals: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let Some(compiled) = self.math_input.active_compiled() else {
            return (vec![0.0; x_vals.len()], vec![0.0; x_vals.len()]);
        };
        let params = self.math_input.param_slice();
        let mut y_out = Vec::with_capacity(x_vals.len());
        let mut dy_out = Vec::with_capacity(x_vals.len());
        for &x in x_vals {
            let y = compiled.eval_1d(x, params);
            let y_plus = compiled.eval_1d(x + CUSTOM_FD_STEP, params);
            let y_minus = compiled.eval_1d(x - CUSTOM_FD_STEP, params);
            let dy = (y_plus - y_minus) / (2.0 * CUSTOM_FD_STEP);
            y_out.push(if y.is_finite() {
                y.clamp(-1e6, 1e6)
            } else {
                f64::NAN
            });
            dy_out.push(if dy.is_finite() {
                dy.clamp(-1e6, 1e6)
            } else {
                f64::NAN
            });
        }
        (y_out, dy_out)
    }

    pub(crate) fn evaluate_curves(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if !self.has_finite_parameters() {
            return (Vec::new(), Vec::new());
        }
        let step = (self.x_max - self.x_min) / (self.points as f64 - 1.0);
        let x_vals: Vec<f64> = (0..self.points)
            .map(|i| self.x_min + (i as f64) * step)
            .collect();

        let (y_vals, dy_vals) = match self.selected_function {
            ActivationFunction::Gelu => evaluate_guarded_gelu(&x_vals),
            ActivationFunction::Custom => self.evaluate_custom_curves(&x_vals),
            other => {
                let x_vec = Vector::from_vec(x_vals.clone());
                let (y_vec, dy_vec) = match other {
                    ActivationFunction::ReLU => (relu(&x_vec), relu_prime(&x_vec)),
                    ActivationFunction::Sigmoid => (sigmoid(&x_vec), sigmoid_prime(&x_vec)),
                    ActivationFunction::Tanh
                    | ActivationFunction::Gelu
                    | ActivationFunction::Custom => (tanh(&x_vec), tanh_prime(&x_vec)),
                };
                (
                    y_vec.iter().copied().collect(),
                    dy_vec.iter().copied().collect(),
                )
            }
        };

        let f_pts = x_vals
            .iter()
            .zip(y_vals.iter())
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|(&x, &y)| [x, y])
            .collect();
        let df_pts = x_vals
            .iter()
            .zip(dy_vals.iter())
            .filter(|(x, dy)| x.is_finite() && dy.is_finite())
            .map(|(&x, &dy)| [x, dy])
            .collect();
        (f_pts, df_pts)
    }
}

impl InteractiveTool for ActivationFunctionsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Activation Functions"
    }

    fn domain(&self) -> &'static str {
        "ai"
    }

    fn has_finite_parameters(&self) -> bool {
        self.x_min.is_finite()
            && self.x_max.is_finite()
            && (ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND).contains(&self.x_min)
            && (ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND).contains(&self.x_max)
            && self.x_min < self.x_max
            && (2..=10_000).contains(&self.points)
    }

    fn save_state(&self) -> Option<String> {
        let is_custom = self.selected_function == ActivationFunction::Custom;
        let custom_expr = (is_custom || self.math_input.raw_text != DEFAULT_ACTIVATION_EXPR)
            .then(|| self.math_input.raw_text.clone());
        let custom_params = if is_custom {
            self.math_input.params_map()
        } else {
            BTreeMap::new()
        };
        serde_json::to_string(&ActivationFunctionsState {
            selected_function: self.selected_function,
            x_min: self.x_min,
            x_max: self.x_max,
            points: self.points,
            custom_expr,
            custom_params,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: ActivationFunctionsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid activation state: {e}"))?;
        if !parsed.x_min.is_finite()
            || !parsed.x_max.is_finite()
            || !(ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND).contains(&parsed.x_min)
            || !(ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND).contains(&parsed.x_max)
            || parsed.x_min >= parsed.x_max
            || !(2..=10_000).contains(&parsed.points)
        {
            return Err("Activation plot range and point count must be valid and finite".into());
        }
        self.selected_function = parsed.selected_function;
        self.x_min = parsed.x_min;
        self.x_max = parsed.x_max;
        self.points = parsed.points;
        self.math_input.set_expression(
            parsed
                .custom_expr
                .as_deref()
                .unwrap_or(DEFAULT_ACTIVATION_EXPR),
        );
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
        let (f_pts, df_pts) = self.evaluate_curves();
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let line = Line::new("f(x)", PlotPoints::new(f_pts)).width(2.0_f32);
            let deriv_line = Line::new("f'(x)", PlotPoints::new(df_pts)).width(1.5_f32);
            Plot::new("activation_function_plot")
                .legend(Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(line);
                    plot_ui.line(deriv_line);
                });
        });
    }
}

// [cite:algorithmic_information_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ActivationFunctionsTool",
        display_name: "Activation Functions",
        domain: "ai",
        tags: &["relu", "sigmoid", "tanh", "gelu", "softmax", "swish", "neural network"],
        build: || Box::new(ActivationFunctionsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ActivationFunctionsTool {
    fn theory_description(&self) -> String {
        match self.selected_function {
            ActivationFunction::ReLU => "Rectified Linear Unit: $f(x) = \\max(0, x)$".into(),
            ActivationFunction::Sigmoid => {
                "Sigmoid activation function: $f(x) = \\frac{1}{1 + e^{-x}}$".into()
            }
            ActivationFunction::Tanh => "Hyperbolic tangent: $f(x) = \\tanh(x)$".into(),
            ActivationFunction::Gelu => {
                "Gaussian Error Linear Unit: $f(x) \\approx 0.5x(1 + \\tanh(\\sqrt{2/\\pi}(x + 0.044715x^3)))$".into()
            }
            ActivationFunction::Custom => {
                "Custom compiled neural activation $f(x)$ with central finite-difference gradient $f'(x)$".into()
            }
        }
    }

    fn phonetic_description(&self) -> String {
        "Activation functions theory".into()
    }

    fn theory_citation(&self) -> String {
        "Deep Learning (Goodfellow et al., 2016)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Activation Functions",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "selected_function": "Gelu",
    "x_min": -4.0,
    "x_max": 4.0,
    "points": 200
  }
}"#;

const CUSTOM_HARDTANH_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Parameterized HardTanh",
    "domain": "ai",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "selected_function": "Custom",
    "x_min": -4.0,
    "x_max": 4.0,
    "points": 200,
    "custom_expr": "clamp(a * x, -1, 1)",
    "custom_params": {
      "a": 1.5
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "activation_functions_starter",
        title: "Activation Functions",
        description: "Comparison of non-linear activations and derivatives: ReLU, Sigmoid, Tanh, and GELU.",
        category: "Neural Architectures",
        icon: "⚡",
        domain: "ai",
        tool_name: Some("ActivationFunctionsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "activation_custom_hardtanh",
        title: "Parameterized HardTanh",
        description: "Piecewise saturated activation clamp(a*x, -1, 1) and finite-difference gradient f'(x).",
        category: "Neural Architectures",
        icon: "🧮",
        domain: "ai",
        tool_name: Some("ActivationFunctionsTool"),
        preset_generator: || CUSTOM_HARDTANH_PRESET_JSON,
    }
}

#[cfg(test)]
#[path = "activation_functions/tests.rs"]
mod tests;
