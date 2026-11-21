#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{bounded_drag_value, enforce_ordered_drag_range, InteractiveTool};
use crate::reflective_ui::render_formula_with_export;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::ai::deep_learning_theory::calculus::{
    gelu, gelu_prime, relu, relu_prime, sigmoid, sigmoid_prime, tanh, tanh_prime,
};
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;

pub(crate) const ACTIVATION_X_MIN_BOUND: f64 = -100.0;
pub(crate) const ACTIVATION_X_MAX_BOUND: f64 = 100.0;
const ACTIVATION_MIN_SPAN: f64 = 0.1;
const GELU_ASYMPTOTIC_CUTOFF: f64 = 20.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ActivationFunctionsState {
    selected_function: ActivationFunction,
    x_min: f64,
    x_max: f64,
    points: usize,
}

pub struct ActivationFunctionsTool {
    pub(crate) selected_function: ActivationFunction,
    pub(crate) x_min: f64,
    pub(crate) x_max: f64,
    pub(crate) points: usize,
}

impl Default for ActivationFunctionsTool {
    fn default() -> Self {
        Self {
            selected_function: ActivationFunction::ReLU,
            x_min: -5.0,
            x_max: 5.0,
            points: 200,
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
    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id =
            crate::framework::scoped_panel_id(parent_ui, "activation_functions_controls");
        crate::framework::PaneSidePanel::right(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();
            ui.label("Select Activation Function:");
            for (fn_kind, label) in [
                (ActivationFunction::ReLU, "ReLU"),
                (ActivationFunction::Sigmoid, "Sigmoid"),
                (ActivationFunction::Tanh, "Tanh"),
                (ActivationFunction::Gelu, "GELU"),
            ] {
                ui.radio_value(&mut self.selected_function, fn_kind, label);
            }
            self.draw_range_controls(ui);
            self.draw_math_info(ui);
        });
    }

    fn draw_range_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Plot Range");
        let mut min_changed = false;
        let mut max_changed = false;
        ui.horizontal(|ui| {
            ui.label("X Min:");
            min_changed = ui
                .add(bounded_drag_value(
                    &mut self.x_min,
                    ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND,
                    0.1,
                ))
                .changed();
        });
        ui.horizontal(|ui| {
            ui.label("X Max:");
            max_changed = ui
                .add(bounded_drag_value(
                    &mut self.x_max,
                    ACTIVATION_X_MIN_BOUND..=ACTIVATION_X_MAX_BOUND,
                    0.1,
                ))
                .changed();
        });
        enforce_ordered_drag_range(
            &mut self.x_min,
            &mut self.x_max,
            ACTIVATION_X_MIN_BOUND,
            ACTIVATION_X_MAX_BOUND,
            ACTIVATION_MIN_SPAN,
            min_changed,
            max_changed,
        );
    }

    fn draw_math_info(&self, ui: &mut egui::Ui) {
        ui.separator();
        ui.label("Math Info");
        match self.selected_function {
            ActivationFunction::ReLU => {
                render_formula_with_export(ui, r"f(x) = \max(0, x)", Some("f(x) = max(0, x)"));
                render_formula_with_export(
                    ui,
                    r"f'(x) = 1 \text{ if } x > 0 \text{ else } 0",
                    Some("f'(x) = 1 if x > 0 else 0"),
                );
            }
            ActivationFunction::Sigmoid => {
                render_formula_with_export(
                    ui,
                    r"f(x) = \frac{1}{1 + e^{-x}}",
                    Some("f(x) = 1 / (1 + e^(-x))"),
                );
                render_formula_with_export(
                    ui,
                    r"f'(x) = f(x)(1 - f(x))",
                    Some("f'(x) = f(x) * (1 - f(x))"),
                );
            }
            ActivationFunction::Tanh => {
                render_formula_with_export(ui, r"f(x) = \tanh(x)", Some("f(x) = tanh(x)"));
                render_formula_with_export(
                    ui,
                    r"f'(x) = 1 - \tanh^2(x)",
                    Some("f'(x) = 1 - tanh^2(x)"),
                );
            }
            ActivationFunction::Gelu => {
                render_formula_with_export(
                    ui,
                    r"f(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))",
                    Some("f(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))"),
                );
            }
        }
    }

    fn evaluate_curves(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if !self.has_finite_parameters() {
            return (Vec::new(), Vec::new());
        }
        let step = (self.x_max - self.x_min) / (self.points as f64 - 1.0);
        let x_vals: Vec<f64> = (0..self.points)
            .map(|i| self.x_min + (i as f64) * step)
            .collect();

        let (y_vals, dy_vals) = match self.selected_function {
            ActivationFunction::Gelu => evaluate_guarded_gelu(&x_vals),
            other => {
                let x_vec = Vector::from_vec(x_vals.clone());
                let (y_vec, dy_vec) = match other {
                    ActivationFunction::ReLU => (relu(&x_vec), relu_prime(&x_vec)),
                    ActivationFunction::Sigmoid => (sigmoid(&x_vec), sigmoid_prime(&x_vec)),
                    ActivationFunction::Tanh | ActivationFunction::Gelu => {
                        (tanh(&x_vec), tanh_prime(&x_vec))
                    }
                };
                (y_vec.iter().copied().collect(), dy_vec.iter().copied().collect())
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
        serde_json::to_string(&ActivationFunctionsState {
            selected_function: self.selected_function,
            x_min: self.x_min,
            x_max: self.x_max,
            points: self.points,
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
        tags: &[],
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
