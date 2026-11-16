use crate::framework::InteractiveTool;
use crate::reflective_ui::render_formula_with_export;
use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use math_explorer::ai::deep_learning_theory::calculus::{
    gelu, gelu_prime, relu, relu_prime, sigmoid, sigmoid_prime, tanh, tanh_prime,
};
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    Gelu,
}

pub struct ActivationFunctionsTool {
    selected_function: ActivationFunction,
    x_min: f64,
    x_max: f64,
    points: usize,
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

impl InteractiveTool for ActivationFunctionsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Activation Functions"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("activation_functions_controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            ui.label("Select Activation Function:");
            ui.radio_value(
                &mut self.selected_function,
                ActivationFunction::ReLU,
                "ReLU",
            );
            ui.radio_value(
                &mut self.selected_function,
                ActivationFunction::Sigmoid,
                "Sigmoid",
            );
            ui.radio_value(
                &mut self.selected_function,
                ActivationFunction::Tanh,
                "Tanh",
            );
            ui.radio_value(
                &mut self.selected_function,
                ActivationFunction::Gelu,
                "GELU",
            );

            ui.separator();
            ui.label("Plot Range");
            ui.horizontal(|ui| {
                ui.label("X Min:");
                ui.add(egui::DragValue::new(&mut self.x_min).speed(0.1));
            });
            ui.horizontal(|ui| {
                ui.label("X Max:");
                ui.add(egui::DragValue::new(&mut self.x_max).speed(0.1));
            });

            // Ensure min < max
            if self.x_min >= self.x_max {
                self.x_max = self.x_min + 1.0;
            }

            ui.separator();
            ui.label("Math Info");
            match self.selected_function {
                ActivationFunction::ReLU => {
                    render_formula_with_export(
                        ui,
                        r"f(x) = \max(0, x)",
                        Some("f(x) = max(0, x)"),
                    );
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
                    render_formula_with_export(
                        ui,
                        r"f(x) = \tanh(x)",
                        Some("f(x) = tanh(x)"),
                    );
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
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Generate x values
            let step = (self.x_max - self.x_min) / (self.points as f64 - 1.0);
            let mut x_vals = Vec::with_capacity(self.points);
            for i in 0..self.points {
                x_vals.push(self.x_min + (i as f64) * step);
            }

            let x_vec = Vector::from_vec(x_vals.clone());

            // Compute y values and derivatives
            let (y_vec, dy_vec) = match self.selected_function {
                ActivationFunction::ReLU => (relu(&x_vec), relu_prime(&x_vec)),
                ActivationFunction::Sigmoid => (sigmoid(&x_vec), sigmoid_prime(&x_vec)),
                ActivationFunction::Tanh => (tanh(&x_vec), tanh_prime(&x_vec)),
                ActivationFunction::Gelu => (gelu(&x_vec), gelu_prime(&x_vec)),
            };

            // Map to egui_plot points
            let plot_points: PlotPoints = x_vals
                .iter()
                .zip(y_vec.iter())
                .map(|(x, y)| [*x, *y])
                .collect();

            let deriv_points: PlotPoints = x_vals
                .iter()
                .zip(dy_vec.iter())
                .map(|(x, y)| [*x, *y])
                .collect();

            let line = Line::new("f(x)", plot_points).width(2.0_f32);
            let deriv_line = Line::new("f'(x)", deriv_points).width(1.5_f32);

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
        name: "ActivationFunctionsTool",
        domain: "ai",
        tags: &[],
        build: || Box::new(ActivationFunctionsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ActivationFunctionsTool {
    fn theory_description(&self) -> String {
        match self.selected_function {
            ActivationFunction::ReLU => "Rectified Linear Unit: $f(x) = \\max(0, x)$".into(),
            ActivationFunction::Sigmoid => "Sigmoid activation function: $f(x) = \\frac{1}{1 + e^{-x}}$".into(),
            ActivationFunction::Tanh => "Hyperbolic tangent: $f(x) = \\tanh(x)$".into(),
            ActivationFunction::Gelu => "Gaussian Error Linear Unit: $f(x) \\approx 0.5x(1 + \\tanh(\\sqrt{2/\\pi}(x + 0.044715x^3)))$".into(),
        }
    }
    fn phonetic_description(&self) -> String { "Activation functions theory".into() }
    fn theory_citation(&self) -> String { "Deep Learning (Goodfellow et al., 2016)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Activation Functions",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "active_function": "GELU",
    "input_range": [
      -3.0,
      3.0
    ]
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "activation_functions_starter",
        title: "Activation Functions",
        description: "Comparison of non-linear activations: ReLU, GELU, Sigmoid, and Swish functions.",
        category: "Neural Architectures",
        icon: "⚡",
        domain: "ai",
        tool_name: Some("ActivationFunctionsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
