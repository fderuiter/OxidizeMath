use crate::framework::{InteractiveTool, MathExpressionInputBar};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::complex::mapping::conformal_scale_factor;
use num_complex::Complex64;

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum MappingFunction {
    ZSquared,
    Exponential,
    Sine,
    Inverse,
    Custom,
}

impl MappingFunction {
    fn name(&self) -> &'static str {
        match self {
            MappingFunction::ZSquared => "w = z^2",
            MappingFunction::Exponential => "w = e^z",
            MappingFunction::Sine => "w = sin(z)",
            MappingFunction::Inverse => "w = 1/z",
            MappingFunction::Custom => "w = f(z) (Custom)",
        }
    }

    fn evaluate(&self, z: Complex64) -> Complex64 {
        match self {
            MappingFunction::ZSquared => z * z,
            MappingFunction::Exponential => z.exp(),
            MappingFunction::Sine => z.sin(),
            MappingFunction::Inverse => {
                if z.norm() < 1e-6 {
                    Complex64::new(1e6, 1e6) // Avoid division by zero
                } else {
                    1.0 / z
                }
            }
            MappingFunction::Custom => Complex64::new(1e6, 1e6),
        }
    }

    fn derivative(&self, z: Complex64) -> Complex64 {
        match self {
            MappingFunction::ZSquared => 2.0 * z,
            MappingFunction::Exponential => z.exp(),
            MappingFunction::Sine => z.cos(),
            MappingFunction::Inverse => {
                if z.norm() < 1e-6 {
                    Complex64::new(1e6, 1e6) // Avoid division by zero
                } else {
                    -1.0 / (z * z)
                }
            }
            MappingFunction::Custom => Complex64::new(0.0, 0.0),
        }
    }
}

pub struct ComplexMappingTool {
    function: MappingFunction,
    grid_min: f64,
    grid_max: f64,
    grid_density: usize,
    inspection_point: [f64; 2],
    math_input: MathExpressionInputBar,
}

impl Default for ComplexMappingTool {
    fn default() -> Self {
        Self {
            function: MappingFunction::ZSquared,
            grid_min: -2.0,
            grid_max: 2.0,
            grid_density: 10,
            inspection_point: [1.0, 1.0],
            math_input: MathExpressionInputBar::new(
                "w",
                "z^3 - 1/z",
                vec![
                    ("w = z^3 - 1/z", "z^3 - 1/z"),
                    ("w = z^2", "z^2"),
                    ("w = exp(z)", "exp(z)"),
                    ("w = sin(z)", "sin(z)"),
                    ("w = 1/z", "1/z"),
                    ("w = z + 1/z (Joukowsky)", "z + 1/z"),
                ],
            ),
        }
    }
}

impl ComplexMappingTool {
    fn eval_z(&self, z: Complex64) -> Complex64 {
        match self.function {
            MappingFunction::Custom => {
                if let Some(expr) = self.math_input.expr() {
                    let w = expr.eval_complex_z(z);
                    if w.re.is_finite() && w.im.is_finite() {
                        w
                    } else {
                        Complex64::new(1e6, 1e6)
                    }
                } else {
                    Complex64::new(1e6, 1e6)
                }
            }
            _ => self.function.evaluate(z),
        }
    }

    fn deriv_z(&self, z: Complex64) -> Complex64 {
        match self.function {
            MappingFunction::Custom => {
                let h = Complex64::new(1e-7, 0.0);
                let f_plus = self.eval_z(z + h);
                let f_minus = self.eval_z(z - h);
                (f_plus - f_minus) / (2.0 * h)
            }
            _ => self.function.derivative(z),
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ComplexMappingState {
    function: MappingFunction,
    grid_min: f64,
    grid_max: f64,
    grid_density: usize,
    inspection_point: [f64; 2],
}

impl InteractiveTool for ComplexMappingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Complex Mapping"
    }
    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&ComplexMappingState {
            function: self.function,
            grid_min: self.grid_min,
            grid_max: self.grid_max,
            grid_density: self.grid_density,
            inspection_point: self.inspection_point,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(s) = serde_json::from_str::<ComplexMappingState>(state) {
            self.function = s.function;
            self.grid_min = s.grid_min;
            self.grid_max = s.grid_max;
            self.grid_density = s.grid_density;
            self.inspection_point = s.inspection_point;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("complex_mapping_controls").show(ctx, |ui| {
            ui.heading("Mapping Controls");
            ui.separator();

            ui.label("Function:");
            egui::ComboBox::from_id_salt("mapping_function")
                .selected_text(self.function.name())
                .show_ui(ui, |ui| {
                    ui.selectable_value(
                        &mut self.function,
                        MappingFunction::ZSquared,
                        MappingFunction::ZSquared.name(),
                    );
                    ui.selectable_value(
                        &mut self.function,
                        MappingFunction::Exponential,
                        MappingFunction::Exponential.name(),
                    );
                    ui.selectable_value(
                        &mut self.function,
                        MappingFunction::Sine,
                        MappingFunction::Sine.name(),
                    );
                    ui.selectable_value(
                        &mut self.function,
                        MappingFunction::Inverse,
                        MappingFunction::Inverse.name(),
                    );
                    ui.selectable_value(
                        &mut self.function,
                        MappingFunction::Custom,
                        MappingFunction::Custom.name(),
                    );
                });

            if self.function == MappingFunction::Custom {
                ui.add_space(6.0);
                self.math_input.ui(ui);
            }

            ui.add_space(10.0);
            ui.label("Grid Range [min, max]:");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut self.grid_min).speed(0.1));
                ui.label("to");
                ui.add(egui::DragValue::new(&mut self.grid_max).speed(0.1));
            });

            if self.grid_min >= self.grid_max {
                ui.colored_label(egui::Color32::RED, "min must be < max");
            }

            ui.add_space(10.0);
            ui.add(egui::Slider::new(&mut self.grid_density, 2..=math_commons::registry::MAX_GRID_SIZE).text("Grid Density:"));

            ui.separator();
            ui.heading("Inspection Point (z)");
            ui.horizontal(|ui| {
                ui.label("Re:");
                ui.add(egui::DragValue::new(&mut self.inspection_point[0]).speed(0.1));
                ui.label("Im:");
                ui.add(egui::DragValue::new(&mut self.inspection_point[1]).speed(0.1));
            });

            let z0 = Complex64::new(self.inspection_point[0], self.inspection_point[1]);
            let w0 = self.eval_z(z0);
            let derivative_fn = |z| self.deriv_z(z);
            let scale_factor = conformal_scale_factor(derivative_fn, z0);

            render_copyable_metric(
                ui,
                "w = f(z)",
                &format!("{:.3} + {:.3}i", w0.re, w0.im),
            );
            render_copyable_metric(
                ui,
                "Scale factor |f'(z)|",
                &format!("{:.3}", scale_factor),
            );
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Conformal Mapping Visualization");

            ui.columns(2, |columns| {
                columns[0].label("z-plane (Input)");
                Plot::new("z_plane_plot")
                    .view_aspect(1.0)
                    .x_axis_label("Re(z)")
                    .y_axis_label("Im(z)")
                    .show(&mut columns[0], |plot_ui| {
                        if self.grid_max > self.grid_min && self.grid_density > 1 {
                            let step =
                                (self.grid_max - self.grid_min) / (self.grid_density as f64 - 1.0);

                            for i in 0..self.grid_density {
                                let x = self.grid_min + (i as f64) * step;
                                let points = vec![[x, self.grid_min], [x, self.grid_max]];
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(100, 150, 250)),
                                );
                            }

                            for i in 0..self.grid_density {
                                let y = self.grid_min + (i as f64) * step;
                                let points = vec![[self.grid_min, y], [self.grid_max, y]];
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(250, 150, 100)),
                                );
                            }
                        }

                        plot_ui.points(
                            egui_plot::Points::new(
                                "",
                                PlotPoints::new(vec![self.inspection_point]),
                            )
                            .color(egui::Color32::RED)
                            .radius(5.0_f32),
                        );
                    });

                columns[1].label("w-plane (Output)");
                Plot::new("w_plane_plot")
                    .view_aspect(1.0)
                    .x_axis_label("Re(w)")
                    .y_axis_label("Im(w)")
                    .show(&mut columns[1], |plot_ui| {
                        if self.grid_max > self.grid_min && self.grid_density > 1 {
                            let step =
                                (self.grid_max - self.grid_min) / (self.grid_density as f64 - 1.0);
                            let resolution = 50;

                            for i in 0..self.grid_density {
                                let x = self.grid_min + (i as f64) * step;
                                let mut points = Vec::with_capacity(resolution);
                                for j in 0..resolution {
                                    let y = self.grid_min
                                        + (self.grid_max - self.grid_min) * (j as f64)
                                            / (resolution as f64 - 1.0);
                                    let z = Complex64::new(x, y);
                                    let w = self.eval_z(z);
                                    points.push([w.re, w.im]);
                                }
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(100, 150, 250)),
                                );
                            }

                            for i in 0..self.grid_density {
                                let y = self.grid_min + (i as f64) * step;
                                let mut points = Vec::with_capacity(resolution);
                                for j in 0..resolution {
                                    let x = self.grid_min
                                        + (self.grid_max - self.grid_min) * (j as f64)
                                            / (resolution as f64 - 1.0);
                                    let z = Complex64::new(x, y);
                                    let w = self.eval_z(z);
                                    points.push([w.re, w.im]);
                                }
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(250, 150, 100)),
                                );
                            }
                        }

                        let z0 = Complex64::new(self.inspection_point[0], self.inspection_point[1]);
                        let w0 = self.eval_z(z0);
                        plot_ui.points(
                            egui_plot::Points::new("", PlotPoints::new(vec![[w0.re, w0.im]]))
                                .color(egui::Color32::RED)
                                .radius(5.0_f32),
                        );
                    });
            });
        });
    }
}

// [cite:partitions_implementation]

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "ComplexMappingTool",
        domain: "analysis",
        tags: &[],
        build: || Box::new(ComplexMappingTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for ComplexMappingTool {
    fn theory_description(&self) -> String {
        let fn_desc = match self.function {
            MappingFunction::ZSquared => "quadratic power map $w = z^2$",
            MappingFunction::Exponential => "entire exponential map $w = e^z$",
            MappingFunction::Sine => "complex trigonometric map $w = \\sin(z)$",
            MappingFunction::Inverse => "inversion map $w = 1/z$",
            MappingFunction::Custom => "custom holomorphic map $w = f(z)$",
        };
        format!(
            "Conformal complex mapping {} preserving angles and orientation under Cauchy-Riemann equations with local scaling factor $|f'(z)|$.",
            fn_desc
        )
    }
    fn phonetic_description(&self) -> String { "Conformal complex mapping and Cauchy-Riemann equations theory".into() }
    fn theory_citation(&self) -> String { "Visual Complex Analysis (Needham, 1997); Complex Analysis (Ahlfors, 1979)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("ZSquared".into(), "w = z^2 doubling angles at the origin".into());
        map.insert("Exponential".into(), "w = e^z mapping vertical lines to circles".into());
        map.insert("Sine".into(), "w = sin(z) mapping strips to upper half-plane".into());
        map.insert("Inverse".into(), "w = 1/z conformal inversion through unit circle".into());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_custom_complex_mapping_evaluation() {
        let mut tool = ComplexMappingTool {
            function: MappingFunction::Custom,
            ..Default::default()
        };
        tool.math_input.set_expression("z^3 - 1/z");

        let z0 = Complex64::new(2.0, 0.0);
        let w0 = tool.eval_z(z0);
        assert!((w0.re - 7.5).abs() < 1e-6);
        assert!(w0.im.abs() < 1e-6);

        let deriv = tool.deriv_z(z0);
        // f'(z) = 3z^2 + 1/z^2 = 3*(4) + 0.25 = 12.25
        assert!((deriv.re - 12.25).abs() < 1e-4);
        assert!(deriv.im.abs() < 1e-4);
    }

    #[test]
    fn test_custom_complex_mapping_division_by_zero_safe() {
        let mut tool = ComplexMappingTool {
            function: MappingFunction::Custom,
            ..Default::default()
        };
        tool.math_input.set_expression("1/z");

        let z_zero = Complex64::new(0.0, 0.0);
        let w_zero = tool.eval_z(z_zero);
        assert_eq!(w_zero, Complex64::new(1e6, 1e6));
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Conformal Mapping",
    "domain": "analysis",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "function": "z^2",
    "grid_bound": 2.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "complex_mapping_starter",
        title: "Conformal Mapping",
        description: "w = f(z): Complex holomorphic functions mapping orthogonal grid coordinate contours.",
        category: "Complex Analysis",
        icon: "🌀",
        domain: "analysis",
        tool_name: Some("ComplexMappingTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
