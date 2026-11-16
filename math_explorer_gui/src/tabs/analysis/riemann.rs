use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Polygon};
use math_explorer::pure_math::analysis::integration::{Integrator, Trapezoidal};

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum FunctionPreset {
    SinX,
    XSquared,
    EToX,
}

impl FunctionPreset {
    fn evaluate(&self, x: f64) -> f64 {
        match self {
            FunctionPreset::SinX => x.sin(),
            FunctionPreset::XSquared => x * x,
            FunctionPreset::EToX => x.exp(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            FunctionPreset::SinX => "f(x) = sin(x)",
            FunctionPreset::XSquared => "f(x) = x^2",
            FunctionPreset::EToX => "f(x) = e^x",
        }
    }
}

pub struct RiemannIntegrationTool {
    preset: FunctionPreset,
    min_x: f64,
    max_x: f64,
    steps: usize,
    result: f64,
    cached_trapezoid_polygons: Vec<Vec<[f64; 2]>>,
    cached_true_curve: Vec<[f64; 2]>,
}

impl Default for RiemannIntegrationTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: FunctionPreset::SinX,
            min_x: 0.0,
            max_x: std::f64::consts::PI,
            steps: 10,
            result: 0.0,
            cached_trapezoid_polygons: Vec::new(),
            cached_true_curve: Vec::new(),
        };
        tool.recalculate();
        tool
    }
}

impl RiemannIntegrationTool {
    fn recalculate(&mut self) {
        if self.min_x >= self.max_x || self.steps == 0 {
            self.result = 0.0;
            self.cached_trapezoid_polygons.clear();
            self.cached_true_curve.clear();
            return;
        }

        let integrator = Trapezoidal::new(self.steps);
        let f = |x: f64| self.preset.evaluate(x);
        let res = integrator.integrate(f, self.min_x, self.max_x, 1e-6);
        self.result = res.value;

        let span = self.max_x - self.min_x;
        let plot_min = self.min_x - span * 0.1;
        let plot_max = self.max_x + span * 0.1;

        self.cached_true_curve = (0..=500)
            .map(|i| {
                let x = plot_min + (plot_max - plot_min) * (i as f64 / 500.0);
                [x, f(x)]
            })
            .collect();

        let dx = (self.max_x - self.min_x) / (self.steps as f64);
        self.cached_trapezoid_polygons = (0..self.steps)
            .map(|i| {
                let x0 = self.min_x + (i as f64) * dx;
                let x1 = x0 + dx;
                let y0 = f(x0);
                let y1 = f(x1);

                vec![[x0, 0.0], [x1, 0.0], [x1, y1], [x0, y0]]
            })
            .collect();
    }
}

impl InteractiveTool for RiemannIntegrationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Riemann Integration"
    }
    fn domain(&self) -> &'static str {
        "analysis"
    }

    fn save_state(&self) -> Option<String> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct RiemannState {
            preset: FunctionPreset,
            min_x: f64,
            max_x: f64,
            steps: usize,
        }
        serde_json::to_string(&RiemannState {
            preset: self.preset,
            min_x: self.min_x,
            max_x: self.max_x,
            steps: self.steps,
        }).ok()
    }

    fn load_state(&mut self, state: &str) {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct RiemannState {
            preset: FunctionPreset,
            min_x: f64,
            max_x: f64,
            steps: usize,
        }
        if let Ok(s) = serde_json::from_str::<RiemannState>(state) {
            self.preset = s.preset;
            self.min_x = s.min_x;
            self.max_x = s.max_x;
            self.steps = s.steps;
            self.recalculate();
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        let mut changed = false;

        egui::SidePanel::left("riemann_controls").show(ctx, |ui| {
            ui.heading("Integration Controls");
            ui.separator();

            ui.label("Function:");
            egui::ComboBox::from_id_salt("function_preset")
                .selected_text(self.preset.name())
                .show_ui(ui, |ui| {
                    if ui
                        .selectable_value(
                            &mut self.preset,
                            FunctionPreset::SinX,
                            FunctionPreset::SinX.name(),
                        )
                        .changed()
                    {
                        changed = true;
                        self.min_x = 0.0;
                        self.max_x = std::f64::consts::PI;
                    }
                    if ui
                        .selectable_value(
                            &mut self.preset,
                            FunctionPreset::XSquared,
                            FunctionPreset::XSquared.name(),
                        )
                        .changed()
                    {
                        changed = true;
                        self.min_x = -2.0;
                        self.max_x = 2.0;
                    }
                    if ui
                        .selectable_value(
                            &mut self.preset,
                            FunctionPreset::EToX,
                            FunctionPreset::EToX.name(),
                        )
                        .changed()
                    {
                        changed = true;
                        self.min_x = -2.0;
                        self.max_x = 2.0;
                    }
                });

            ui.add_space(10.0);
            ui.label("Integration Interval [min, max]");
            ui.horizontal(|ui| {
                if ui
                    .add(egui::DragValue::new(&mut self.min_x).speed(0.1))
                    .changed()
                {
                    changed = true;
                }
                ui.label("to");
                if ui
                    .add(egui::DragValue::new(&mut self.max_x).speed(0.1))
                    .changed()
                {
                    changed = true;
                }
            });

            if self.min_x >= self.max_x {
                ui.colored_label(egui::Color32::RED, "min_x must be < max_x");
            }

            ui.add_space(10.0);
            if ui.add(egui::Slider::new(&mut self.steps, 1..=200).logarithmic(true).text("Partitions (N)"))
                .changed()
            {
                changed = true;
            }

            ui.separator();
            ui.heading("Result");
            render_copyable_metric(ui, "Area", &format!("{:.6}", self.result));
        });

        if changed {
            self.recalculate();
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Visualization");

            // Calculate plotting bounds
            let span = if self.max_x > self.min_x {
                self.max_x - self.min_x
            } else {
                2.0
            };
            let plot_min = self.min_x - span * 0.1;
            let plot_max = self.max_x + span * 0.1;

            let true_line = Line::new(
                self.preset.name(),
                PlotPoints::new(self.cached_true_curve.clone()),
            )
            .name(self.preset.name())
            .color(egui::Color32::LIGHT_BLUE);

            Plot::new("riemann_plot")
                .view_aspect(2.0)
                .x_axis_label("x")
                .y_axis_label("f(x)")
                .show(ui, |plot_ui| {
                    plot_ui.line(true_line);
                    for poly in &self.cached_trapezoid_polygons {
                        let polygon = Polygon::new("", PlotPoints::new(poly.clone()))
                            .fill_color(egui::Color32::from_rgba_unmultiplied(200, 100, 50, 100));
                        plot_ui.polygon(polygon);
                    }

                    // Add a zero line
                    plot_ui.line(
                        Line::new("", PlotPoints::new(vec![[plot_min, 0.0], [plot_max, 0.0]]))
                            .color(egui::Color32::DARK_GRAY),
                    );
                });
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "RiemannIntegrationTool",
        domain: "analysis",
        tags: &[],
        build: || Box::new(RiemannIntegrationTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RiemannIntegrationTool {
    fn theory_description(&self) -> String {
        format!(
            "Numerical approximation of the definite integral $\\int_a^b f(x) dx$ for {} using composite trapezoidal rule: $T_n = \\sum_{{i=1}}^n \\frac{{f(x_{{i-1}}) + f(x_i)}}{{2}} \\Delta x$ with {} steps over interval [{:.2}, {:.2}].",
            self.preset.name(), self.steps, self.min_x, self.max_x
        )
    }
    fn phonetic_description(&self) -> String { "Riemann integration and composite trapezoidal approximation theory".into() }
    fn theory_citation(&self) -> String { "Calculus, Vol. 1 (Apostol, 1967); Principles of Mathematical Analysis (Rudin, 1976)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("SinX".into(), "Integral of sin(x) showing oscillatory area cancellation".into());
        map.insert("XSquared".into(), "Integral of x^2 demonstrating parabolic quadrature".into());
        map.insert("EToX".into(), "Integral of e^x demonstrating exponential growth integration".into());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riemann_tool_default_initialization() {
        let tool = RiemannIntegrationTool::default();
        assert_eq!(tool.steps, 10);
        assert_eq!(tool.cached_trapezoid_polygons.len(), 10);
        assert_eq!(tool.cached_true_curve.len(), 501);
        for poly in &tool.cached_trapezoid_polygons {
            assert_eq!(poly.len(), 4);
        }
        // Sin(x) from 0 to PI area is approximately 2.0
        assert!((tool.result - 2.0).abs() < 0.05);
    }

    #[test]
    fn test_riemann_recalculate_parameter_updates() {
        let mut tool = RiemannIntegrationTool {
            steps: 25,
            preset: FunctionPreset::XSquared,
            min_x: 0.0,
            max_x: 3.0,
            ..Default::default()
        };
        tool.recalculate();

        assert_eq!(tool.cached_trapezoid_polygons.len(), 25);
        assert_eq!(tool.cached_true_curve.len(), 501);
        // Integral of x^2 from 0 to 3 is 9.0
        assert!((tool.result - 9.0).abs() < 0.05);

        // Verify first polygon geometry
        let first_poly = &tool.cached_trapezoid_polygons[0];
        let dx = 3.0 / 25.0;
        assert_eq!(first_poly[0], [0.0, 0.0]);
        assert_eq!(first_poly[1], [dx, 0.0]);
        assert_eq!(first_poly[2], [dx, dx * dx]);
        assert_eq!(first_poly[3], [0.0, 0.0]);

        // Verify last polygon geometry
        let last_poly = &tool.cached_trapezoid_polygons[24];
        let last_x0 = 24.0 * dx;
        let last_x1 = 3.0;
        assert_eq!(last_poly[0], [last_x0, 0.0]);
        assert_eq!(last_poly[1], [last_x1, 0.0]);
        assert_eq!(last_poly[2], [last_x1, last_x1 * last_x1]);
        assert_eq!(last_poly[3], [last_x0, last_x0 * last_x0]);
    }

    #[test]
    fn test_riemann_recalculate_invalid_bounds_clearing() {
        let mut tool = RiemannIntegrationTool::default();
        assert!(!tool.cached_trapezoid_polygons.is_empty());
        assert!(!tool.cached_true_curve.is_empty());

        // When min_x > max_x
        tool.min_x = 5.0;
        tool.max_x = 2.0;
        tool.recalculate();
        assert_eq!(tool.result, 0.0);
        assert!(tool.cached_trapezoid_polygons.is_empty());
        assert!(tool.cached_true_curve.is_empty());

        // When min_x == max_x
        tool.min_x = 2.0;
        tool.max_x = 2.0;
        tool.recalculate();
        assert_eq!(tool.result, 0.0);
        assert!(tool.cached_trapezoid_polygons.is_empty());
        assert!(tool.cached_true_curve.is_empty());
    }

    #[test]
    fn test_riemann_recalculate_zero_steps() {
        let mut tool = RiemannIntegrationTool::default();
        assert!(!tool.cached_trapezoid_polygons.is_empty());
        assert!(!tool.cached_true_curve.is_empty());

        tool.steps = 0;
        tool.recalculate();
        assert_eq!(tool.result, 0.0);
        assert!(tool.cached_trapezoid_polygons.is_empty());
        assert!(tool.cached_true_curve.is_empty());
    }

    #[test]
    fn test_riemann_plot_screenshot_generation() {
        let tool = RiemannIntegrationTool::default();
        assert!(!tool.cached_true_curve.is_empty());

        let datasets = vec![("sin(x)".to_string(), tool.cached_true_curve.clone())];
        let options = egui_plot::export::ExportOptions::default();
        if let Ok(png_data) = egui_plot::export::export_custom_png(&datasets, &options) {
            let _ = std::fs::write("/tmp/riemann_memoized_plot.png", png_data);
        }
    }
}
