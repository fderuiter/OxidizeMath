use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints, Polygon};
use math_explorer::pure_math::analysis::integration::{Integrator, Trapezoidal};

#[derive(PartialEq, Clone, Copy)]
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
}

impl Default for RiemannIntegrationTool {
    fn default() -> Self {
        let mut tool = Self {
            preset: FunctionPreset::SinX,
            min_x: 0.0,
            max_x: std::f64::consts::PI,
            steps: 10,
            result: 0.0,
        };
        tool.recalculate();
        tool
    }
}

impl RiemannIntegrationTool {
    fn recalculate(&mut self) {
        if self.min_x >= self.max_x {
            self.result = 0.0;
            return;
        }

        let integrator = Trapezoidal::new(self.steps);
        let f = |x: f64| self.preset.evaluate(x);
        let res = integrator.integrate(f, self.min_x, self.max_x, 1e-6);
        self.result = res.value;
    }
}

impl InteractiveTool for RiemannIntegrationTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Riemann Integration"
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

            let f = |x: f64| self.preset.evaluate(x);

            // Calculate plotting bounds
            let span = if self.max_x > self.min_x {
                self.max_x - self.min_x
            } else {
                2.0
            };
            let plot_min = self.min_x - span * 0.1;
            let plot_max = self.max_x + span * 0.1;

            // True curve
            let plot_points_true: Vec<[f64; 2]> = (0..=500)
                .map(|i| {
                    let x = plot_min + (plot_max - plot_min) * (i as f64 / 500.0);
                    [x, f(x)]
                })
                .collect();
            let true_line = Line::new(self.preset.name(), PlotPoints::new(plot_points_true))
                .name(self.preset.name())
                .color(egui::Color32::LIGHT_BLUE);

            let mut polygons = Vec::new();

            if self.steps > 0 && self.min_x < self.max_x {
                let dx = (self.max_x - self.min_x) / (self.steps as f64);
                for i in 0..self.steps {
                    let x0 = self.min_x + (i as f64) * dx;
                    let x1 = x0 + dx;
                    let y0 = f(x0);
                    let y1 = f(x1);

                    // Trapezoid points: (x0, 0), (x1, 0), (x1, y1), (x0, y0)
                    let trapezoid_points = vec![[x0, 0.0], [x1, 0.0], [x1, y1], [x0, y0]];

                    let polygon = Polygon::new("", PlotPoints::new(trapezoid_points))
                        .fill_color(egui::Color32::from_rgba_unmultiplied(200, 100, 50, 100));
                    polygons.push(polygon);
                }
            }

            Plot::new("riemann_plot")
                .view_aspect(2.0)
                .x_axis_label("x")
                .y_axis_label("f(x)")
                .show(ui, |plot_ui| {
                    plot_ui.line(true_line);
                    for poly in polygons {
                        plot_ui.polygon(poly);
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
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
