use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::pure_math::analysis::complex::mapping::conformal_scale_factor;
use num_complex::Complex64;

#[derive(PartialEq, Clone, Copy)]
enum MappingFunction {
    ZSquared,
    Exponential,
    Sine,
    Inverse,
}

impl MappingFunction {
    fn name(&self) -> &'static str {
        match self {
            MappingFunction::ZSquared => "w = z^2",
            MappingFunction::Exponential => "w = e^z",
            MappingFunction::Sine => "w = sin(z)",
            MappingFunction::Inverse => "w = 1/z",
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
        }
    }
}

pub struct ComplexMappingTool {
    function: MappingFunction,
    grid_min: f64,
    grid_max: f64,
    grid_density: usize,
    inspection_point: [f64; 2],
}

impl Default for ComplexMappingTool {
    fn default() -> Self {
        Self {
            function: MappingFunction::ZSquared,
            grid_min: -2.0,
            grid_max: 2.0,
            grid_density: 10,
            inspection_point: [1.0, 1.0],
        }
    }
}

impl InteractiveTool for ComplexMappingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Complex Mapping"
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
                });

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
            let w0 = self.function.evaluate(z0);
            let derivative_fn = |z| self.function.derivative(z);
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

            // Create two plots side by side
            ui.columns(2, |columns| {
                columns[0].label("z-plane (Input)");
                Plot::new("z_plane_plot")
                    .view_aspect(1.0)
                    .x_axis_label("Re(z)")
                    .y_axis_label("Im(z)")
                    .show(&mut columns[0], |plot_ui| {
                        // Draw grid lines
                        if self.grid_max > self.grid_min && self.grid_density > 1 {
                            let step =
                                (self.grid_max - self.grid_min) / (self.grid_density as f64 - 1.0);

                            // Vertical lines (constant x)
                            for i in 0..self.grid_density {
                                let x = self.grid_min + (i as f64) * step;
                                let points = vec![[x, self.grid_min], [x, self.grid_max]];
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(100, 150, 250)),
                                );
                            }

                            // Horizontal lines (constant y)
                            for i in 0..self.grid_density {
                                let y = self.grid_min + (i as f64) * step;
                                let points = vec![[self.grid_min, y], [self.grid_max, y]];
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(250, 150, 100)),
                                );
                            }
                        }

                        // Inspection point
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
                            let resolution = 50; // points per line

                            // Map vertical lines (constant x)
                            for i in 0..self.grid_density {
                                let x = self.grid_min + (i as f64) * step;
                                let mut points = Vec::with_capacity(resolution);
                                for j in 0..resolution {
                                    let y = self.grid_min
                                        + (self.grid_max - self.grid_min) * (j as f64)
                                            / (resolution as f64 - 1.0);
                                    let z = Complex64::new(x, y);
                                    let w = self.function.evaluate(z);
                                    points.push([w.re, w.im]);
                                }
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(100, 150, 250)),
                                );
                            }

                            // Map horizontal lines (constant y)
                            for i in 0..self.grid_density {
                                let y = self.grid_min + (i as f64) * step;
                                let mut points = Vec::with_capacity(resolution);
                                for j in 0..resolution {
                                    let x = self.grid_min
                                        + (self.grid_max - self.grid_min) * (j as f64)
                                            / (resolution as f64 - 1.0);
                                    let z = Complex64::new(x, y);
                                    let w = self.function.evaluate(z);
                                    points.push([w.re, w.im]);
                                }
                                plot_ui.line(
                                    Line::new("", PlotPoints::new(points))
                                        .color(egui::Color32::from_rgb(250, 150, 100)),
                                );
                            }
                        }

                        // Mapped inspection point
                        let z0 = Complex64::new(self.inspection_point[0], self.inspection_point[1]);
                        let w0 = self.function.evaluate(z0);
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
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
