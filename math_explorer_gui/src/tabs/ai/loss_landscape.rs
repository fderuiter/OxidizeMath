use crate::framework::InteractiveTool;
use eframe::egui;
use egui::Color32;
use egui_plot::{Plot, Points};
use nalgebra::DVector;

pub struct LossLandscapeTool {
    // Data
    x_data: DVector<f64>,
    y_data: DVector<f64>,

    // Parameters Range
    w_min: f64,
    w_max: f64,
    b_min: f64,
    b_max: f64,

    // Resolution
    resolution: usize,
}

impl Default for LossLandscapeTool {
    fn default() -> Self {
        // Simple linear dataset: y = 2x + 1
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];

        Self {
            x_data: DVector::from_vec(x),
            y_data: DVector::from_vec(y),
            w_min: 0.0,
            w_max: 4.0,
            b_min: -1.0,
            b_max: 3.0,
            resolution: 20, // 20x20 grid = 400 points
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
}

impl InteractiveTool for LossLandscapeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Loss Landscape"
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&LossLandscapeState {
            w_min: self.w_min,
            w_max: self.w_max,
            b_min: self.b_min,
            b_max: self.b_max,
            resolution: self.resolution,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: LossLandscapeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid loss landscape state: {e}"))?;
        if !parsed.w_min.is_finite()
            || !parsed.w_max.is_finite()
            || parsed.w_min >= parsed.w_max
            || !parsed.b_min.is_finite()
            || !parsed.b_max.is_finite()
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
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "loss_landscape_controls");
        egui::SidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Parameters");
            ui.separator();

            ui.label("Weight (Slope) Range");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.w_min)
                        .speed(0.1)
                        .prefix("Min: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.w_max)
                        .speed(0.1)
                        .prefix("Max: "),
                );
            });

            ui.label("Bias (Intercept) Range");
            ui.horizontal(|ui| {
                ui.add(
                    egui::DragValue::new(&mut self.b_min)
                        .speed(0.1)
                        .prefix("Min: "),
                );
                ui.add(
                    egui::DragValue::new(&mut self.b_max)
                        .speed(0.1)
                        .prefix("Max: "),
                );
            });

            ui.separator();
            ui.add(egui::Slider::new(&mut self.resolution, 10..=50).text("Resolution (Grid Size) - points"));

            ui.separator();
            ui.label("True Model: y = 2x + 1");
            ui.label("Loss: Mean Squared Error (MSE)");
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let mut points = Vec::with_capacity(self.resolution * self.resolution);

            let w_step = (self.w_max - self.w_min) / (self.resolution as f64 - 1.0);
            let b_step = (self.b_max - self.b_min) / (self.resolution as f64 - 1.0);

            // Find min/max loss for color normalization
            let mut min_loss = f64::MAX;
            let mut max_loss = f64::MIN;
            let mut loss_grid = Vec::new();

            let n = self.x_data.len() as f64;

            for i in 0..self.resolution {
                for j in 0..self.resolution {
                    let w = self.w_min + i as f64 * w_step;
                    let b = self.b_min + j as f64 * b_step;

                    // Calculate predictions and MSE manually to avoid inner-loop allocation overhead
                    let mut sum_sq_error = 0.0;
                    for k in 0..self.x_data.len() {
                        let y_pred = w * self.x_data[k] + b;
                        let err = y_pred - self.y_data[k];
                        sum_sq_error += err * err;
                    }
                    let loss = sum_sq_error / n;

                    if loss < min_loss {
                        min_loss = loss;
                    }
                    if loss > max_loss {
                        max_loss = loss;
                    }

                    loss_grid.push((w, b, loss));
                }
            }

            // Create points with color
            for (w, b, loss) in loss_grid {
                // Normalize loss to 0..1 for color
                // We use log scale for better visualization as loss can grow fast
                let normalized = if max_loss > min_loss {
                    (loss - min_loss) / (max_loss - min_loss)
                } else {
                    0.0
                };

                let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                let low = palette.axis_z;
                let high = palette.axis_x;
                let lerp = |a: u8, b: u8| {
                    (f64::from(a) + (f64::from(b) - f64::from(a)) * normalized.clamp(0.0, 1.0))
                        .round() as u8
                };

                points.push(
                    egui_plot::Points::new(format!("Loss: {:.4}", loss), vec![[w, b]])
                        .color(Color32::from_rgb(
                            lerp(low.r(), high.r()),
                            lerp(low.g(), high.g()),
                            lerp(low.b(), high.b()),
                        ))
                        .radius(3.0_f32),
                );
            }

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            Plot::new("loss_landscape_plot")
                .x_axis_label("Weight (w)")
                .y_axis_label("Bias (b)")
                .data_aspect(1.0)
                .show(ui, |plot_ui| {
                    for point in points {
                        plot_ui.points(point);
                    }

                    // Mark the true solution (w=2, b=1)
                    plot_ui.points(
                        Points::new("Global Minima (True Solution)", vec![[2.0, 1.0]])
                            .color(palette.success)
                            .radius(6.0_f32)
                            .shape(egui_plot::MarkerShape::Circle),
                    );
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
        "Mean squared error loss landscape over parameter space $(w, b)$: $L(w, b) = \\frac{1}{2n}\\sum_{i=1}^n (w x_i + b - y_i)^2$. The geometric curvature of the landscape is governed by the Hessian matrix $H = \\nabla^2 L$.".into()
    }
    fn phonetic_description(&self) -> String { "Loss landscape and mean squared error optimization theory".into() }
    fn theory_citation(&self) -> String { "Visualizing the Loss Landscape of Neural Nets (Li et al., 2018); Deep Learning (Goodfellow et al., 2016)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
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
