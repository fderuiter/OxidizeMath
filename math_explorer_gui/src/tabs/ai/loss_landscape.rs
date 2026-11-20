#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{bounded_drag_value, enforce_ordered_drag_range, InteractiveTool};
use eframe::egui;
use egui::Color32;
use egui_plot::{Plot, Points};
use nalgebra::DVector;

pub(crate) const LOSS_PARAM_MIN_BOUND: f64 = -100.0;
pub(crate) const LOSS_PARAM_MAX_BOUND: f64 = 100.0;
const LOSS_MIN_SPAN: f64 = 0.1;

pub struct LossLandscapeTool {
    x_data: DVector<f64>,
    y_data: DVector<f64>,
    pub(crate) w_min: f64,
    pub(crate) w_max: f64,
    pub(crate) b_min: f64,
    pub(crate) b_max: f64,
    pub(crate) resolution: usize,
}

impl Default for LossLandscapeTool {
    fn default() -> Self {
        let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
        Self {
            x_data: DVector::from_vec(x),
            y_data: DVector::from_vec(y),
            w_min: 0.0,
            w_max: 4.0,
            b_min: -1.0,
            b_max: 3.0,
            resolution: 20,
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
        egui::SidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Parameters");
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
            ui.label("True Model: y = 2x + 1");
            ui.label("Loss: Mean Squared Error (MSE)");
        });
    }

    fn compute_loss_points(&self, low: Color32, high: Color32) -> Vec<Points<'static>> {
        if !self.has_finite_parameters() {
            return Vec::new();
        }
        let w_step = (self.w_max - self.w_min) / (self.resolution as f64 - 1.0);
        let b_step = (self.b_max - self.b_min) / (self.resolution as f64 - 1.0);
        let n = self.x_data.len() as f64;
        let mut min_loss = f64::MAX;
        let mut max_loss = f64::MIN;
        let mut loss_grid = Vec::with_capacity(self.resolution * self.resolution);

        for i in 0..self.resolution {
            for j in 0..self.resolution {
                let w = self.w_min + i as f64 * w_step;
                let b = self.b_min + j as f64 * b_step;
                let mut sum_sq_error = 0.0;
                for k in 0..self.x_data.len() {
                    let err = (w * self.x_data[k] + b) - self.y_data[k];
                    sum_sq_error += err * err;
                }
                let loss = sum_sq_error / n;
                if w.is_finite() && b.is_finite() && loss.is_finite() {
                    min_loss = min_loss.min(loss);
                    max_loss = max_loss.max(loss);
                    loss_grid.push((w, b, loss));
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
                Points::new(format!("Loss: {:.4}", loss), vec![[w, b]])
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
