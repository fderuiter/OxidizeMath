use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Line, Plot, Points};
use math_explorer::ai::deep_learning_theory::cycle::TrainingLoop;
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;
use math_explorer::ai::deep_learning_theory::model::TwoLayerMLP;
use math_explorer::ai::optimization::Adam;
use std::f64::consts::PI;

pub struct TrainingMonitorTool {
    training_loop: TrainingLoop<TwoLayerMLP>,
    data: Vec<(Vector, Vector)>,
    is_training: bool,
    epoch: usize,
    learning_rate: f64,
    hidden_dim: usize,
    loss_history: Vec<[f64; 2]>,
    accuracy_history: Vec<[f64; 2]>,
}

impl Default for TrainingMonitorTool {
    fn default() -> Self {
        let hidden_dim = 16;
        let learning_rate = 0.01;
        let optimizer: Box<dyn math_explorer::ai::optimization::Optimizer<f64>> =
            match Adam::new(learning_rate) {
                Ok(adam) => Box::new(adam),
                Err(_) => Box::new(math_explorer::ai::optimization::SGD::new(learning_rate)),
            };

        Self {
            training_loop: TrainingLoop::new(2, hidden_dim, 2, optimizer),
            data: generate_spiral_data(100),
            is_training: false,
            epoch: 0,
            learning_rate,
            hidden_dim,
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
        }
    }
}

impl TrainingMonitorTool {
    fn reinit_model(&mut self) {
        let optimizer: Box<dyn math_explorer::ai::optimization::Optimizer<f64>> =
            match Adam::new(self.learning_rate) {
                Ok(adam) => Box::new(adam),
                Err(_) => Box::new(math_explorer::ai::optimization::SGD::new(self.learning_rate)),
            };

        self.training_loop = TrainingLoop::new(2, self.hidden_dim, 2, optimizer);
        self.epoch = 0;
        self.loss_history.clear();
        self.accuracy_history.clear();
        self.is_training = false;
        self.data = generate_spiral_data(100);
    }

    pub fn reset(&mut self) {
        self.reinit_model();
    }

    pub(crate) fn step_once(&mut self) {
        self.is_training = false;
        self.step();
    }

    fn step(&mut self) {
        let mut epoch_loss = 0.0;
        let mut correct = 0;

        for (x, y_true) in &self.data {
            let loss = self.training_loop.train_step(x, y_true).unwrap_or(0.0);
            epoch_loss += loss;

            let y_pred = self.training_loop.predict(x);
            if argmax(&y_pred) == argmax(y_true) {
                correct += 1;
            }
        }

        let avg_loss = epoch_loss / self.data.len() as f64;
        let accuracy = correct as f64 / self.data.len() as f64;

        self.loss_history.push([self.epoch as f64, avg_loss]);
        self.accuracy_history.push([self.epoch as f64, accuracy]);
        self.epoch += 1;
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "training_monitor_controls");
        egui::SidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();

            if let Some(action) = PlaybackTransport::in_thread(
                "training_monitor_transport",
                "Training Monitor",
                self.is_training,
            )
            .play_label("▶ Train")
            .step_label("⏭ Step Epoch")
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => self.is_training = true,
                    PlaybackAction::Pause => self.is_training = false,
                    PlaybackAction::Step => self.step_once(),
                    PlaybackAction::Reset => self.reset(),
                }
            }

            ui.separator();
            ui.heading("Hyperparameters");

            if ui
                .add(
                    egui::Slider::new(&mut self.learning_rate, 0.001..=0.1)
                        .logarithmic(true)
                        .text("Learning Rate"),
                )
                .changed()
            {
                self.reinit_model();
            }

            if ui
                .add(egui::Slider::new(&mut self.hidden_dim, 2..=64).text("Hidden Neurons"))
                .changed()
            {
                self.reinit_model();
            }

            ui.separator();
            ui.label(format!("Epoch: {}", self.epoch));
            if let Some(last_loss) = self.loss_history.last() {
                ui.label(format!("Loss: {:.4}", last_loss[1]));
            }
            if let Some(last_acc) = self.accuracy_history.last() {
                ui.label(format!("Accuracy: {:.1}%", last_acc[1] * 100.0));
            }
        });
    }

    fn draw_plots(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.columns(2, |columns| {
                columns[0].heading("Loss History");
                Plot::new("loss_plot")
                    .height(200.0)
                    .show(&mut columns[0], |plot_ui| {
                        plot_ui.line(Line::new("Loss", self.loss_history.clone()));
                    });

                let palette = crate::app::SemanticPalette::from_visuals(columns[1].visuals());
                columns[1].heading("Accuracy History");
                Plot::new("accuracy_plot")
                    .height(200.0)
                    .show(&mut columns[1], |plot_ui| {
                        plot_ui.line(
                            Line::new("Accuracy", self.accuracy_history.clone())
                                .color(palette.success),
                        );
                    });
            });

            ui.separator();

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            ui.heading("Decision Boundary Visualization");
            Plot::new("data_plot").data_aspect(1.0).show(ui, |plot_ui| {
                let mut class_0_points = Vec::new();
                let mut class_1_points = Vec::new();

                for (x, y) in &self.data {
                    let point = [x[0], x[1]];
                    if y[0] > 0.5 {
                        class_0_points.push(point);
                    } else {
                        class_1_points.push(point);
                    }
                }

                plot_ui.points(
                    Points::new("Class 0", class_0_points)
                        .color(palette.axis_x)
                        .radius(4.0_f32),
                );
                plot_ui.points(
                    Points::new("Class 1", class_1_points)
                        .color(palette.axis_z)
                        .radius(4.0_f32),
                );
            });
        });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingMonitorState {
    learning_rate: f64,
    hidden_dim: usize,
    #[serde(default)]
    is_training: bool,
}

impl InteractiveTool for TrainingMonitorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Training Monitor"
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&TrainingMonitorState {
            learning_rate: self.learning_rate,
            hidden_dim: self.hidden_dim,
            is_training: self.is_training,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: TrainingMonitorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid training monitor state: {e}"))?;
        if !parsed.learning_rate.is_finite()
            || !(0.001..=0.1).contains(&parsed.learning_rate)
            || !(2..=64).contains(&parsed.hidden_dim)
        {
            return Err("Training monitor learning_rate or hidden_dim out of bounds".into());
        }
        self.learning_rate = parsed.learning_rate;
        self.hidden_dim = parsed.hidden_dim;
        self.reinit_model();
        self.is_training = parsed.is_training;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        if self.is_training {
            self.step();
            ui.ctx().request_repaint();
        }
        self.draw_controls(ui);
        self.draw_plots(ui);
    }
}

fn generate_spiral_data(points_per_class: usize) -> Vec<(Vector, Vector)> {
    use rand::Rng;
    use rand::seq::SliceRandom;

    let mut rng = oxidize_core::rng::OxidizeRng::default();
    let mut data = Vec::new();
    let n = points_per_class;

    for class_idx in 0..2 {
        for i in 0..n {
            let r = i as f64 / n as f64;
            let t = 2.5 * r * 2.0 * PI + (class_idx as f64 * PI);

            let x = r * t.sin() + rng.gen::<f64>() * 0.1;
            let y = r * t.cos() + rng.gen::<f64>() * 0.1;

            let input = Vector::from_vec(vec![x, y]);
            let target = if class_idx == 0 {
                Vector::from_vec(vec![1.0, 0.0])
            } else {
                Vector::from_vec(vec![0.0, 1.0])
            };
            data.push((input, target));
        }
    }

    data.shuffle(&mut rng);
    data
}

fn argmax(v: &Vector) -> usize {
    let mut max_idx = 0;
    let mut max_val = f64::MIN;
    for (i, &val) in v.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    max_idx
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "TrainingMonitorTool",
        display_name: "Training Monitor",
        domain: "ai",
        tags: &[],
        build: || Box::new(TrainingMonitorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for TrainingMonitorTool {
    fn theory_description(&self) -> String {
        format!(
            "Neural network training dynamics for a two-layer MLP with hidden dimension {} and learning rate $\\alpha = {:.4}$: gradient descent update $\\theta_{{t+1}} = \\theta_t - \\alpha \\nabla L(\\theta_t)$ tracking empirical risk minimization.",
            self.hidden_dim, self.learning_rate
        )
    }
    fn phonetic_description(&self) -> String {
        "Neural network training monitor and optimization theory".into()
    }
    fn theory_citation(&self) -> String {
        "Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014); Deep Learning (Goodfellow et al., 2016)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_monitor_reset_and_step_once() {
        let mut tool = TrainingMonitorTool {
            learning_rate: 0.05,
            hidden_dim: 32,
            epoch: 10,
            is_training: true,
            ..Default::default()
        };
        tool.loss_history.push([1.0, 0.5]);
        tool.accuracy_history.push([1.0, 0.8]);

        tool.reset();

        assert_eq!(tool.learning_rate, 0.05);
        assert_eq!(tool.hidden_dim, 32);
        assert_eq!(tool.epoch, 0);
        assert!(!tool.is_training);
        assert!(tool.loss_history.is_empty());
        assert!(tool.accuracy_history.is_empty());

        tool.step_once();
        assert!(!tool.is_training);
        assert_eq!(tool.epoch, 1);
        assert_eq!(tool.loss_history.len(), 1);
    }

    #[test]
    fn test_training_monitor_learning_rate_reinit_applies_immediately() {
        let mut slow = TrainingMonitorTool {
            learning_rate: 0.001,
            ..Default::default()
        };
        slow.reinit_model();
        slow.step_once();
        slow.step_once();

        let mut fast = TrainingMonitorTool {
            learning_rate: 0.08,
            ..Default::default()
        };
        fast.reinit_model();
        fast.step_once();
        fast.step_once();

        assert_ne!(slow.loss_history[1][1], fast.loss_history[1][1]);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Training Monitor",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "learning_rate": 0.02,
    "hidden_dim": 32,
    "is_training": false
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "training_monitor_starter",
        title: "Training Monitor",
        description: "Two-layer MLP spiral classification training monitor tracking live loss and accuracy curves.",
        category: "Deep Learning",
        icon: "📈",
        domain: "ai",
        tool_name: Some("TrainingMonitorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
