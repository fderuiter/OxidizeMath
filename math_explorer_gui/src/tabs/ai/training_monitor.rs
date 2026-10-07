use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui::Color32;
use egui_plot::{Line, Plot, Points};
use math_explorer::ai::deep_learning_theory::cycle::TrainingLoop;
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;
use math_explorer::ai::deep_learning_theory::model::TwoLayerMLP;
use math_explorer::ai::optimization::Adam;
use std::f64::consts::PI;

pub struct TrainingMonitorTool {
    training_loop: TrainingLoop<TwoLayerMLP>,

    // Dataset: (x, y_true)
    data: Vec<(Vector, Vector)>,

    // UI State
    is_training: bool,
    epoch: usize,

    // Hyperparameters
    learning_rate: f64,
    hidden_dim: usize,

    // Metrics History
    loss_history: Vec<[f64; 2]>,
    accuracy_history: Vec<[f64; 2]>,
}

impl Default for TrainingMonitorTool {
    fn default() -> Self {
        let hidden_dim = 16;
        let learning_rate = 0.01;

        // Fallback to SGD if Adam fails (though it shouldn't for f64 constants).
        let optimizer: Box<dyn math_explorer::ai::optimization::Optimizer<f64>> =
            match Adam::new(learning_rate) {
                Ok(adam) => Box::new(adam),
                Err(_) => Box::new(math_explorer::ai::optimization::SGD::new(learning_rate)),
            };

        let training_loop = TrainingLoop::new(2, hidden_dim, 2, optimizer);

        let data = generate_spiral_data(100);

        Self {
            training_loop,
            data,
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
        self.learning_rate = 0.01;
        self.hidden_dim = 16;
        self.reinit_model();
    }

    fn step(&mut self) {
        let mut epoch_loss = 0.0;
        let mut correct = 0;

        // Simple Full Batch (or online for simplicity)
        // Here we do online updates (SGD style loop) over the dataset
        for (x, y_true) in &self.data {
            let loss = self.training_loop.train_step(x, y_true).unwrap_or(0.0);
            epoch_loss += loss;

            // Check accuracy
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
}

impl InteractiveTool for TrainingMonitorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Training Monitor"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        // Run training step if active
        if self.is_training {
            self.step();
            // Request repaint for smooth animation
            ctx.request_repaint();
        }

        egui::SidePanel::left("training_monitor_controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            let training_btn = ui.button(if self.is_training {
                "⏹ Stop Training"
            } else {
                "▶ Start Training"
            });

            if training_btn
                .accessible_hover_text(if self.is_training {
                    "Stop the continuous training loop"
                } else {
                    "Start the continuous training loop"
                })
                .clicked()
            {
                self.is_training = !self.is_training;
            }

            if ui
                .button("↻ Reset to Defaults")
                .accessible_hover_text("Re-initialize neural network weights and reset hyperparameters to default values")
                .clicked()
            {
                self.reset();
            }

            ui.separator();
            ui.heading("Hyperparameters");

            if ui.add(egui::Slider::new(&mut self.learning_rate, 0.001..=0.1).logarithmic(true).text("Learning Rate"))
                .changed()
            {
                // Ideally, update optimizer LR immediately, but for simplicity we apply on reset
                // or we could expose set_lr on the optimizer trait if needed.
                // For this demo, let's just hint that reset is needed or update on reset.
                ui.label("Note: Requires Reset to apply effectively");
            }

            if ui.add(egui::Slider::new(&mut self.hidden_dim, 2..=64).text("Hidden Neurons"))
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

        egui::CentralPanel::default().show(ctx, |ui| {
            // Top: Metrics
            ui.columns(2, |columns| {
                columns[0].heading("Loss History");
                Plot::new("loss_plot")
                    .height(200.0)
                    .show(&mut columns[0], |plot_ui| {
                        plot_ui.line(Line::new("Loss", self.loss_history.clone()));
                    });

                columns[1].heading("Accuracy History");
                Plot::new("accuracy_plot")
                    .height(200.0)
                    .show(&mut columns[1], |plot_ui| {
                        plot_ui.line(
                            Line::new("Accuracy", self.accuracy_history.clone())
                                .color(Color32::GREEN),
                        );
                    });
            });

            ui.separator();

            // Bottom: Data Visualization
            ui.heading("Decision Boundary Visualization");
            Plot::new("data_plot").data_aspect(1.0).show(ui, |plot_ui| {
                // Draw actual data points
                let mut class_0_points = Vec::new();
                let mut class_1_points = Vec::new();

                for (x, y) in &self.data {
                    let point = [x[0], x[1]];
                    if y[0] > 0.5 {
                        // Class 0
                        class_0_points.push(point);
                    } else {
                        // Class 1
                        class_1_points.push(point);
                    }
                }

                plot_ui.points(
                    Points::new("Class 0", class_0_points)
                        .color(Color32::RED)
                        .radius(4.0_f32),
                );
                plot_ui.points(
                    Points::new("Class 1", class_1_points)
                        .color(Color32::BLUE)
                        .radius(4.0_f32),
                );

                // Optionally draw prediction grid (expensive, maybe skip for now or low res)
                // If we want to show decision boundary, we can sample a grid.
            });
        });
    }
}

/// Generates a 2D spiral dataset for classification.
/// Returns a vector of (Input, Target) pairs.
fn generate_spiral_data(points_per_class: usize) -> Vec<(Vector, Vector)> {
    let mut data = Vec::new();
    let n = points_per_class; // Points per class

    for class_idx in 0..2 {
        for i in 0..n {
            let r = i as f64 / n as f64; // Radius
            let t = 2.5 * r * 2.0 * PI + (class_idx as f64 * PI); // Angle

            // allow(entropy_guard)
            let x = r * t.sin() + rand::random::<f64>() * 0.1; // Add noise
            // allow(entropy_guard)
            let y = r * t.cos() + rand::random::<f64>() * 0.1;

            let input = Vector::from_vec(vec![x, y]);

            // One-hot encoding
            let target = if class_idx == 0 {
                Vector::from_vec(vec![1.0, 0.0])
            } else {
                Vector::from_vec(vec![0.0, 1.0])
            };

            data.push((input, target));
        }
    }

    // Shuffle data (simple shuffle)
    use rand::seq::SliceRandom;
    let mut rng = oxidize_core::rng::OxidizeRng::default();
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
        name: "TrainingMonitorTool",
        domain: "ai",
        tags: &[],
        build: || Box::new(TrainingMonitorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for TrainingMonitorTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_monitor_reset() {
        let mut tool = TrainingMonitorTool::default();

        tool.learning_rate = 0.05;
        tool.hidden_dim = 32;
        tool.epoch = 10;
        tool.is_training = true;
        tool.loss_history.push([1.0, 0.5]);
        tool.accuracy_history.push([1.0, 0.8]);

        tool.reset();

        assert_eq!(tool.learning_rate, 0.01);
        assert_eq!(tool.hidden_dim, 16);
        assert_eq!(tool.epoch, 0);
        assert!(!tool.is_training);
        assert!(tool.loss_history.is_empty());
        assert!(tool.accuracy_history.is_empty());
    }
}
