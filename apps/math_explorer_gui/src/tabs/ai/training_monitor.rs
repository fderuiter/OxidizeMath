#[path = "training_monitor/canvas.rs"]
mod canvas;
#[cfg(test)]
#[path = "training_monitor/tests.rs"]
mod tests;

use crate::app::SemanticPalette;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::ai::deep_learning_theory::cycle::TrainingLoop;
use math_explorer::ai::deep_learning_theory::linear_algebra::Vector;
use math_explorer::ai::deep_learning_theory::model::TwoLayerMLP;
use math_explorer::ai::optimization::{cross_entropy_softmax_prime, Adam, Optimizer, SGD};
use std::f64::consts::PI;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OptimizerChoice {
    Sgd,
    #[default]
    Adam,
}

impl OptimizerChoice {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Sgd => "SGD",
            Self::Adam => "Adam",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WeightInitChoice {
    #[default]
    HeRelu,
    XavierGlorot,
}

impl WeightInitChoice {
    pub const fn label(self) -> &'static str {
        match self {
            Self::HeRelu => "He (ReLU)",
            Self::XavierGlorot => "Xavier / Glorot",
        }
    }
}

pub struct TrainingMonitorTool {
    training_loop: TrainingLoop<TwoLayerMLP>,
    data: Vec<(Vector, Vector)>,
    is_training: bool,
    epoch: usize,
    learning_rate: f64,
    hidden_dim: usize,
    optimizer_choice: OptimizerChoice,
    weight_init: WeightInitChoice,
    last_grad_norm: f64,
    loss_history: Vec<[f64; 2]>,
    accuracy_history: Vec<[f64; 2]>,
    model_revision: u64,
    boundary_cache: canvas::DecisionBoundaryCache,
}

impl Default for TrainingMonitorTool {
    fn default() -> Self {
        let hidden_dim = 16;
        let learning_rate = 0.01;
        let optimizer_choice = OptimizerChoice::default();
        let weight_init = WeightInitChoice::default();
        let model = build_mlp_model(hidden_dim, weight_init);
        let optimizer = build_optimizer(optimizer_choice, learning_rate);

        Self {
            training_loop: TrainingLoop::new_with_model(model, optimizer),
            data: generate_spiral_data(100),
            is_training: false,
            epoch: 0,
            learning_rate,
            hidden_dim,
            optimizer_choice,
            weight_init,
            last_grad_norm: 0.0,
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            model_revision: 1,
            boundary_cache: canvas::DecisionBoundaryCache::default(),
        }
    }
}

fn build_mlp_model(hidden_dim: usize, weight_init: WeightInitChoice) -> TwoLayerMLP {
    match weight_init {
        WeightInitChoice::HeRelu => TwoLayerMLP::new(2, hidden_dim, 2),
        WeightInitChoice::XavierGlorot => TwoLayerMLP::new_xavier(2, hidden_dim, 2),
    }
}

fn build_optimizer(choice: OptimizerChoice, learning_rate: f64) -> Box<dyn Optimizer<f64>> {
    let lr = if learning_rate.is_finite() && learning_rate > 0.0 {
        learning_rate
    } else {
        0.01
    };
    match choice {
        OptimizerChoice::Sgd => Box::new(SGD::new(lr)),
        OptimizerChoice::Adam => match Adam::new(lr) {
            Ok(adam) => Box::new(adam),
            Err(_) => Box::new(SGD::new(lr)),
        },
    }
}

impl TrainingMonitorTool {
    fn rebuild_optimizer(&mut self) {
        self.training_loop.optimizer = build_optimizer(self.optimizer_choice, self.learning_rate);
    }

    fn reinit_model(&mut self) {
        let model = build_mlp_model(self.hidden_dim, self.weight_init);
        let optimizer = build_optimizer(self.optimizer_choice, self.learning_rate);
        self.training_loop = TrainingLoop::new_with_model(model, optimizer);
        self.epoch = 0;
        self.last_grad_norm = 0.0;
        self.loss_history.clear();
        self.accuracy_history.clear();
        self.is_training = false;
        self.data = generate_spiral_data(100);
        self.model_revision = self.model_revision.wrapping_add(1);
        self.boundary_cache.invalidate();
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

        let count = self.data.len().max(1) as f64;
        let avg_loss = epoch_loss / count;
        let accuracy = correct as f64 / count;

        self.last_grad_norm = self.compute_empirical_grad_norm();
        self.loss_history.push([self.epoch as f64, avg_loss]);
        self.accuracy_history.push([self.epoch as f64, accuracy]);
        self.epoch += 1;
        self.model_revision = self.model_revision.wrapping_add(1);
    }

    fn compute_empirical_grad_norm(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let m = &self.training_loop.model;
        let h_dim = m.layer1.bias.len();
        let mut gw1 = vec![0.0_f64; h_dim * 2];
        let mut gb1 = vec![0.0_f64; h_dim];
        let mut gw2 = vec![0.0_f64; 2 * h_dim];
        let mut gb2 = [0.0_f64; 2];
        let scale = 1.0 / (self.data.len() as f64);

        for (x, y_true) in &self.data {
            let z1 = m.layer1.forward(x);
            let a1 = z1.map(|v| v.max(0.0));
            let logits = m.layer2.forward(&a1);
            let d_logits = cross_entropy_softmax_prime(&logits, y_true);
            gb2[0] += d_logits[0] * scale;
            gb2[1] += d_logits[1] * scale;
            for j in 0..h_dim {
                gw2[j] += d_logits[0] * a1[j] * scale;
                gw2[h_dim + j] += d_logits[1] * a1[j] * scale;
            }
            let d_a1 = m.layer2.weights.transpose() * &d_logits;
            for i in 0..h_dim {
                let dz1 = if z1[i] > 0.0 { d_a1[i] * scale } else { 0.0 };
                gb1[i] += dz1;
                gw1[i * 2] += dz1 * x[0];
                gw1[i * 2 + 1] += dz1 * x[1];
            }
        }
        let sq = gb2[0].powi(2)
            + gb2[1].powi(2)
            + gw2.iter().map(|v| v * v).sum::<f64>()
            + gw1.iter().map(|v| v * v).sum::<f64>()
            + gb1.iter().map(|v| v * v).sum::<f64>();
        sq.sqrt()
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "training_monitor_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
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
                    PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
                    PlaybackAction::Reset => self.reset(),
                }
            }

            ui.separator();
            self.draw_hyperparameters(ui);
            ui.separator();
            self.draw_runtime_metrics(ui);
        });
    }

    fn draw_hyperparameters(&mut self, ui: &mut egui::Ui) {
        ui.heading("Hyperparameters");
        if ui
            .add(
                egui::Slider::new(&mut self.learning_rate, 0.001..=0.5)
                    .logarithmic(true)
                    .text("Learning Rate"),
            )
            .changed()
        {
            self.rebuild_optimizer();
        }
        if ui
            .add(egui::Slider::new(&mut self.hidden_dim, 2..=64).text("Hidden Neurons"))
            .changed()
        {
            self.reinit_model();
        }
        let prev_opt = self.optimizer_choice;
        egui::ComboBox::from_label("Optimizer")
            .selected_text(self.optimizer_choice.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.optimizer_choice, OptimizerChoice::Sgd, "SGD");
                ui.selectable_value(&mut self.optimizer_choice, OptimizerChoice::Adam, "Adam");
            });
        if self.optimizer_choice != prev_opt {
            self.rebuild_optimizer();
        }
        let prev_init = self.weight_init;
        egui::ComboBox::from_label("Weight Init")
            .selected_text(self.weight_init.label())
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.weight_init, WeightInitChoice::HeRelu, "He (ReLU)");
                ui.selectable_value(
                    &mut self.weight_init,
                    WeightInitChoice::XavierGlorot,
                    "Xavier / Glorot",
                );
            });
        if self.weight_init != prev_init {
            self.reinit_model();
        }
    }

    fn draw_runtime_metrics(&self, ui: &mut egui::Ui) {
        let palette = SemanticPalette::from_visuals(ui.visuals());
        ui.strong("RUNTIME METRICS");
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_min_width(ui.available_width());
            ui.label(format!("Epoch: {}", self.epoch));
            if let Some(last_loss) = self.loss_history.last() {
                ui.colored_label(palette.info, format!("Loss: {:.4}", last_loss[1]));
            } else {
                ui.colored_label(palette.muted_text, "Loss: —");
            }
            if let Some(last_acc) = self.accuracy_history.last() {
                ui.colored_label(
                    palette.success,
                    format!("Accuracy: {:.1}%", last_acc[1] * 100.0),
                );
            } else {
                ui.colored_label(palette.muted_text, "Accuracy: —");
            }
            ui.monospace(format!("Grad Norm: {:.4}", self.last_grad_norm));
        });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrainingMonitorState {
    learning_rate: f64,
    hidden_dim: usize,
    #[serde(default)]
    optimizer: OptimizerChoice,
    #[serde(default)]
    weight_init: WeightInitChoice,
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
            optimizer: self.optimizer_choice,
            weight_init: self.weight_init,
            is_training: self.is_training,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: TrainingMonitorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid training monitor state: {e}"))?;
        if !parsed.learning_rate.is_finite()
            || !(0.001..=0.5).contains(&parsed.learning_rate)
            || !(2..=64).contains(&parsed.hidden_dim)
        {
            return Err("Training monitor learning_rate or hidden_dim out of bounds".into());
        }
        self.learning_rate = parsed.learning_rate;
        self.hidden_dim = parsed.hidden_dim;
        self.optimizer_choice = parsed.optimizer;
        self.weight_init = parsed.weight_init;
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
    use rand::seq::SliceRandom;
    use rand::Rng;

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
        tags: &["gradient descent", "sgd", "adam", "loss curve", "mlp", "backpropagation"],
        build: || Box::new(TrainingMonitorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for TrainingMonitorTool {
    fn theory_description(&self) -> String {
        format!(
            "Neural network training dynamics for a two-layer MLP with hidden dimension {}, {} initialization, and {} optimizer ($\\alpha = {:.4}$): empirical risk minimization with live $p(y=0 \\mid x) = 0.50$ decision-boundary contour.",
            self.hidden_dim,
            self.weight_init.label(),
            self.optimizer_choice.label(),
            self.learning_rate
        )
    }
    fn phonetic_description(&self) -> String {
        "Neural network training monitor and optimization theory".into()
    }
    fn theory_citation(&self) -> String {
        "Adam: A Method for Stochastic Optimization (Kingma & Ba, 2014); Delving Deep into Rectifiers (He et al., 2015); Deep Learning (Goodfellow et al., 2016)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
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
    "optimizer": "Adam",
    "weight_init": "HeRelu",
    "is_training": false
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "training_monitor_starter",
        title: "Training Monitor",
        description: "Two-layer MLP spiral classification training monitor tracking live loss, accuracy, and decision-boundary contours.",
        category: "Deep Learning",
        icon: "📈",
        domain: "ai",
        tool_name: Some("TrainingMonitorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
