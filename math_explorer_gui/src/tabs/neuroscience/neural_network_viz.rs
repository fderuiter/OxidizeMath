use crate::accessibility::AccessibleHoverText;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui::Color32;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::f32::consts::TAU;

pub struct NeuralNetworkVizTool {
    neurons: Vec<HodgkinHuxleyNeuron>,
    positions: Vec<[f32; 2]>,
    weights: Vec<Vec<f64>>,
    external_input: Vec<f64>,
    is_running: bool,
    dt: f64,
}

impl Default for NeuralNetworkVizTool {
    fn default() -> Self {
        let num_neurons = 5;
        let mut neurons = Vec::new();
        let mut positions = Vec::new();
        let mut weights = vec![vec![0.0; num_neurons]; num_neurons];
        let mut external_input = vec![0.0; num_neurons];

        let radius = 100.0;
        for i in 0..num_neurons {
            neurons.push(HodgkinHuxleyNeuron::new(-65.0));
            let angle = i as f32 * TAU / num_neurons as f32;
            positions.push([angle.cos() * radius, angle.sin() * radius]);
        }

        for i in 0..num_neurons {
            let next = (i + 1) % num_neurons;
            let prev = (i + num_neurons - 1) % num_neurons;
            if let Some(row) = weights.get_mut(next) {
                row[i] = 2.0;
            }
            if let Some(row) = weights.get_mut(prev) {
                row[i] = -1.0;
            }
        }

        external_input[0] = 10.0;

        Self {
            neurons,
            positions,
            weights,
            external_input,
            is_running: false,
            dt: 0.05,
        }
    }
}

impl NeuralNetworkVizTool {
    fn reset(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.step();
    }

    fn step(&mut self) {
        let num_neurons = self.neurons.len();
        let mut next_inputs = vec![0.0; num_neurons];

        for (i, input) in next_inputs.iter_mut().enumerate().take(num_neurons) {
            for (j, neuron) in self.neurons.iter().enumerate().take(num_neurons) {
                if neuron.v() > 0.0 {
                    *input += self.weights[i][j] * 50.0;
                }
            }
        }

        for (i, input) in next_inputs.iter().enumerate().take(num_neurons) {
            let total_i_ext = self.external_input[i] + input;
            self.neurons[i].update(self.dt, total_i_ext);
        }
    }

    fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("nn_viz_controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            if let Some(action) = PlaybackTransport::in_thread(
                "nn_viz_transport",
                "Neural Network Viz",
                self.is_running,
            )
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => self.is_running = true,
                    PlaybackAction::Pause => self.is_running = false,
                    PlaybackAction::Step => self.step_once(),
                    PlaybackAction::Reset => self.reset(),
                }
            }

            ui.separator();
            ui.heading("External Current (uA/cm^2)");
            for i in 0..self.external_input.len() {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::Slider::new(&mut self.external_input[i], -10.0..=50.0)
                            .text(format!("Neuron {i}")),
                    );
                });
            }

            ui.separator();
            ui.label(
                "Synaptic activity is modeled simply: if a presynaptic neuron's voltage > 0, a scaled current is added to the postsynaptic neuron.",
            );
        });
    }

    fn draw_network(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Network Visualization");

            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());
            let _ = response.clone().accessible_hover_text(
                "Neural Network Visualization: Showing interconnected neurons",
            );

            let center = response.rect.center();
            let num_neurons = self.neurons.len();

            for i in 0..num_neurons {
                for j in 0..num_neurons {
                    let weight = self.weights[i][j];
                    if weight != 0.0 {
                        let p1 = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                        let p2 = center + egui::vec2(self.positions[j][0], self.positions[j][1]);
                        let color = if weight > 0.0 {
                            Color32::from_rgba_premultiplied(0, 255, 0, 100)
                        } else {
                            Color32::from_rgba_premultiplied(255, 0, 0, 100)
                        };
                        painter.line_segment([p2, p1], (1.0 + weight.abs() as f32, color));
                    }
                }
            }

            for i in 0..num_neurons {
                let v = self.neurons[i].v() as f32;
                let p = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                let normalized_v = ((v + 65.0) / 105.0).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    (normalized_v * 255.0) as u8,
                    ((1.0 - normalized_v) * 100.0) as u8,
                    255,
                );
                painter.circle_filled(p, 15.0, color);
                painter.circle_stroke(p, 15.0, (1.0, Color32::WHITE));
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("N{i}\n{v:.0}mV"),
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
                );
            }
        });
    }
}

impl InteractiveTool for NeuralNetworkVizTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Neural Network Viz"
    }

    fn show(&mut self, ctx: &egui::Context) {
        if self.is_running {
            for _ in 0..10 {
                self.step();
            }
            ctx.request_repaint();
        }

        self.draw_controls(ctx);
        self.draw_network(ctx);
    }
}

// [cite:cera_framework]

inventory::submit! {
    crate::framework::ToolMetadata {
        name: "NeuralNetworkVizTool",
        domain: "neuroscience",
        tags: &[],
        build: || Box::new(NeuralNetworkVizTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for NeuralNetworkVizTool {
    fn theory_description(&self) -> String {
        format!(
            "Coupled network of {} biophysical Hodgkin-Huxley conductance-based neurons: membrane potential dynamics $C_m \\frac{{dV_i}}{{dt}} = I_{{ext,i}} - I_{{ion}}(V_i) + \\sum_j W_{{ij}} s_j(t)$ with synaptic weight matrix $W$.",
            self.neurons.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Coupled Hodgkin-Huxley neural network and synaptic transmission theory".into()
    }
    fn theory_citation(&self) -> String {
        "A Quantitative Description of Membrane Current and Its Application to Conduction and Excitation in Nerve (Hodgkin & Huxley, 1952); Theoretical Neuroscience (Dayan & Abbott, 2001)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_network_viz_step_once_keeps_paused() {
        let mut tool = NeuralNetworkVizTool::default();
        assert!(!tool.is_running);
        let initial_v0 = tool.neurons[0].v();
        tool.step_once();
        assert!(!tool.is_running);
        assert_ne!(tool.neurons[0].v(), initial_v0);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spiking Neural Circuit",
    "domain": "neuroscience",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "layers": [
      4,
      8,
      2
    ],
    "activation": "ReLU"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "neural_network_viz_starter",
        title: "Spiking Neural Circuit",
        description: "Multi-layer synaptic weight propagation and receptive field neuron activation patterns.",
        category: "Neural Circuits",
        icon: "⚡",
        domain: "neuroscience",
        tool_name: Some("NeuralNetworkVizTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
