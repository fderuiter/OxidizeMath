use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui::Color32;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::f32::consts::TAU;

pub struct NeuralNetworkVizTool {
    neurons: Vec<HodgkinHuxleyNeuron>,
    /// Positions for drawing the nodes in a circle
    positions: Vec<[f32; 2]>,
    /// Adjacency matrix: `weights[i][j]` is the weight of connection from j to i
    weights: Vec<Vec<f64>>,
    /// Current external input injected to each neuron
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

            // Arrange neurons in a circle
            let angle = i as f32 * TAU / num_neurons as f32;
            positions.push([angle.cos() * radius, angle.sin() * radius]);
        }

        // Define some simple connections (e.g., a ring with excitatory forward and inhibitory backward)
        for i in 0..num_neurons {
            let next = (i + 1) % num_neurons;
            let prev = (i + num_neurons - 1) % num_neurons;

            if let Some(row) = weights.get_mut(next) {
                row[i] = 2.0; // Excitatory connection
            }
            if let Some(row) = weights.get_mut(prev) {
                row[i] = -1.0; // Inhibitory connection
            }
        }

        // Inject some initial current into the first neuron to kickstart
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

    fn step(&mut self) {
        let num_neurons = self.neurons.len();
        let mut next_inputs = vec![0.0; num_neurons];

        // Compute synaptic currents.
        // Simple model: if a presynaptic neuron is firing (V > 0), inject current.
        for (i, input) in next_inputs.iter_mut().enumerate().take(num_neurons) {
            for (j, neuron) in self.neurons.iter().enumerate().take(num_neurons) {
                if neuron.v() > 0.0 {
                    *input += self.weights[i][j] * 50.0; // Synaptic strength scalar
                }
            }
        }

        for (i, input) in next_inputs.iter().enumerate().take(num_neurons) {
            let total_i_ext = self.external_input[i] + input;
            self.neurons[i].update(self.dt, total_i_ext);
        }
    }
}

impl InteractiveTool for NeuralNetworkVizTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Neural Network Viz"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        if self.is_running {
            for _ in 0..10 {
                // Run multiple steps per frame to speed up simulation
                self.step();
            }
            ctx.request_repaint();
        }

        egui::SidePanel::left("nn_viz_controls").show(ctx, |ui| {
            ui.heading("Controls");
            ui.separator();

            if ui
                .button(if self.is_running { "⏸ Pause" } else { "▶ Start" })
                .accessible_hover_text(if self.is_running {
                    "Pause the Neural Network simulation"
                } else {
                    "Start the Neural Network simulation"
                })
                .clicked()
            {
                self.is_running = !self.is_running;
            }

            if ui
                .button("↻ Reset")
                .accessible_hover_text("Reset the simulation to its initial state")
                .clicked()
            {
                self.reset();
            }

            ui.separator();
            ui.heading("External Current (uA/cm^2)");
            for i in 0..self.external_input.len() {
                ui.horizontal(|ui| {
                     
                    ui.add(egui::Slider::new(&mut self.external_input[i], -10.0..=50.0).text(format!("Neuron {}", i)));
                });
            }

            ui.separator();
            ui.label("Synaptic activity is modeled simply: if a presynaptic neuron's voltage > 0, a scaled current is added to the postsynaptic neuron.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Network Visualization");

            // Draw custom graph
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());
            let _ = response.clone().accessible_hover_text(
                "Neural Network Visualization: Showing interconnected neurons",
            );

            let rect = response.rect;
            let center = rect.center();

            let num_neurons = self.neurons.len();

            // Draw edges
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            for i in 0..num_neurons {
                for j in 0..num_neurons {
                    let weight = self.weights[i][j];
                    if weight != 0.0 {
                        let p1 = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                        let p2 = center + egui::vec2(self.positions[j][0], self.positions[j][1]);

                        let base = if weight > 0.0 {
                            palette.success // Excitatory
                        } else {
                            palette.error // Inhibitory
                        };
                        let color = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 170);

                        painter.line_segment([p2, p1], (1.0 + weight.abs() as f32, color));
                    }
                }
            }

            // Draw nodes
            for i in 0..num_neurons {
                let v = self.neurons[i].v() as f32;
                let p = center + egui::vec2(self.positions[i][0], self.positions[i][1]);

                // Color based on voltage: resting (~-65mV) -> spike (~+40mV)
                let normalized_v = ((v + 65.0) / 105.0).clamp(0.0, 1.0);
                let low = palette.axis_z;
                let high = palette.axis_x;
                let lerp = |a: u8, b: u8| {
                    (f32::from(a) + (f32::from(b) - f32::from(a)) * normalized_v).round() as u8
                };
                let color = Color32::from_rgb(
                    lerp(low.r(), high.r()),
                    lerp(low.g(), high.g()),
                    lerp(low.b(), high.b()),
                );

                painter.circle_filled(p, 15.0, color);
                painter.circle_stroke(p, 15.0, (1.0, palette.node_stroke));

                // Draw text label
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("N{}\n{:.0}mV", i, v),
                    egui::FontId::proportional(10.0),
                    crate::app::contrasting_text_color(color),
                );
            }
        });
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
    fn phonetic_description(&self) -> String { "Coupled Hodgkin-Huxley neural network and synaptic transmission theory".into() }
    fn theory_citation(&self) -> String { "A Quantitative Description of Membrane Current and Its Application to Conduction and Excitation in Nerve (Hodgkin & Huxley, 1952); Theoretical Neuroscience (Dayan & Abbott, 2001)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
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
