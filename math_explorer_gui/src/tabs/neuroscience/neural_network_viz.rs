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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NeuralNetworkVizState {
    external_input: Vec<f64>,
    dt: f64,
    is_running: bool,
}

impl InteractiveTool for NeuralNetworkVizTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Neural Network Viz"
    }
    fn domain(&self) -> &'static str {
        "neuroscience"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&NeuralNetworkVizState {
            external_input: self.external_input.clone(),
            dt: self.dt,
            is_running: self.is_running,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: NeuralNetworkVizState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid neural network viz state: {e}"))?;
        if parsed.external_input.len() != self.neurons.len()
            || !parsed
                .external_input
                .iter()
                .all(|v| v.is_finite() && (-10.0..=50.0).contains(v))
            || !parsed.dt.is_finite()
            || !(0.001..=0.5).contains(&parsed.dt)
        {
            return Err("NeuralNetworkViz external_input or dt out of bounds".into());
        }
        self.reset();
        self.external_input = parsed.external_input;
        self.dt = parsed.dt;
        self.is_running = parsed.is_running;
        Ok(())
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
            for i in 0..num_neurons {
                for j in 0..num_neurons {
                    let weight = self.weights[i][j];
                    if weight != 0.0 {
                        let p1 = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                        let p2 = center + egui::vec2(self.positions[j][0], self.positions[j][1]);

                        let color = if weight > 0.0 {
                            Color32::from_rgba_premultiplied(0, 255, 0, 100) // Excitatory: green
                        } else {
                            Color32::from_rgba_premultiplied(255, 0, 0, 100) // Inhibitory: red
                        };

                        // To show direction, we can draw a simple line, but offset slightly
                        // Here, just draw a line
                        painter.line_segment([p2, p1], (1.0 + weight.abs() as f32, color));
                    }
                }
            }

            // Draw nodes
            for i in 0..num_neurons {
                let v = self.neurons[i].v() as f32;
                let p = center + egui::vec2(self.positions[i][0], self.positions[i][1]);

                // Color based on voltage
                // Resting is ~-65, Spike is ~+40
                let normalized_v = ((v + 65.0) / 105.0).clamp(0.0, 1.0);
                let color = Color32::from_rgb(
                    (normalized_v * 255.0) as u8,
                    ((1.0 - normalized_v) * 100.0) as u8, // Base blue-ish
                    255,
                );

                painter.circle_filled(p, 15.0, color);
                painter.circle_stroke(p, 15.0, (1.0, Color32::WHITE));

                // Draw text label
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("N{}\n{:.0}mV", i, v),
                    egui::FontId::proportional(10.0),
                    Color32::WHITE,
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
    "external_input": [12.0, 0.0, 0.0, 0.0, 0.0],
    "dt": 0.05,
    "is_running": false
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "neural_network_viz_starter",
        title: "Spiking Neural Circuit",
        description: "Coupled 5-neuron Hodgkin-Huxley ring circuit with excitatory and inhibitory synaptic currents.",
        category: "Neural Circuits",
        icon: "⚡",
        domain: "neuroscience",
        tool_name: Some("NeuralNetworkVizTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
