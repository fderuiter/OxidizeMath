use crate::accessibility::AccessibleHoverText;
use crate::app::dataflow_bus::MAX_DATAFLOW_GRAPH_NODES;
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui::Color32;
use math_explorer::biology::neuroscience::{HodgkinHuxleyNeuron, KuramotoNetwork};
use std::collections::VecDeque;
use std::f32::consts::TAU;

#[path = "kuramoto_ui.rs"]
mod kuramoto_ui;

pub(crate) const MAX_ORDER_HISTORY: usize = 200;

/// Selectable dynamical regime in [`NeuralNetworkVizTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralDynamicsMode {
    /// Biophysical conductance-based Hodgkin-Huxley spiking neurons.
    #[default]
    HodgkinHuxley,
    /// Coupled Kuramoto phase oscillators tracking order parameter $r(t)$.
    Kuramoto,
}

impl NeuralDynamicsMode {
    const fn is_default(&self) -> bool {
        matches!(self, Self::HodgkinHuxley)
    }
}

pub struct NeuralNetworkVizTool {
    pub(crate) neurons: Vec<HodgkinHuxleyNeuron>,
    pub(crate) positions: Vec<[f32; 2]>,
    pub(crate) weights: Vec<Vec<f64>>,
    pub(crate) external_input: Vec<f64>,
    pub(crate) scratch_inputs: Vec<f64>,
    pub(crate) mode: NeuralDynamicsMode,
    pub(crate) kuramoto: KuramotoNetwork,
    pub(crate) order_history: VecDeque<[f64; 2]>,
    pub(crate) is_running: bool,
    pub(crate) dt: f64,
    prev_ring_v: [f64; 5],
    sonification: crate::audio::SonificationFrame,
    pub(crate) last_upstream_graph_rev: u64,
    pub(crate) upstream_linked: bool,
    revision: u64,
}

impl Default for NeuralNetworkVizTool {
    fn default() -> Self {
        let num_neurons = 5;
        let mut neurons = Vec::with_capacity(num_neurons);
        let mut positions = Vec::with_capacity(num_neurons);
        let mut weights = vec![vec![0.0; num_neurons]; num_neurons];
        let mut external_input = vec![0.0; num_neurons];
        let radius = 100.0;
        for i in 0..num_neurons {
            neurons.push(HodgkinHuxleyNeuron::new(-65.0));
            let angle = i as f32 * TAU / num_neurons as f32;
            positions.push([angle.cos() * radius, angle.sin() * radius]);
            weights[(i + 1) % num_neurons][i] = 2.0;
            weights[(i + num_neurons - 1) % num_neurons][i] = -1.0;
        }
        external_input[0] = 10.0;
        let kuramoto = KuramotoNetwork::new(num_neurons, 3.0);
        let mut order_history = VecDeque::with_capacity(MAX_ORDER_HISTORY + 16);
        order_history.push_back([0.0, kuramoto.order_parameter().0]);

        Self {
            neurons,
            positions,
            weights,
            external_input,
            scratch_inputs: vec![0.0; num_neurons],
            mode: NeuralDynamicsMode::HodgkinHuxley,
            kuramoto,
            order_history,
            is_running: false,
            dt: 0.05,
            prev_ring_v: [-65.0; 5],
            sonification: crate::audio::SonificationFrame::silent(),
            last_upstream_graph_rev: 0,
            upstream_linked: false,
            revision: 1,
        }
    }
}

impl NeuralNetworkVizTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    fn reset(&mut self) {
        let keep_mode = self.mode;
        *self = Self::default();
        self.mode = keep_mode;
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.step();
    }

    /// Advances either Hodgkin-Huxley or Kuramoto dynamics by one `dt` step with zero heap allocations.
    pub fn step(&mut self) {
        if self.mode == NeuralDynamicsMode::Kuramoto {
            self.kuramoto.step(self.dt);
            self.record_kuramoto_history_zero_alloc();
            return;
        }
        let num_neurons = self.neurons.len();
        if num_neurons == 0 || self.scratch_inputs.len() < num_neurons {
            return;
        }
        self.scratch_inputs[..num_neurons].fill(0.0);
        for (i, input) in self.scratch_inputs.iter_mut().enumerate().take(num_neurons) {
            for (j, neuron) in self.neurons.iter().enumerate().take(num_neurons) {
                if neuron.v() > 0.0 {
                    *input += self.weights[i][j] * 50.0;
                }
            }
        }

        let mut before = [-65.0; 5];
        for (idx, slot) in before.iter_mut().enumerate() {
            if let Some(neuron) = self.neurons.get(idx) {
                *slot = neuron.v();
            }
        }
        for (i, &synaptic) in self.scratch_inputs.iter().enumerate().take(num_neurons) {
            let total_i_ext = self.external_input[i] + synaptic;
            self.neurons[i].update(self.dt, total_i_ext);
        }
        let mut after = [-65.0; 5];
        for (idx, slot) in after.iter_mut().enumerate() {
            if let Some(neuron) = self.neurons.get(idx) {
                *slot = neuron.v();
            }
        }
        self.prev_ring_v = before;
        self.sonification = crate::audio::mappings::neural_ring_frame(&after, &before, 0.0);
    }

    fn sync_kuramoto_from_weight_matrix(&mut self, spread: f64) {
        let n = self.neurons.len();
        let mut edges = Vec::new();
        for (dst, row) in self.weights.iter().enumerate().take(n) {
            for (src, &w) in row.iter().enumerate().take(n) {
                if src != dst && w > 0.0 {
                    edges.push((src, dst, w));
                }
            }
        }
        self.kuramoto.reconfigure_topology(n, &edges, true, spread);
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "nn_viz_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
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
                self.bump_revision();
            }

            ui.separator();
            self.draw_mode_and_kuramoto_controls(ui);

            if self.mode == NeuralDynamicsMode::HodgkinHuxley {
                ui.separator();
                crate::audio::render_sonification_panel(
                    ui,
                    "neural_network_viz",
                    &self.sonification,
                    !self.is_running,
                );
                ui.separator();
                ui.heading("External Current (uA/cm^2)");
                let max_sliders = self.external_input.len().min(12);
                for i in 0..max_sliders {
                    if ui
                        .add(
                            egui::Slider::new(&mut self.external_input[i], -10.0..=50.0)
                                .text(format!("Neuron {i}")),
                        )
                        .changed()
                    {
                        self.bump_revision();
                    }
                }
            }
        });
    }

    pub(super) fn draw_synaptic_edges(
        &self,
        painter: &egui::Painter,
        center: egui::Pos2,
        palette: &crate::app::SemanticPalette,
    ) {
        let num_neurons = self.neurons.len();
        for i in 0..num_neurons {
            for j in 0..num_neurons {
                let weight = self.weights[i][j];
                if weight != 0.0 {
                    let p1 = center + egui::vec2(self.positions[j][0], self.positions[j][1]);
                    let p2 = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                    let color = if weight > 0.0 {
                        palette.success
                    } else {
                        palette.error
                    };
                    painter.line_segment([p1, p2], (weight.abs().min(3.5) as f32, color));
                }
            }
        }
    }

    fn draw_network(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            if self.mode == NeuralDynamicsMode::Kuramoto {
                self.draw_kuramoto_view(ui);
                return;
            }
            ui.heading("Network Visualization");
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::focusable_noninteractive());
            let label = "Neural Network Visualization: Showing interconnected neurons";
            let _ = response.clone().accessible_hover_text(label);
            let summary = format!("{} interconnected neurons", self.neurons.len());
            crate::accessibility::register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                label,
                Some(&summary),
            );
            crate::accessibility::paint_canvas_focus_ring(ui, &response);
            let center = response.rect.center();
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            self.draw_synaptic_edges(&painter, center, &palette);
            for (i, neuron) in self.neurons.iter().enumerate() {
                let v = neuron.v() as f32;
                let pos = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                let norm_v = ((v + 65.0) / 105.0).clamp(0.0, 1.0);
                let (low, high) = (palette.axis_z, palette.axis_x);
                let lerp = |a: u8, b: u8| {
                    (f32::from(a) + (f32::from(b) - f32::from(a)) * norm_v).round() as u8
                };
                let color = Color32::from_rgb(
                    lerp(low.r(), high.r()),
                    lerp(low.g(), high.g()),
                    lerp(low.b(), high.b()),
                );
                painter.circle(pos, 15.0, color, (1.0, palette.node_stroke));
                painter.text(
                    pos,
                    egui::Align2::CENTER_CENTER,
                    format!("N{i}\n{v:.0}mV"),
                    egui::FontId::proportional(10.0),
                    crate::app::contrasting_text_color(color),
                );
            }
        });
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NeuralNetworkVizState {
    external_input: Vec<f64>,
    #[serde(default)]
    weights: Vec<Vec<f64>>,
    dt: f64,
    is_running: bool,
    #[serde(default, skip_serializing_if = "NeuralDynamicsMode::is_default")]
    mode: NeuralDynamicsMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kuramoto_coupling: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_spread: Option<f64>,
}

impl InteractiveTool for NeuralNetworkVizTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Neural Network Viz"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "neuroscience"
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }

    fn save_state(&self) -> Option<String> {
        let include_kuramoto = self.mode == NeuralDynamicsMode::Kuramoto;
        serde_json::to_string(&NeuralNetworkVizState {
            external_input: self.external_input.clone(),
            weights: self.weights.clone(),
            dt: self.dt,
            is_running: self.is_running,
            mode: self.mode,
            kuramoto_coupling: include_kuramoto.then(|| self.kuramoto.coupling()),
            frequency_spread: include_kuramoto.then(|| self.kuramoto.frequency_spread()),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: NeuralNetworkVizState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid neural network viz state: {e}"))?;
        let n = s.external_input.len();
        let weights_valid = s.weights.is_empty()
            || (s.weights.len() == n
                && s.weights
                    .iter()
                    .all(|row| row.len() == n && row.iter().all(|w| w.is_finite())));
        if !(1..=MAX_DATAFLOW_GRAPH_NODES).contains(&n)
            || !weights_valid
            || !s
                .external_input
                .iter()
                .all(|v| v.is_finite() && (-10.0..=50.0).contains(v))
            || !s.dt.is_finite()
            || !(0.001..=0.5).contains(&s.dt)
        {
            return Err("NeuralNetworkViz parameters or dimensions out of bounds".into());
        }
        if s.weights.is_empty() && n == 5 {
            self.reset();
        } else {
            let radius = 100.0;
            self.neurons = (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
            self.scratch_inputs = vec![0.0; n];
            self.positions = (0..n)
                .map(|i| {
                    let angle = i as f32 * TAU / n as f32;
                    [angle.cos() * radius, angle.sin() * radius]
                })
                .collect();
            self.weights = if s.weights.is_empty() {
                vec![vec![0.0; n]; n]
            } else {
                s.weights
            };
        }
        self.external_input = s.external_input;
        self.dt = s.dt;
        self.is_running = s.is_running;
        self.mode = s.mode;
        let spread = s.frequency_spread.unwrap_or(0.6);
        self.sync_kuramoto_from_weight_matrix(spread);
        if let Some(k) = s.kuramoto_coupling {
            self.kuramoto.set_coupling(k);
        }
        self.bump_revision();
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.poll_upstream_graph(ui.ctx());
        if self.is_running {
            let substeps = if self.mode == NeuralDynamicsMode::Kuramoto {
                2
            } else {
                10
            };
            for _ in 0..substeps {
                self.step();
            }
            ui.ctx().request_repaint();
        }
        self.draw_controls(ui);
        self.draw_network(ui);
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

// [cite:cera_framework]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NeuralNetworkVizTool",
        display_name: "Neural Network Viz",
        domain: "neuroscience",
        tags: &["kuramoto", "synchronization", "hodgkin-huxley"],
        build: || Box::new(NeuralNetworkVizTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for NeuralNetworkVizTool {
    fn theory_description(&self) -> String {
        match self.mode {
            NeuralDynamicsMode::HodgkinHuxley => format!(
                "Coupled network of {} biophysical Hodgkin-Huxley conductance-based neurons: membrane potential dynamics $C_m \\frac{{dV_i}}{{dt}} = I_{{ext,i}} - I_{{ion}}(V_i) + \\sum_j W_{{ij}} s_j(t)$ with synaptic weight matrix $W$.",
                self.neurons.len()
            ),
            NeuralDynamicsMode::Kuramoto => format!(
                "Coupled network of {} Kuramoto phase oscillators: $\\frac{{d\\theta_i}}{{dt}} = \\omega_i + \\frac{{K}}{{N}} \\sum_j A_{{ji}} \\sin(\\theta_j - \\theta_i)$ with complex synchronization order parameter $r(t) e^{{i\\psi(t)}} = \\frac{{1}}{{N}} \\sum_j e^{{i\\theta_j}}$.",
                self.kuramoto.num_nodes()
            ),
        }
    }
    fn phonetic_description(&self) -> String {
        "Coupled Hodgkin-Huxley neural network and Kuramoto phase synchronization theory".into()
    }
    fn theory_citation(&self) -> String {
        "Hodgkin & Huxley (1952); Chemical Oscillations, Waves, and Turbulence (Kuramoto, 1984); Theoretical Neuroscience (Dayan & Abbott, 2001)".into()
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
    "external_input": [15.0, 0.0, 0.0, 0.0, 0.0],
    "weights": [
      [0.0, -1.0, 0.0, 0.0, 2.0],
      [2.0, 0.0, -1.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, 2.0, 0.0, -1.0],
      [-1.0, 0.0, 0.0, 2.0, 0.0]
    ],
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
