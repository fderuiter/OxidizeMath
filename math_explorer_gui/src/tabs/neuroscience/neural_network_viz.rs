#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "neural_network_viz/bus_sync.rs"]
mod bus_sync;
#[path = "neural_network_viz/presets.rs"]
mod presets;

use crate::accessibility::AccessibleHoverText;
use crate::app::data_bus::ui::{render_follow_workspace_toggle, render_node_budget_banner};
use crate::app::data_bus::{
    SharedGraph, MAX_HH_NEURON_BUDGET, SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES,
};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui::Color32;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::f32::consts::TAU;
use std::sync::Arc;

fn default_true() -> bool {
    true
}

const fn is_true(value: &bool) -> bool {
    *value
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NeuralNetworkVizState {
    external_input: Vec<f64>,
    #[serde(default)]
    weights: Vec<Vec<f64>>,
    dt: f64,
    is_running: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seeded_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_graph: Option<SharedGraph>,
}

pub struct NeuralNetworkVizTool {
    neurons: Vec<HodgkinHuxleyNeuron>,
    positions: Vec<[f32; 2]>,
    weights: Vec<Vec<f64>>,
    external_input: Vec<f64>,
    is_running: bool,
    dt: f64,
    pub(crate) follow_workspace_graph: bool,
    pub(crate) applied_bus_revision: u64,
    clamped_source_nodes: Option<usize>,
    active_shared_graph: Option<Arc<SharedGraph>>,
    seeded_graph: Option<SharedGraph>,
    pending_seed_publish: bool,
    prev_ring_v: [f64; 5],
    sonification: crate::audio::SonificationFrame,
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
        }

        for i in 0..num_neurons {
            let next = (i + 1) % num_neurons;
            let prev = (i + num_neurons - 1) % num_neurons;
            weights[next][i] = 2.0;
            weights[prev][i] = -1.0;
        }
        external_input[0] = 10.0;

        Self {
            neurons,
            positions,
            weights,
            external_input,
            is_running: false,
            dt: 0.05,
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            clamped_source_nodes: None,
            active_shared_graph: None,
            seeded_graph: None,
            pending_seed_publish: false,
            prev_ring_v: [-65.0; 5],
            sonification: crate::audio::SonificationFrame::silent(),
        }
    }
}

impl NeuralNetworkVizTool {
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

        let mut before = [-65.0; 5];
        for (idx, slot) in before.iter_mut().enumerate() {
            if let Some(neuron) = self.neurons.get(idx) {
                *slot = neuron.v();
            }
        }
        for (i, input) in next_inputs.iter().enumerate().take(num_neurons) {
            let total_i_ext = self.external_input[i] + input;
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

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "nn_viz_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();

            let rev_opt = (self.applied_bus_revision > 0).then_some(self.applied_bus_revision);
            if render_follow_workspace_toggle(
                ui,
                &mut self.follow_workspace_graph,
                "Neural Network Viz",
                rev_opt,
            ) && self.follow_workspace_graph
            {
                self.applied_bus_revision = 0;
                self.sync_from_bus(ui.ctx());
            }
            render_node_budget_banner(
                ui,
                self.clamped_source_nodes,
                MAX_HH_NEURON_BUDGET,
                "Hodgkin-Huxley neurons",
            );
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
            crate::audio::render_sonification_panel(
                ui,
                "neural_network_viz",
                &self.sonification,
                !self.is_running,
            );

            ui.separator();
            ui.heading("External Current (uA/cm^2)");
            egui::ScrollArea::vertical()
                .max_height(240.0)
                .show(ui, |ui| {
                    for i in 0..self.external_input.len() {
                        ui.add(
                            egui::Slider::new(&mut self.external_input[i], -10.0..=50.0)
                                .text(format!("Neuron {i}")),
                        );
                    }
                });

            ui.separator();
            ui.label(
                "Signed synapses: +w excitatory edges depolarize postsynaptic neurons; -w inhibitory edges hyperpolarize and quench spike loops.",
            );
        });
    }

    fn draw_synaptic_edges(
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
                    let p1 = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
                    let p2 = center + egui::vec2(self.positions[j][0], self.positions[j][1]);
                    let base = if weight > 0.0 {
                        palette.success
                    } else {
                        palette.error
                    };
                    let color = Color32::from_rgba_unmultiplied(base.r(), base.g(), base.b(), 185);
                    painter.line_segment([p2, p1], (1.0 + weight.abs().min(4.0) as f32, color));
                }
            }
        }
    }

    fn draw_network(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Network Visualization");

            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::focusable_noninteractive(),
            );
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

            for i in 0..self.neurons.len() {
                let v = self.neurons[i].v() as f32;
                let p = center + egui::vec2(self.positions[i][0], self.positions[i][1]);
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
                painter.text(
                    p,
                    egui::Align2::CENTER_CENTER,
                    format!("N{i}\n{v:.0}mV"),
                    egui::FontId::proportional(10.0),
                    crate::app::contrasting_text_color(color),
                );
            }
        });
    }
}

fn validate_loaded_nn_state(s: &NeuralNetworkVizState) -> Result<usize, String> {
    let n = s.external_input.len();
    let weights_valid = s.weights.is_empty()
        || (s.weights.len() == n
            && s.weights
                .iter()
                .all(|row| row.len() == n && row.iter().all(|w| w.is_finite())));
    if !(1..=MAX_HH_NEURON_BUDGET).contains(&n)
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
    for g in [&s.seeded_graph, &s.frozen_graph].into_iter().flatten() {
        g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
    }
    Ok(n)
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

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    fn save_state(&self) -> Option<String> {
        let frozen_graph = (!self.follow_workspace_graph)
            .then(|| self.active_shared_graph.as_deref().cloned())
            .flatten();
        let weights = if self.seeded_graph.is_some() {
            Vec::new()
        } else {
            self.weights.clone()
        };
        serde_json::to_string(&NeuralNetworkVizState {
            external_input: self.external_input.clone(),
            weights,
            dt: self.dt,
            is_running: self.is_running,
            follow_workspace_graph: self.follow_workspace_graph,
            seeded_graph: self.seeded_graph.clone(),
            frozen_graph,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: NeuralNetworkVizState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid neural network viz state: {e}"))?;
        let n = validate_loaded_nn_state(&s)?;

        self.follow_workspace_graph = s.follow_workspace_graph;
        self.seeded_graph = s.seeded_graph.clone();
        self.active_shared_graph = None;
        self.clamped_source_nodes = None;
        self.applied_bus_revision = 0;
        self.external_input = s.external_input.clone();

        if let Some(g) = s.frozen_graph.or(s.seeded_graph) {
            self.apply_shared_graph(&Arc::new(g));
            for (i, &val) in s.external_input.iter().enumerate() {
                if let Some(slot) = self.external_input.get_mut(i) {
                    *slot = val;
                }
            }
            self.pending_seed_publish = self.seeded_graph.is_some();
        } else if s.weights.is_empty() && n == 5 {
            let follow = self.follow_workspace_graph;
            *self = Self::default();
            self.follow_workspace_graph = follow;
            self.external_input = s.external_input;
        } else {
            let radius = 100.0;
            self.neurons = (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
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
            self.pending_seed_publish = false;
            self.prev_ring_v = [-65.0; 5];
            self.sonification = crate::audio::SonificationFrame::silent();
        }

        self.dt = s.dt;
        self.is_running = s.is_running;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_from_bus(&ctx);

        if self.is_running {
            for _ in 0..10 {
                self.step();
            }
            ctx.request_repaint();
        }

        self.draw_controls(ui);
        self.sync_from_bus(&ctx);
        self.draw_network(ui);
    }
}

// [cite:cera_framework]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NeuralNetworkVizTool",
        display_name: "Neural Network Viz",
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
