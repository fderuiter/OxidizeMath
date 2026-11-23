#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "neural_network_viz/ui.rs"]
mod ui;

use crate::app::{
    SharedGraphTopology, SharedSimulationBus, TopologySource, MAX_NEURAL_SHARED_NODES,
};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::biology::neuroscience::{step_csr_hodgkin_huxley_network, HodgkinHuxleyNeuron};
use std::f32::consts::TAU;

const SYNAPTIC_GAIN: f64 = 50.0;

pub struct NeuralNetworkVizTool {
    pub(crate) neurons: Vec<HodgkinHuxleyNeuron>,
    pub(crate) positions: Vec<[f32; 2]>,
    pub(crate) weights: Vec<Vec<f64>>,
    pub(crate) external_input: Vec<f64>,
    local_preset_inputs: Vec<f64>,
    local_preset_weights: Vec<Vec<f64>>,
    next_inputs: Vec<f64>,
    pub(crate) csr_row_ptr: Vec<usize>,
    pub(crate) csr_col_idx: Vec<usize>,
    pub(crate) csr_weights: Vec<f64>,
    pub topology_source: TopologySource,
    pub(crate) synced_bus_revision: Option<u64>,
    clamped_source_nodes: Option<usize>,
    pub(crate) is_running: bool,
    dt: f64,
    prev_ring_v: [f64; 5],
    pub(crate) sonification: crate::audio::SonificationFrame,
}

fn dense_weights_to_csr(weights: &[Vec<f64>]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let n = weights.len();
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut csr_weights = Vec::new();
    row_ptr.push(0);
    for pre in 0..n {
        for (post, row) in weights.iter().enumerate().take(n) {
            if let Some(&w) = row.get(pre) {
                if w != 0.0 && w.is_finite() {
                    col_idx.push(post);
                    csr_weights.push(w);
                }
            }
        }
        row_ptr.push(col_idx.len());
    }
    (row_ptr, col_idx, csr_weights)
}

impl Default for NeuralNetworkVizTool {
    fn default() -> Self {
        let mut tool = Self {
            neurons: Vec::new(),
            positions: Vec::new(),
            weights: Vec::new(),
            external_input: Vec::new(),
            local_preset_inputs: Vec::new(),
            local_preset_weights: Vec::new(),
            next_inputs: vec![0.0; MAX_NEURAL_SHARED_NODES],
            csr_row_ptr: Vec::new(),
            csr_col_idx: Vec::new(),
            csr_weights: Vec::new(),
            topology_source: TopologySource::SharedBus,
            synced_bus_revision: None,
            clamped_source_nodes: None,
            is_running: false,
            dt: 0.05,
            prev_ring_v: [-65.0; 5],
            sonification: crate::audio::SonificationFrame::silent(),
        };
        tool.build_default_ring();
        tool
    }
}

impl NeuralNetworkVizTool {
    fn build_default_ring(&mut self) {
        let num_neurons = 5;
        self.neurons = (0..num_neurons)
            .map(|_| HodgkinHuxleyNeuron::new(-65.0))
            .collect();
        let radius = 100.0;
        self.positions = (0..num_neurons)
            .map(|i| {
                let angle = i as f32 * TAU / num_neurons as f32;
                [angle.cos() * radius, angle.sin() * radius]
            })
            .collect();
        let mut weights = vec![vec![0.0; num_neurons]; num_neurons];
        for i in 0..num_neurons {
            let next = (i + 1) % num_neurons;
            let prev = (i + num_neurons - 1) % num_neurons;
            weights[next][i] = 2.0;
            weights[prev][i] = -1.0;
        }
        let mut external_input = vec![0.0; num_neurons];
        external_input[0] = 10.0;
        self.local_preset_inputs = external_input.clone();
        self.local_preset_weights = weights.clone();
        self.external_input = external_input;
        self.weights = weights;
        self.rebuild_csr_from_dense();
        self.prev_ring_v = [-65.0; 5];
        self.sonification = crate::audio::SonificationFrame::silent();
    }

    fn rebuild_csr_from_dense(&mut self) {
        let n = self.neurons.len();
        if self.next_inputs.len() < n {
            self.next_inputs.resize(n.max(MAX_NEURAL_SHARED_NODES), 0.0);
        }
        let (rp, ci, cw) = dense_weights_to_csr(&self.weights);
        self.csr_row_ptr = rp;
        self.csr_col_idx = ci;
        self.csr_weights = cw;
    }

    pub(crate) fn reset(&mut self) {
        let src = self.topology_source;
        *self = Self::default();
        self.topology_source = src;
    }

    /// Synchronizes Hodgkin-Huxley neurons (`1..=32`) and directed/symmetric synapses from `SharedSimulationBus`.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision) && self.apply_shared_graph(&snap)
                {
                    self.synced_bus_revision = Some(snap.revision);
                }
            }
            None => {
                if self.synced_bus_revision.take().is_some() {
                    self.clamped_source_nodes = None;
                    self.restore_local_circuit();
                }
            }
        }
    }

    fn restore_local_circuit(&mut self) {
        let n = self.local_preset_inputs.len();
        if n == 0 {
            self.build_default_ring();
            return;
        }
        self.neurons = (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
        let radius = 100.0;
        self.positions = (0..n)
            .map(|i| {
                let angle = i as f32 * TAU / n as f32;
                [angle.cos() * radius, angle.sin() * radius]
            })
            .collect();
        self.external_input = self.local_preset_inputs.clone();
        self.weights = self.local_preset_weights.clone();
        self.rebuild_csr_from_dense();
    }

    fn apply_shared_graph(&mut self, snap: &SharedGraphTopology) -> bool {
        let raw_n = snap.node_count();
        if raw_n == 0 {
            return false;
        }
        let source_total = snap.original_node_count.max(raw_n);
        let n = raw_n.min(MAX_NEURAL_SHARED_NODES);
        self.clamped_source_nodes =
            (source_total > MAX_NEURAL_SHARED_NODES).then_some(source_total);
        self.neurons = (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
        let mapped = snap.map_positions_to_bounds(-120.0, -120.0, 120.0, 120.0);
        self.positions = mapped.into_iter().take(n).collect();

        let mut dense = vec![vec![0.0; n]; n];
        let mut row_ptr = Vec::with_capacity(n + 1);
        let mut col_idx = Vec::new();
        let mut csr_weights = Vec::new();
        row_ptr.push(0);
        for pre in 0..n {
            let start = snap.row_ptr[pre];
            let end = snap.row_ptr[pre + 1];
            for idx in start..end {
                let post = snap.col_idx[idx];
                let w = snap.weights[idx];
                if post < n && post != pre && w.is_finite() && w != 0.0 {
                    dense[post][pre] = w;
                    col_idx.push(post);
                    csr_weights.push(w);
                }
            }
            row_ptr.push(col_idx.len());
        }
        self.weights = dense;
        self.csr_row_ptr = row_ptr;
        self.csr_col_idx = col_idx;
        self.csr_weights = csr_weights;
        self.external_input = vec![0.0; n];
        self.external_input[0] = 12.0;
        if self.next_inputs.len() < n {
            self.next_inputs.resize(MAX_NEURAL_SHARED_NODES, 0.0);
        }
        true
    }

    /// Returns the human-readable neuron clamping status badge text when `> 32` nodes were clamped.
    #[must_use]
    pub fn clamp_badge_text(&self) -> Option<String> {
        self.clamped_source_nodes
            .map(|orig| format!("Clamped {orig} -> {MAX_NEURAL_SHARED_NODES} neurons"))
    }

    /// Returns the currently synced shared bus revision, if active.
    #[cfg(test)]
    #[must_use]
    pub fn synced_bus_revision(&self) -> Option<u64> {
        self.synced_bus_revision
    }

    /// Returns the active neuron count (`1..=32`).
    #[cfg(test)]
    #[must_use]
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Returns incoming synaptic weight `pre -> post` (`weights[post][pre]`).
    #[cfg(test)]
    #[must_use]
    pub fn synaptic_weight(&self, pre: usize, post: usize) -> f64 {
        self.weights
            .get(post)
            .and_then(|r| r.get(pre))
            .copied()
            .unwrap_or(0.0)
    }

    pub(crate) fn step_once(&mut self) {
        self.is_running = false;
        self.step();
    }

    /// Advances all coupled Hodgkin-Huxley neurons by one `dt` step with zero heap allocations.
    pub fn step(&mut self) {
        let mut before = [-65.0; 5];
        for (idx, slot) in before.iter_mut().enumerate() {
            if let Some(neuron) = self.neurons.get(idx) {
                *slot = neuron.v();
            }
        }
        step_csr_hodgkin_huxley_network(
            &mut self.neurons,
            &self.external_input,
            &mut self.next_inputs,
            &self.csr_row_ptr,
            &self.csr_col_idx,
            &self.csr_weights,
            SYNAPTIC_GAIN,
            0.0,
            self.dt,
        );
        let mut after = [-65.0; 5];
        for (idx, slot) in after.iter_mut().enumerate() {
            if let Some(neuron) = self.neurons.get(idx) {
                *slot = neuron.v();
            }
        }
        self.prev_ring_v = before;
        self.sonification = crate::audio::mappings::neural_ring_frame(&after, &before, 0.0);
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
    #[serde(default, skip_serializing_if = "TopologySource::is_default")]
    topology_source: TopologySource,
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

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("neuron_count", self.neurons.len() as f64),
            ("synapse_count", self.csr_weights.len() as f64),
            (
                "clamped_source_nodes",
                self.clamped_source_nodes.unwrap_or(0) as f64,
            ),
            (
                "synced_bus_revision",
                self.synced_bus_revision.unwrap_or(0) as f64,
            ),
        ]
    }

    fn save_state(&self) -> Option<String> {
        let on_bus = self.synced_bus_revision.is_some();
        serde_json::to_string(&NeuralNetworkVizState {
            external_input: if on_bus {
                self.local_preset_inputs.clone()
            } else {
                self.external_input.clone()
            },
            weights: if on_bus {
                self.local_preset_weights.clone()
            } else {
                self.weights.clone()
            },
            dt: self.dt,
            is_running: self.is_running,
            topology_source: self.topology_source,
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
        if !(1..=MAX_NEURAL_SHARED_NODES).contains(&n)
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
        self.synced_bus_revision = None;
        self.clamped_source_nodes = None;
        self.topology_source = s.topology_source;
        if s.weights.is_empty() && n == 5 {
            self.build_default_ring();
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
        }
        self.external_input = s.external_input;
        self.local_preset_inputs = self.external_input.clone();
        self.local_preset_weights = self.weights.clone();
        self.rebuild_csr_from_dense();
        self.dt = s.dt;
        self.is_running = s.is_running;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.sync_shared_topology(&ctx);
        if self.is_running {
            for _ in 0..10 {
                self.step();
            }
            ctx.request_repaint();
        }

        self.draw_controls(ui);
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
