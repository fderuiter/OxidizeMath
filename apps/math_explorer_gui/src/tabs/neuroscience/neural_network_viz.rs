#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "neural_network_viz/bus_sync.rs"]
mod bus_sync;
#[path = "kuramoto_ui.rs"]
mod kuramoto_ui;
#[path = "neural_network_viz/presets.rs"]
mod presets;
#[path = "neural_network_viz/topology.rs"]
mod topology;
#[path = "neural_network_viz/ui.rs"]
mod ui;

use crate::app::data_bus::SharedGraph;
use crate::app::{TopologySource, MAX_NEURAL_SHARED_NODES};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::biology::neuroscience::{
    step_csr_hodgkin_huxley_network, HodgkinHuxleyNeuron, KuramotoNetwork,
};
use std::collections::VecDeque;
use std::f32::consts::TAU;
use std::sync::Arc;

const SYNAPTIC_GAIN: f64 = 50.0;
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
    pub(crate) const fn is_default(&self) -> bool {
        matches!(self, Self::HodgkinHuxley)
    }
}

pub struct NeuralNetworkVizTool {
    pub(crate) neurons: Vec<HodgkinHuxleyNeuron>,
    pub(crate) positions: Vec<[f32; 2]>,
    pub(crate) weights: Vec<Vec<f64>>,
    pub(crate) external_input: Vec<f64>,
    pub(super) local_preset_inputs: Vec<f64>,
    pub(super) local_preset_weights: Vec<Vec<f64>>,
    pub(super) next_inputs: Vec<f64>,
    pub(crate) csr_row_ptr: Vec<usize>,
    pub(crate) csr_col_idx: Vec<usize>,
    pub(crate) csr_weights: Vec<f64>,
    pub topology_source: TopologySource,
    pub(crate) synced_bus_revision: Option<u64>,
    pub(super) clamped_source_nodes: Option<usize>,
    pub(crate) mode: NeuralDynamicsMode,
    pub(crate) kuramoto: KuramotoNetwork,
    pub(crate) order_history: VecDeque<[f64; 2]>,
    pub(crate) is_running: bool,
    pub(crate) dt: f64,
    pub(crate) follow_workspace_graph: bool,
    pub(crate) applied_bus_revision: u64,
    pub(super) active_shared_graph: Option<Arc<SharedGraph>>,
    pub(super) seeded_graph: Option<SharedGraph>,
    pub(super) pending_seed_publish: bool,
    pub(super) prev_ring_v: [f64; 5],
    pub(crate) sonification: crate::audio::SonificationFrame,
    pub(crate) last_upstream_graph_rev: u64,
    pub(crate) upstream_linked: bool,
    revision: u64,
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
        let kuramoto = KuramotoNetwork::new(5, 3.0);
        let mut order_history = VecDeque::with_capacity(MAX_ORDER_HISTORY + 16);
        order_history.push_back([0.0, kuramoto.order_parameter().0]);
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
            mode: NeuralDynamicsMode::HodgkinHuxley,
            kuramoto,
            order_history,
            is_running: false,
            dt: 0.05,
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            active_shared_graph: None,
            seeded_graph: None,
            pending_seed_publish: false,
            prev_ring_v: [-65.0; 5],
            sonification: crate::audio::SonificationFrame::silent(),
            last_upstream_graph_rev: 0,
            upstream_linked: false,
            revision: 1,
        };
        tool.build_default_ring();
        tool
    }
}

impl NeuralNetworkVizTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(super) fn build_default_ring(&mut self) {
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
        for (col, next) in (0..num_neurons).map(|i| (i, (i + 1) % num_neurons)) {
            let prev = (col + num_neurons - 1) % num_neurons;
            weights[next][col] = 2.0;
            weights[prev][col] = -1.0;
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

    pub(super) fn rebuild_csr_from_dense(&mut self) {
        let n = self.neurons.len();
        if self.next_inputs.len() < n {
            self.next_inputs.resize(n.max(MAX_NEURAL_SHARED_NODES), 0.0);
        }
        let (rp, ci, cw) = dense_weights_to_csr(&self.weights);
        self.csr_row_ptr = rp;
        self.csr_col_idx = ci;
        self.csr_weights = cw;
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
        self.save_nn_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_nn_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_upstream_graph(&ctx);
        if !self.upstream_linked {
            self.sync_shared_topology(&ctx);
            if !crate::app::SharedSimulationBus::is_link_topology_enabled(&ctx) {
                self.sync_from_bus(&ctx);
            }
        }
        if self.is_running {
            let substeps = if self.mode == NeuralDynamicsMode::Kuramoto {
                2
            } else {
                10
            };
            for _ in 0..substeps {
                self.step();
            }
            ctx.request_repaint();
        }
        self.draw_controls(ui);
        if !self.upstream_linked && !crate::app::SharedSimulationBus::is_link_topology_enabled(&ctx)
        {
            self.sync_from_bus(&ctx);
        }
        self.draw_network(ui);
    }
}

// [cite:cera_framework]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NeuralNetworkVizTool",
        display_name: "Neural Network Viz",
        domain: "neuroscience",
        tags: &[
            "synapse",
            "coupled neurons",
            "connectome",
            "membrane",
            "spiking",
            "kuramoto",
            "synchronization",
            "hodgkin-huxley",
        ],
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
