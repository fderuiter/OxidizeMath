//! Workspace data bus synchronization and state persistence for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NeuralDynamicsMode, NeuralNetworkVizTool};
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    read_shared_graph, seed_shared_graph_if_empty, SharedGraph, MAX_HH_NEURON_BUDGET,
    SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES,
};
use crate::app::dataflow_bus::MAX_DATAFLOW_GRAPH_NODES;
use crate::app::{TopologySource, MAX_NEURAL_SHARED_NODES};
use crate::framework::InteractiveTool;
use eframe::egui;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::f32::consts::TAU;
use std::sync::Arc;

const fn default_true() -> bool {
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
    #[serde(default, skip_serializing_if = "TopologySource::is_default")]
    topology_source: TopologySource,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    seeded_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frozen_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "NeuralDynamicsMode::is_default")]
    mode: NeuralDynamicsMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    kuramoto_coupling: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    frequency_spread: Option<f64>,
}

fn validate_loaded_nn_state(s: &NeuralNetworkVizState) -> Result<usize, String> {
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
    for g in [&s.seeded_graph, &s.frozen_graph].into_iter().flatten() {
        g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
    }
    Ok(n)
}

impl NeuralNetworkVizTool {
    pub(crate) fn reset(&mut self) {
        let src = self.topology_source;
        let keep_mode = self.mode;
        let follow = self.follow_workspace_graph;
        if let Some(shared) = self.active_shared_graph.clone() {
            for neuron in &mut self.neurons {
                *neuron = HodgkinHuxleyNeuron::new(-65.0);
            }
            self.apply_shared_graph(&shared);
            self.is_running = false;
            self.topology_source = src;
            self.follow_workspace_graph = follow;
            self.mode = keep_mode;
        } else {
            *self = Self::default();
            self.topology_source = src;
            self.follow_workspace_graph = follow;
            self.mode = keep_mode;
        }
    }

    pub(crate) fn apply_shared_graph(&mut self, shared: &Arc<SharedGraph>) {
        let (clamped, orig_count) = shared.clamped_to_budget(MAX_HH_NEURON_BUDGET);
        self.clamped_source_nodes = orig_count;
        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(Arc::clone(&clamped));

        let n = clamped.nodes.len().max(1);
        self.neurons = (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
        if self.next_inputs.len() < n {
            self.next_inputs.resize(n.max(MAX_NEURAL_SHARED_NODES), 0.0);
        }
        self.prev_ring_v = [-65.0; 5];
        self.sonification = crate::audio::SonificationFrame::silent();
        if clamped.nodes.is_empty() {
            self.positions = vec![[0.0, 0.0]];
            self.weights = vec![vec![0.0]];
            self.external_input = vec![10.0];
            self.rebuild_csr_from_dense();
            let spread = self.kuramoto.frequency_spread();
            self.sync_kuramoto_from_weight_matrix(spread);
            self.bump_revision();
            return;
        }

        self.positions = clamped
            .nodes
            .iter()
            .map(|nd| {
                [
                    (nd.norm_pos[0].clamp(0.0, 1.0) - 0.5) * 230.0,
                    (nd.norm_pos[1].clamp(0.0, 1.0) - 0.5) * 200.0,
                ]
            })
            .collect();
        self.weights = vec![vec![0.0; n]; n];
        for &(u, v, w) in &clamped.edges {
            if u < n && v < n && u != v && w.is_finite() {
                self.weights[u][v] = w;
                self.weights[v][u] = w;
            }
        }
        self.rebuild_csr_from_dense();

        let mut next_ext = vec![0.0; n];
        for (i, slot) in next_ext.iter_mut().enumerate() {
            if let Some(&existing) = self.external_input.get(i) {
                *slot = existing;
            }
        }
        if next_ext.iter().all(|&v| v.abs() < 1e-6) {
            next_ext[0] = 12.0;
        }
        self.external_input = next_ext;
        let spread = self.kuramoto.frequency_spread();
        self.sync_kuramoto_from_weight_matrix(spread);
        self.bump_revision();
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
        if self.topology_source == TopologySource::LocalPreset {
            if self.active_shared_graph.take().is_some() && self.seeded_graph.is_none() {
                self.applied_bus_revision = 0;
                self.clamped_source_nodes = None;
                self.restore_local_circuit();
                self.bump_revision();
            }
            return;
        }
        if sync_consumer_follow_flag(ctx, &mut self.follow_workspace_graph, "Neural Network Viz")
            && self.follow_workspace_graph
        {
            self.applied_bus_revision = 0;
        }
        if self.pending_seed_publish {
            self.pending_seed_publish = false;
            if let Some(ref seeded) = self.seeded_graph {
                let arc = seed_shared_graph_if_empty(ctx, seeded);
                if self.follow_workspace_graph {
                    self.apply_shared_graph(&arc);
                }
            }
        }
        if !self.follow_workspace_graph {
            return;
        }
        if let Some(shared) = read_shared_graph(ctx) {
            if shared.revision != self.applied_bus_revision {
                self.apply_shared_graph(&shared);
            }
        }
    }

    pub(super) fn save_nn_state(&self) -> Option<String> {
        let on_external = self.synced_bus_revision.is_some()
            || self.upstream_linked
            || (self.active_shared_graph.is_some()
                && self.follow_workspace_graph
                && self.seeded_graph.is_none());
        let frozen_graph = (!self.follow_workspace_graph)
            .then(|| self.active_shared_graph.as_deref().cloned())
            .flatten();
        let external_input = if on_external && !self.local_preset_inputs.is_empty() {
            self.local_preset_inputs.clone()
        } else {
            self.external_input.clone()
        };
        let weights = if self.seeded_graph.is_some() || frozen_graph.is_some() {
            Vec::new()
        } else if on_external && !self.local_preset_weights.is_empty() {
            self.local_preset_weights.clone()
        } else {
            self.weights.clone()
        };
        let include_kuramoto = self.mode == NeuralDynamicsMode::Kuramoto;
        serde_json::to_string(&NeuralNetworkVizState {
            external_input,
            weights,
            dt: self.dt,
            is_running: self.is_running,
            topology_source: self.topology_source,
            follow_workspace_graph: self.follow_workspace_graph,
            seeded_graph: self.seeded_graph.clone(),
            frozen_graph,
            mode: self.mode,
            kuramoto_coupling: include_kuramoto.then(|| self.kuramoto.coupling()),
            frequency_spread: include_kuramoto.then(|| self.kuramoto.frequency_spread()),
        })
        .ok()
    }

    pub(super) fn load_nn_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let s: NeuralNetworkVizState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid neural network viz state: {e}"))?;
        let n = validate_loaded_nn_state(&s)?;

        self.synced_bus_revision = None;
        self.last_upstream_graph_rev = 0;
        self.upstream_linked = false;
        self.topology_source = s.topology_source;
        self.follow_workspace_graph = s.follow_workspace_graph;
        self.seeded_graph = s.seeded_graph.clone();
        self.active_shared_graph = None;
        self.clamped_source_nodes = None;
        self.applied_bus_revision = 0;

        if let Some(g) = s.frozen_graph.or(s.seeded_graph) {
            self.apply_shared_graph(&Arc::new(g));
            for (i, &val) in s.external_input.iter().enumerate() {
                if let Some(slot) = self.external_input.get_mut(i) {
                    *slot = val;
                }
            }
            self.pending_seed_publish =
                self.seeded_graph.is_some() && self.topology_source == TopologySource::SharedBus;
        } else {
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
            self.pending_seed_publish = false;
            self.prev_ring_v = [-65.0; 5];
            self.sonification = crate::audio::SonificationFrame::silent();
        }

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
}
