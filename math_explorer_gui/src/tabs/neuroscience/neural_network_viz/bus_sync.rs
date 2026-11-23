//! Workspace data bus synchronization for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NeuralNetworkVizTool;
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    read_shared_graph, seed_shared_graph_if_empty, SharedGraph, MAX_HH_NEURON_BUDGET,
};
use eframe::egui;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::sync::Arc;

impl NeuralNetworkVizTool {
    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn synaptic_weight(&self, post: usize, pre: usize) -> f64 {
        self.weights
            .get(post)
            .and_then(|row| row.get(pre))
            .copied()
            .unwrap_or(0.0)
    }

    pub(super) fn reset(&mut self) {
        if let Some(shared) = self.active_shared_graph.clone() {
            for neuron in &mut self.neurons {
                *neuron = HodgkinHuxleyNeuron::new(-65.0);
            }
            self.apply_shared_graph(&shared);
            self.is_running = false;
        } else {
            let follow = self.follow_workspace_graph;
            *self = Self::default();
            self.follow_workspace_graph = follow;
        }
    }

    pub(crate) fn apply_shared_graph(&mut self, shared: &Arc<SharedGraph>) {
        let (clamped, orig_count) = shared.clamped_to_budget(MAX_HH_NEURON_BUDGET);
        self.clamped_source_nodes = orig_count;
        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(Arc::clone(&clamped));

        let n = clamped.nodes.len().max(1);
        self.neurons = (0..n).map(|_| HodgkinHuxleyNeuron::new(-65.0)).collect();
        self.prev_ring_v = [-65.0; 5];
        self.sonification = crate::audio::SonificationFrame::silent();
        if clamped.nodes.is_empty() {
            self.positions = vec![[0.0, 0.0]];
            self.weights = vec![vec![0.0]];
            self.external_input = vec![10.0];
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
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
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
}
