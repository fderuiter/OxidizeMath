//! `SharedSimulationBus` CSR topology synchronization and clamping helpers for [`super::NeuralNetworkVizTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NeuralNetworkVizTool;
use crate::app::{SharedGraphTopology, SharedSimulationBus, MAX_NEURAL_SHARED_NODES};
use eframe::egui;
use math_explorer::biology::neuroscience::HodgkinHuxleyNeuron;
use std::f32::consts::TAU;

impl NeuralNetworkVizTool {
    /// Synchronizes Hodgkin-Huxley neurons (`1..=32`) and directed/symmetric synapses from `SharedSimulationBus`.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision) && self.apply_shared_graph(&snap)
                {
                    self.synced_bus_revision = Some(snap.revision);
                    self.bump_revision();
                }
            }
            None => {
                if self.synced_bus_revision.take().is_some() {
                    self.clamped_source_nodes = None;
                    self.restore_local_circuit();
                    self.bump_revision();
                }
            }
        }
    }

    pub(super) fn restore_local_circuit(&mut self) {
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
        let spread = self.kuramoto.frequency_spread();
        self.sync_kuramoto_from_weight_matrix(spread);
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
        for (pre, (&start, &end)) in snap.row_ptr[..n]
            .iter()
            .zip(&snap.row_ptr[1..=n])
            .enumerate()
        {
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
        let spread = self.kuramoto.frequency_spread();
        self.sync_kuramoto_from_weight_matrix(spread);
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

    /// Returns the active neuron count.
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
}
