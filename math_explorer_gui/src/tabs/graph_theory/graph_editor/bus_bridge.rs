//! Publishes immutable copy-on-write CSR [`SharedGraphTopology`] snapshots from [`GraphEditorTool`]
//! into the workspace [`SharedSimulationBus`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use crate::app::{
    MAX_SHARED_GRAPH_EDGES, MAX_SHARED_GRAPH_NODES, SharedGraphTopology, SharedSimulationBus,
};
use eframe::egui;
#[cfg(test)]
use eframe::egui::Pos2;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

impl GraphEditorTool {
    /// Computes a deterministic structural + geometric fingerprint for copy-on-write publication.
    #[must_use]
    pub fn topology_bus_fingerprint(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        let is_directed = self.directed_edges || self.braess.config.enabled;
        is_directed.hash(&mut h);
        self.node_positions.len().hash(&mut h);
        for (&id, pos) in &self.node_positions {
            id.hash(&mut h);
            pos.x.to_bits().hash(&mut h);
            pos.y.to_bits().hash(&mut h);
        }
        self.edges.len().hash(&mut h);
        for &(u, v, w) in &self.edges {
            u.hash(&mut h);
            v.hash(&mut h);
            w.to_bits().hash(&mut h);
        }
        h.finish()
    }

    /// Publishes an immutable [`SharedGraphTopology`] snapshot when nodes, positions, or edges change.
    pub fn publish_shared_topology(&self, ctx: &egui::Context) -> Option<Arc<SharedGraphTopology>> {
        let bus = SharedSimulationBus::from_context_or_insert(ctx);
        if self.node_positions.is_empty() && bus.latest_graph().is_none() {
            return None;
        }
        let fp = self.topology_bus_fingerprint();
        let is_directed = self.directed_edges || self.braess.config.enabled;
        let raw_nodes: Vec<(usize, [f32; 2])> = self
            .node_positions
            .iter()
            .map(|(&id, pos)| (id, [pos.x, pos.y]))
            .collect();
        let (snap, changed) =
            bus.publish_graph_if_changed(fp, is_directed, &raw_nodes, &self.edges)?;
        if changed && SharedSimulationBus::is_link_topology_enabled(ctx) {
            ctx.request_repaint();
        }
        Some(snap)
    }

    /// Renders directed-edge toggle and shared topology bus status/clamping badges in the sidebar.
    pub(super) fn show_bus_bridge_controls(&mut self, ui: &mut egui::Ui) {
        let prev = self.directed_edges;
        if ui
            .checkbox(&mut self.directed_edges, "Directed Edges (Synapses)")
            .on_hover_text("Publish edges as directed u -> v synapses on the Shared Simulation Bus")
            .changed()
            && self.directed_edges != prev
        {
            self.bump_revision();
        }
        if self.node_positions.len() > MAX_SHARED_GRAPH_NODES
            || self.edges.len() > MAX_SHARED_GRAPH_EDGES
        {
            ui.small(format!(
                "Shared Bus cap: <= {MAX_SHARED_GRAPH_NODES} nodes, <= {MAX_SHARED_GRAPH_EDGES} edges"
            ));
        }
        let _ = self.publish_shared_topology(ui.ctx());
    }

    /// Programmatically adds a node at `pos`, returning its assigned editor node ID.
    #[cfg(test)]
    pub fn insert_node_at(&mut self, pos: Pos2) -> usize {
        self.push_undo_snapshot();
        let id = self.next_node_id;
        let idx = self.graph.add_node(pos);
        self.node_indices.insert(id, idx);
        self.node_positions.insert(id, pos);
        self.next_node_id += 1;
        self.bump_revision();
        id
    }

    /// Programmatically adds or updates a weighted edge `(u, v, weight)`.
    #[cfg(test)]
    pub fn upsert_weighted_edge(&mut self, u: usize, v: usize, weight: f64) -> bool {
        if u == v
            || !weight.is_finite()
            || !self.node_positions.contains_key(&u)
            || !self.node_positions.contains_key(&v)
        {
            return false;
        }
        self.push_undo_snapshot();
        if let Some(existing) = self.edges.iter_mut().find(|(eu, ev, _)| {
            (*eu == u && *ev == v) || (!self.directed_edges && *eu == v && *ev == u)
        }) {
            existing.2 = weight;
        } else {
            self.edges.push((u, v, weight));
            self.braess.edge_latencies.push(super::AffineLatency {
                a: 0.0,
                b: weight.max(0.0),
            });
        }
        self.rebuild_graph();
        if self.braess.config.enabled {
            self.recompute_braess();
        }
        self.bump_revision();
        true
    }

    /// Programmatically removes any edge connecting `u` and `v`.
    #[cfg(test)]
    pub fn remove_edge_between(&mut self, u: usize, v: usize) -> bool {
        let Some(pos) = self
            .edges
            .iter()
            .position(|&(eu, ev, _)| (eu == u && ev == v) || (eu == v && ev == u))
        else {
            return false;
        };
        self.push_undo_snapshot();
        self.edges.remove(pos);
        if pos < self.braess.edge_latencies.len() {
            self.braess.edge_latencies.remove(pos);
        }
        self.rebuild_graph();
        if self.braess.config.enabled {
            self.recompute_braess();
        }
        self.bump_revision();
        true
    }
}
