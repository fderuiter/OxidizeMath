//! Publishes compact `GraphTopologyPayload` snapshots from `GraphEditorTool` onto `PaneDataflowBus`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use crate::app::dataflow_bus::{
    GraphTopologyPayload, MAX_DATAFLOW_GRAPH_NODES, publish_pane_graph,
};
use eframe::egui;
use std::collections::HashMap;

impl GraphEditorTool {
    /// Exports the live graph as a compact `0..N` [`GraphTopologyPayload`] (`N <= 200`).
    pub fn export_topology_payload(&self) -> Option<GraphTopologyPayload> {
        if self.node_positions.is_empty() {
            return None;
        }
        let cap = self.node_positions.len().min(MAX_DATAFLOW_GRAPH_NODES);
        let mut id_to_compact = HashMap::with_capacity(cap);
        let mut positions = Vec::with_capacity(cap);
        for (&node_id, &pos) in self.node_positions.iter().take(cap) {
            let idx = positions.len();
            id_to_compact.insert(node_id, idx);
            positions.push([pos.x, pos.y]);
        }
        let mut compact_edges = Vec::with_capacity(self.edges.len());
        for &(u_id, v_id, weight) in &self.edges {
            if let (Some(&u), Some(&v)) = (id_to_compact.get(&u_id), id_to_compact.get(&v_id)) {
                compact_edges.push((u, v, weight));
            }
        }
        let mut payload = GraphTopologyPayload::sanitized(
            self.compute_state_revision(),
            positions,
            compact_edges,
            self.braess.config.enabled,
        );
        if self.node_positions.len() > MAX_DATAFLOW_GRAPH_NODES {
            payload.truncated_from = Some(self.node_positions.len());
        }
        Some(payload)
    }

    pub(super) fn publish_dataflow_topology(&self, ctx: &egui::Context) {
        if let Some(payload) = self.export_topology_payload() {
            publish_pane_graph(ctx, payload);
        }
    }
}
