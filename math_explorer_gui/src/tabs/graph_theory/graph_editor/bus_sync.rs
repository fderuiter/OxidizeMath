//! Normalized 2D coordinate and weighted-edge publisher from [`super::GraphEditorTool`]
//! onto [`crate::app::workspace_bus::GraphTopologyChannel`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use crate::app::workspace_bus::{SharedWorkspaceBus, MAX_TOPOLOGY_EDGES, MAX_TOPOLOGY_NODES};
use std::collections::HashMap;

/// Normalized `[-1.0, 1.0]` node positions paired with dense weighted undirected edges.
pub type NormalizedGraphTopology = (Vec<[f32; 2]>, Vec<(usize, usize, f64)>);

impl GraphEditorTool {
    /// Builds a normalized `[-1.0, 1.0]` 2D node layout and dense weighted edge list
    /// capped at `MAX_TOPOLOGY_NODES` (`256`) and `MAX_TOPOLOGY_EDGES` (`4,096`).
    #[must_use]
    pub fn export_normalized_topology(&self) -> NormalizedGraphTopology {
        if self.node_positions.is_empty() {
            return (Vec::new(), Vec::new());
        }
        let n = self.node_positions.len().min(MAX_TOPOLOGY_NODES);
        let mut id_to_dense = HashMap::with_capacity(n);
        let mut raw_coords = Vec::with_capacity(n);
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for (dense_idx, (&node_id, &pos)) in self.node_positions.iter().take(n).enumerate() {
            id_to_dense.insert(node_id, dense_idx);
            let px = if pos.x.is_finite() { pos.x } else { 0.0 };
            let py = if pos.y.is_finite() { pos.y } else { 0.0 };
            min_x = min_x.min(px);
            max_x = max_x.max(px);
            min_y = min_y.min(py);
            max_y = max_y.max(py);
            raw_coords.push([px, py]);
        }

        let normalized_positions = if n == 1 {
            vec![[0.0, 0.0]]
        } else {
            let cx = 0.5 * (min_x + max_x);
            let cy = 0.5 * (min_y + max_y);
            let half_span = (0.5 * (max_x - min_x).max(max_y - min_y)).max(1e-5);
            raw_coords
                .into_iter()
                .map(|[x, y]| {
                    [
                        ((x - cx) / half_span).clamp(-1.0, 1.0),
                        ((y - cy) / half_span).clamp(-1.0, 1.0),
                    ]
                })
                .collect()
        };

        let mut dense_edges = Vec::with_capacity(self.edges.len().min(MAX_TOPOLOGY_EDGES));
        for &(u_id, v_id, weight) in &self.edges {
            if !weight.is_finite() {
                continue;
            }
            if let (Some(&u), Some(&v)) = (id_to_dense.get(&u_id), id_to_dense.get(&v_id)) {
                if u != v {
                    dense_edges.push((u, v, weight));
                    if dense_edges.len() >= MAX_TOPOLOGY_EDGES {
                        break;
                    }
                }
            }
        }
        (normalized_positions, dense_edges)
    }

    /// Publishes local topology changes onto `bus.topology` when this pane is linked.
    pub(crate) fn publish_topology_to_bus(&mut self, bus: &mut SharedWorkspaceBus) {
        if self.last_published_local_rev.is_none()
            && self.node_positions.is_empty()
            && !bus.topology.is_empty()
        {
            self.last_published_local_rev = Some(self.revision);
            return;
        }
        if self.last_published_local_rev == Some(self.revision) {
            return;
        }
        let (positions, edges) = self.export_normalized_topology();
        bus.topology.publish(&positions, &edges);
        self.last_published_local_rev = Some(self.revision);
    }

    pub(super) fn sync_active_pane_bus(&mut self) {
        if crate::app::workspace_bus::with_linked_bus(|bus| self.publish_topology_to_bus(bus))
            .is_none()
        {
            self.last_published_local_rev = None;
        }
    }
}
