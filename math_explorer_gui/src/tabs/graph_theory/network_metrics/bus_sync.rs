//! Workspace bus topology and coherence synchronization for [`super::NetworkMetricsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkMetricsTool;
use crate::app::workspace_bus::{
    GraphTopologyChannel, SharedWorkspaceBus, KURAMOTO_COHERENCE_SERIES,
};
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::graph::Graph;

impl NetworkMetricsTool {
    /// Computed graph degeneracy $k$.
    #[cfg(test)]
    #[must_use]
    pub fn degeneracy_value(&self) -> usize {
        self.global_degeneracy
    }

    /// Computed graph treewidth (exact for $|V| \le 24$, min-fill upper bound for $|V| > 24$).
    #[cfg(test)]
    #[must_use]
    pub fn treewidth_value(&self) -> usize {
        self.global_treewidth
    }

    /// Computed 2-approximation vertex cover modulator vertex IDs.
    #[cfg(test)]
    #[must_use]
    pub fn vertex_cover_nodes(&self) -> &std::collections::BTreeSet<usize> {
        &self.vertex_cover_nodes
    }

    /// Computed geodesic diameter.
    #[cfg(test)]
    #[must_use]
    pub fn diameter_value(&self) -> Option<usize> {
        self.global_diameter
    }

    #[cfg(test)]
    pub(crate) fn current_diameter(&self) -> Option<usize> {
        self.global_diameter
    }

    #[cfg(test)]
    pub(crate) fn current_average_clustering(&self) -> Option<f64> {
        self.global_clustering
    }

    #[cfg(test)]
    pub(crate) fn recompute_count(&self) -> u64 {
        self.recompute_count
    }

    pub(crate) fn sync_bus_channels(&mut self, bus: &mut SharedWorkspaceBus) {
        self.ingest_topology_channel(&bus.topology);
        self.latest_linked_coherence = bus
            .signals
            .series(KURAMOTO_COHERENCE_SERIES)
            .and_then(|s| s.latest().map(|pt| pt[1]));
    }

    pub(crate) fn ingest_topology_channel(&mut self, channel: &GraphTopologyChannel) {
        let rev = channel.revision();
        if rev == 0 || self.synced_bus_revision == Some(rev) {
            return;
        }
        self.synced_bus_revision = Some(rev);
        self.using_bus_topology = true;
        self.directed = false;
        self.is_custom_topology = false;
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();

        for (id, &[nx, ny]) in channel.positions().iter().enumerate() {
            let pos = Pos2::new(300.0 + nx * 180.0, 300.0 + ny * 180.0);
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
            self.node_positions.insert(id, pos);
        }
        for &(u, v, w) in channel.edges() {
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u), self.node_indices.get(&v))
            {
                self.edges.push((u, v, w));
                self.graph.add_edge(u_idx, v_idx, w);
            }
        }
        self.bump_revision();
        self.compute_metrics();
    }

    pub(super) fn on_bus_unlinked(&mut self) {
        self.latest_linked_coherence = None;
        if self.using_bus_topology {
            self.using_bus_topology = false;
            self.synced_bus_revision = None;
            self.build_sample_graph();
            self.bump_revision();
        }
    }
}
