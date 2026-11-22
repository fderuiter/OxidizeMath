//! Workspace data bus synchronization and starter preset for [`super::NetworkMetricsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkMetricsTool;
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{read_shared_graph, SharedGraph};
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::graph::Graph;
use std::sync::Arc;

impl NetworkMetricsTool {
    pub(crate) fn apply_shared_graph(&mut self, shared: Arc<SharedGraph>) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();

        for node in &shared.nodes {
            let pos = Pos2::new(
                50.0_f32 + node.norm_pos[0].clamp(0.0_f32, 1.0_f32) * 320.0_f32,
                50.0_f32 + node.norm_pos[1].clamp(0.0_f32, 1.0_f32) * 280.0_f32,
            );
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(node.id, idx);
            self.node_positions.insert(node.id, pos);
        }

        for &(u, v, w) in &shared.edges {
            if let (Some(nu), Some(nv)) = (shared.nodes.get(u), shared.nodes.get(v)) {
                if let (Some(&u_idx), Some(&v_idx)) =
                    (self.node_indices.get(&nu.id), self.node_indices.get(&nv.id))
                {
                    let edge_w = w.abs().max(0.1);
                    self.edges.push((nu.id, nv.id, w));
                    self.graph.add_edge(u_idx, v_idx, edge_w);
                }
            }
        }

        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(shared);
        self.compute_metrics();
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
        if sync_consumer_follow_flag(ctx, &mut self.follow_workspace_graph, "Network Metrics")
            && self.follow_workspace_graph
        {
            self.applied_bus_revision = 0;
        }
        if !self.follow_workspace_graph {
            return;
        }
        if let Some(shared) = read_shared_graph(ctx) {
            if shared.revision != self.applied_bus_revision {
                self.apply_shared_graph(shared);
            }
        }
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Network Centrality Metrics",
    "domain": "graph_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "selected_metric": "ClosenessCentrality"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_metrics_starter",
        title: "Network Centrality Metrics",
        description: "Degree centrality, closeness centrality, local clustering coefficient, and graph diameter.",
        category: "Network Science",
        icon: "📊",
        domain: "graph_theory",
        tool_name: Some("NetworkMetricsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
