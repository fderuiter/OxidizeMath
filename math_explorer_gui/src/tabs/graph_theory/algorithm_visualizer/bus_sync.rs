//! Workspace data bus synchronization for [`super::AlgorithmVisualizerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AlgorithmVisualizerTool;
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{read_shared_graph, SharedGraph};
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::graph::Graph;
use std::sync::Arc;

impl AlgorithmVisualizerTool {
    pub(crate) fn apply_shared_graph(&mut self, shared: Arc<SharedGraph>) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();

        for node in &shared.nodes {
            let pos = Pos2::new(
                60.0_f32 + node.norm_pos[0].clamp(0.0_f32, 1.0_f32) * 300.0_f32,
                60.0_f32 + node.norm_pos[1].clamp(0.0_f32, 1.0_f32) * 260.0_f32,
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
                    let cost = w.abs().max(0.1);
                    self.edges.push((nu.id, nv.id, cost));
                    self.graph.add_edge(u_idx, v_idx, cost);
                }
            }
        }

        if !self
            .start_node
            .is_some_and(|id| self.node_indices.contains_key(&id))
        {
            self.start_node = self.node_positions.keys().copied().min();
        }
        if let Some(id) = self.start_node {
            self.focused_node = id;
        }
        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(shared);
        self.run_algorithm();
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
        if sync_consumer_follow_flag(
            ctx,
            &mut self.follow_workspace_graph,
            "Algorithm Visualizer",
        ) && self.follow_workspace_graph
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

    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    pub(crate) fn visit_count(&self) -> usize {
        self.visit_order.len()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Dijkstra Shortest Path",
    "domain": "graph_theory",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "selected_algorithm": "Dijkstra",
    "start_node": 5,
    "playback_speed": 3.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "algorithm_visualizer_starter",
        title: "Dijkstra Shortest Path",
        description: "Priority-queue shortest path relaxation across weighted undirected graph topologies.",
        category: "Graph Algorithms",
        icon: "🛤️",
        domain: "graph_theory",
        tool_name: Some("AlgorithmVisualizerTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
