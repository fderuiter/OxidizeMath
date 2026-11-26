//! Workspace data bus synchronization and starter preset for [`super::NetworkMetricsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MetricType, NetworkMetricsState, NetworkMetricsTool};
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    read_shared_graph, SharedGraph, SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES,
};
use crate::framework::InteractiveTool;
use eframe::egui::{self, Pos2};
use math_explorer::pure_math::graph_theory::graph::Graph;
use std::sync::Arc;

impl NetworkMetricsTool {
    pub(super) fn render_metric_selector(&mut self, ui: &mut egui::Ui) {
        ui.label("Select Metric to Visualize:");
        let mut changed = false;
        for (metric, label) in [
            (MetricType::DegreeCentrality, "Degree Centrality"),
            (MetricType::ClosenessCentrality, "Closeness Centrality"),
            (
                MetricType::ClusteringCoefficient,
                "Local Clustering Coefficient",
            ),
        ] {
            if ui
                .radio_value(&mut self.selected_metric, metric, label)
                .clicked()
            {
                changed = true;
            }
        }
        if changed {
            self.bump_revision();
            self.compute_metrics();
        }
    }

    pub(super) fn render_invariants_summary(&mut self, ui: &mut egui::Ui) {
        ui.heading("Structural Invariants");
        if let Some(d) = self.global_diameter {
            ui.label(format!("Diameter: {d}"));
        }
        if let Some(c) = self.global_clustering {
            ui.label(format!("Avg Clustering Coefficient: {c:.3}"));
        }
        ui.label(format!("Degeneracy k: {}", self.global_degeneracy));
        let tw_mode = if self.node_positions.len() <= 24 {
            "exact"
        } else {
            "min-fill UB"
        };
        ui.label(format!(
            "Treewidth tw(G): {} ({tw_mode})",
            self.global_treewidth
        ));
        ui.label(format!(
            "Vertex Cover |C| (2-approx): {}",
            self.vertex_cover_nodes.len()
        ));
        if ui
            .checkbox(
                &mut self.highlight_vertex_cover,
                "Highlight Vertex Cover Modulator",
            )
            .changed()
        {
            self.bump_revision();
        }
    }

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

        self.directed = false;
        self.is_custom_topology = false;
        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(shared);
        self.bump_revision();
        self.compute_metrics();
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
        if self.topology_source == crate::app::TopologySource::LocalPreset {
            if self.active_shared_graph.take().is_some() && !self.is_custom_topology {
                self.applied_bus_revision = 0;
                self.build_sample_graph();
                self.bump_revision();
            }
            return;
        }
        if sync_consumer_follow_flag(ctx, &mut self.follow_workspace_graph, "Network Metrics")
            && self.follow_workspace_graph
        {
            self.is_custom_topology = false;
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

    /// Returns the active node count in the metrics graph.
    #[cfg(all(
        test,
        feature = "biology",
        feature = "applied",
        feature = "epidemiology",
        feature = "physics",
        feature = "pure_math"
    ))]
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.node_positions.len()
    }

    pub(super) fn save_metrics_state(&self) -> Option<String> {
        let on_bus = self.synced_bus_revision.is_some();
        let custom_topology = (!on_bus && self.is_custom_topology)
            .then(|| self.export_shared_topology().to_compact_state());
        let frozen_graph = (!self.follow_workspace_graph && !self.is_custom_topology)
            .then(|| self.active_shared_graph.as_deref().cloned())
            .flatten();
        serde_json::to_string(&NetworkMetricsState {
            selected_metric: self.selected_metric,
            highlight_vertex_cover: self.highlight_vertex_cover,
            follow_workspace_graph: self.follow_workspace_graph,
            frozen_graph,
            custom_topology,
            topology_source: self.topology_source,
        })
        .ok()
    }

    pub(super) fn load_metrics_state(&mut self, state: &str) -> Result<(), String> {
        self.state_bounds().validate_json(state)?;
        let loaded: NetworkMetricsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid NetworkMetrics state: {e}"))?;
        if let Some(ref g) = loaded.frozen_graph {
            g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
        }
        self.synced_bus_revision = None;
        if let Some(compact) = &loaded.custom_topology {
            compact.validate()?;
            self.active_shared_graph = None;
            self.load_compact_topology(compact);
        } else if let Some(g) = loaded.frozen_graph {
            self.apply_shared_graph(Arc::new(g));
        } else if self.is_custom_topology || self.active_shared_graph.take().is_some() {
            self.applied_bus_revision = 0;
            self.build_sample_graph();
        }
        self.selected_metric = loaded.selected_metric;
        self.highlight_vertex_cover = loaded.highlight_vertex_cover;
        self.topology_source = loaded.topology_source;
        self.follow_workspace_graph =
            loaded.follow_workspace_graph && loaded.custom_topology.is_none();
        if self.follow_workspace_graph {
            self.applied_bus_revision = 0;
        }
        self.bump_revision();
        self.compute_metrics();
        Ok(())
    }
}
