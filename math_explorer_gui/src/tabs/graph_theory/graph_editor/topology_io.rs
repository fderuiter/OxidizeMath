//! `GraphTopologyDocument` import and export adapter for [`super::GraphEditorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use crate::accessibility::AccessibleHoverText;
use crate::tabs::graph_theory::topology_sync::extract_indexed_canvas_graph;
use eframe::egui;
use math_explorer::pure_math::graph_theory::graph::Graph;
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};
use math_explorer::pure_math::graph_theory::layout::{
    circular_layout, force_directed_layout, grid_layout,
};

impl GraphEditorTool {
    /// Builds a [`GraphTopologyDocument`] from the editor's current nodes, edges, and Braess routing state.
    pub fn build_topology_document(&self) -> Result<GraphTopologyDocument, String> {
        let braess_on = self.braess.config.enabled;
        let mut doc = GraphTopologyDocument::new("GraphEditor", braess_on);
        doc.attributes
            .insert("braess_enabled".to_string(), braess_on.to_string());
        if braess_on {
            doc.attributes.insert(
                "total_demand".to_string(),
                format!("{:.3}", self.braess.config.total_demand),
            );
            if let Ok(summary) = &self.braess.status {
                doc.attributes.insert(
                    "wardrop_cost".to_string(),
                    format!("{:.4}", summary.commuter_latency),
                );
            }
        }

        for (&id, &pos) in &self.node_positions {
            let label = if braess_on && id == self.braess.config.source_node {
                format!("{id}:s")
            } else if braess_on && id == self.braess.config.sink_node {
                format!("{id}:t")
            } else {
                id.to_string()
            };
            let mut node = TopologyNode::new(id.to_string(), label)
                .with_position(f64::from(pos.x), f64::from(pos.y));
            if self.selected_node == Some(id) {
                node.shape = TopologyNodeShape::DoubleCircle;
                node.fill_hex = Some("#16A34A".to_string());
            } else {
                node.fill_hex = Some("#2563EB".to_string());
            }
            doc.nodes.push(node);
        }

        for (idx, &(u, v, weight)) in self.edges.iter().enumerate() {
            if !self.node_positions.contains_key(&u) || !self.node_positions.contains_key(&v) {
                continue;
            }
            let mut edge = TopologyEdge::new(u.to_string(), v.to_string(), weight);
            if braess_on {
                if let Some(lat) = self.braess.edge_latencies.get(idx) {
                    edge.label = Some(format!("{:.1}f+{:.1}", lat.a, lat.b));
                    edge.attributes
                        .insert("latency_a".to_string(), format!("{:.3}", lat.a));
                    edge.attributes
                        .insert("latency_b".to_string(), format!("{:.3}", lat.b));
                    if (u, v) == (1, 2) && !self.braess.config.shortcut_enabled {
                        edge.style = TopologyEdgeStyle::Dashed;
                    }
                }
            }
            doc.edges.push(edge);
        }

        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    /// Replaces editor nodes and edges with an imported or synchronized [`GraphTopologyDocument`].
    pub fn load_topology_document(&mut self, doc: &GraphTopologyDocument) -> Result<(), String> {
        doc.validate().map_err(|e| e.to_string())?;
        let (_, positions, edges) = extract_indexed_canvas_graph(doc);
        let max_id = positions.keys().copied().max().map_or(0, |m| m + 1);
        self.node_positions = positions;
        self.edges = edges;
        self.selected_node = None;
        self.dragged_node = None;
        self.hovered_node = None;
        self.hovered_edge = None;
        self.next_node_id = max_id;
        self.braess.sync_with_edges(&self.edges);
        self.bump_revision();
        self.rebuild_graph();
        if self.braess.config.enabled {
            self.recompute_braess();
        }
        Ok(())
    }

    pub(super) fn show_clear_and_layout_controls(&mut self, ui: &mut egui::Ui) {
        if ui
            .button("🔄 Clear Graph")
            .accessible_hover_text("Remove all nodes and edges and start with a fresh graph")
            .clicked()
            && (!self.node_positions.is_empty() || !self.edges.is_empty())
        {
            self.push_undo_snapshot();
            self.graph = Graph::new();
            self.node_indices.clear();
            self.node_positions.clear();
            self.edges.clear();
            self.braess.edge_latencies.clear();
            self.selected_node = None;
            self.next_node_id = 0;
            if self.braess.config.enabled {
                self.recompute_braess();
            }
        }
        ui.separator();
        ui.label("Auto Layout:");
        ui.horizontal(|ui| {
            let bounds = (100.0, 100.0, 700.0, 500.0);
            if ui
                .button("⭕ Circular")
                .accessible_hover_text("Arrange nodes in a circle")
                .clicked()
                && !self.node_positions.is_empty()
            {
                self.push_undo_snapshot();
                let layout = circular_layout(&self.graph, bounds);
                self.apply_layout(&layout);
            }
            if ui
                .button("▦ Grid")
                .accessible_hover_text("Arrange nodes in a grid layout")
                .clicked()
                && !self.node_positions.is_empty()
            {
                self.push_undo_snapshot();
                let layout = grid_layout(&self.graph, bounds);
                self.apply_layout(&layout);
            }
            if ui
                .button("🧲 Force-Directed")
                .accessible_hover_text("Arrange nodes using force-directed physics")
                .clicked()
                && !self.node_positions.is_empty()
            {
                self.push_undo_snapshot();
                let layout = force_directed_layout(&self.graph, bounds, 100);
                self.apply_layout(&layout);
            }
        });
    }
}
