//! Canonical [`GraphTopologyDocument`] builder for [`super::super::GraphEditorTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::GraphEditorTool;
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl GraphEditorTool {
    /// Builds a [`GraphTopologyDocument`] from the editor's current nodes, edges, and Braess routing state.
    pub fn build_topology_document(&self) -> Result<GraphTopologyDocument, String> {
        let braess_on = self.braess.config.enabled;
        let directed = self.directed_edges || braess_on;
        let mut doc = GraphTopologyDocument::new("GraphEditor", directed);
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
}
