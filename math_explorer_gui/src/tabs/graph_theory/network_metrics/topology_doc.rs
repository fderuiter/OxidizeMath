//! Canonical [`GraphTopologyDocument`] builder for [`super::super::NetworkMetricsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::NetworkMetricsTool;
use crate::tabs::graph_theory::io_helpers::color32_to_hex;
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyNode, TopologyNodeShape,
};

impl NetworkMetricsTool {
    /// Builds a [`GraphTopologyDocument`] annotated with centrality scores, degeneracy, vertex cover, and treewidth.
    pub fn build_topology_document(&self) -> Result<GraphTopologyDocument, String> {
        let mut doc = GraphTopologyDocument::new("NetworkMetrics", self.directed);
        doc.attributes.insert(
            "selected_metric".to_string(),
            format!("{:?}", self.selected_metric),
        );
        if let Some(d) = self.global_diameter {
            doc.attributes.insert("diameter".to_string(), d.to_string());
        }
        if let Some(c) = self.global_clustering {
            doc.attributes
                .insert("avg_clustering".to_string(), format!("{c:.6}"));
        }
        doc.attributes
            .insert("degeneracy".to_string(), self.global_degeneracy.to_string());
        doc.attributes
            .insert("treewidth".to_string(), self.global_treewidth.to_string());
        doc.attributes.insert(
            "vertex_cover_size".to_string(),
            self.vertex_cover_nodes.len().to_string(),
        );

        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let (min_val, range) = self.metric_min_and_range();
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();

        for id in sorted_ids {
            let pos = self.node_positions[&id];
            let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
            let fill = self.node_metric_color(val, min_val, range, &palette);
            let in_vc = self.vertex_cover_nodes.contains(&id);
            let mut node = TopologyNode::new(id.to_string(), format!("{id}:{val:.2}"))
                .with_position(f64::from(pos.x), f64::from(pos.y))
                .with_color(color32_to_hex(fill))
                .with_attribute("metric", format!("{val:.6}"))
                .with_attribute("in_vertex_cover", in_vc.to_string());
            if in_vc && self.highlight_vertex_cover {
                node.shape = TopologyNodeShape::DoubleCircle;
            }
            doc.nodes.push(node);
        }

        for &(u, v, weight) in &self.edges {
            if self.node_positions.contains_key(&u) && self.node_positions.contains_key(&v) {
                doc.edges
                    .push(TopologyEdge::new(u.to_string(), v.to_string(), weight));
            }
        }

        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }
}
