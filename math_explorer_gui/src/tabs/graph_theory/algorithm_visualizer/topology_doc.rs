//! Canonical [`GraphTopologyDocument`] builder for [`super::super::AlgorithmVisualizerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::super::{Algorithm, AlgorithmVisualizerTool};
use crate::tabs::graph_theory::io_helpers::color32_to_hex;
use math_explorer::pure_math::graph_theory::interchange::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl AlgorithmVisualizerTool {
    /// Builds a [`GraphTopologyDocument`] annotated with algorithm progress, distances, flows, or MST state.
    pub fn build_topology_document(&self) -> Result<GraphTopologyDocument, String> {
        let mut doc = GraphTopologyDocument::new("AlgorithmVisualizer", self.directed);
        doc.attributes.insert(
            "algorithm".to_string(),
            format!("{:?}", self.selected_algorithm),
        );
        doc.attributes
            .insert("step".to_string(), self.animation_step.to_string());
        if let Some(f) = &self.flow_data {
            doc.attributes
                .insert("max_flow".to_string(), format!("{:.4}", f.max_flow));
            doc.attributes.insert(
                "min_cut_capacity".to_string(),
                format!("{:.4}", f.min_cut_capacity),
            );
        }
        if let Some(m) = &self.mst_data {
            doc.attributes.insert(
                "mst_total_weight".to_string(),
                format!("{:.4}", m.total_weight),
            );
        }

        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        self.populate_topology_nodes(&mut doc, &palette);
        self.populate_topology_edges(&mut doc, &palette);

        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    fn populate_topology_nodes(
        &self,
        doc: &mut GraphTopologyDocument,
        palette: &crate::app::SemanticPalette,
    ) {
        let sink = self.effective_sink_node(self.start_node);
        let settled_slice = &self.visit_order[..self.animation_step.min(self.visit_order.len())];
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();

        for id in sorted_ids {
            let pos = self.node_positions[&id];
            let is_source = self.start_node == Some(id);
            let is_sink = self.selected_algorithm == Algorithm::MaxFlowMinCut && sink == Some(id);
            let label = self.node_topology_label(id, is_source, is_sink);
            let mut node = TopologyNode::new(id.to_string(), label)
                .with_position(f64::from(pos.x), f64::from(pos.y));
            if is_source || is_sink {
                node.shape = TopologyNodeShape::DoubleCircle;
                node.fill_hex = Some(color32_to_hex(palette.accent));
            } else if settled_slice.contains(&id) {
                node.fill_hex = Some(color32_to_hex(palette.success));
            } else {
                node.fill_hex = Some(color32_to_hex(palette.series_color(0)));
            }
            if let Some(&d) = self.distances.get(&id).filter(|d| d.is_finite()) {
                node.attributes
                    .insert("distance".to_string(), format!("{d:.4}"));
            }
            doc.nodes.push(node);
        }
    }

    fn node_topology_label(&self, id: usize, is_source: bool, is_sink: bool) -> String {
        match self.selected_algorithm {
            Algorithm::Dijkstra => match self.distances.get(&id) {
                Some(&d) if d.is_finite() => format!("{id} ({d:.1})"),
                _ => format!("{id} (inf)"),
            },
            Algorithm::MaxFlowMinCut if is_source => format!("{id}:s"),
            Algorithm::MaxFlowMinCut if is_sink => format!("{id}:t"),
            _ => id.to_string(),
        }
    }

    fn populate_topology_edges(
        &self,
        doc: &mut GraphTopologyDocument,
        palette: &crate::app::SemanticPalette,
    ) {
        for (idx, &(u, v, weight)) in self.edges.iter().enumerate() {
            if !self.node_positions.contains_key(&u) || !self.node_positions.contains_key(&v) {
                continue;
            }
            let mut edge = TopologyEdge::new(u.to_string(), v.to_string(), weight)
                .with_label(format!("{weight:.1}"));
            if self.is_edge_highlighted_at_step(idx) {
                edge.style = TopologyEdgeStyle::Thick;
                edge.color_hex = Some(color32_to_hex(palette.warning));
                edge.attributes
                    .insert("highlighted".to_string(), "true".to_string());
            }
            doc.edges.push(edge);
        }
    }
}
