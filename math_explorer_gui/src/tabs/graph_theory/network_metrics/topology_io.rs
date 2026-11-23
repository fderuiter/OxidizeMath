//! `GraphTopologyDocument` import, reset, and export adapter for [`super::NetworkMetricsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MetricType, NetworkMetricsTool};
use crate::tabs::graph_theory::topology_sync::extract_indexed_canvas_graph;
use math_explorer::pure_math::graph_theory::graph::Graph;
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyNode, TopologyNodeShape,
};
use math_explorer::pure_math::graph_theory::parameters::network_metrics::{
    closeness_centrality, clustering_coefficients, degree_centrality,
};

impl NetworkMetricsTool {
    /// Builds a [`GraphTopologyDocument`] annotated with centrality metrics, `degeneracy`, and `vertex_cover`.
    pub fn build_topology_document(&self) -> Result<GraphTopologyDocument, String> {
        let mut doc = GraphTopologyDocument::new("NetworkMetrics", false);
        doc.attributes.insert(
            "selected_metric".to_string(),
            self.selected_metric.label().to_string(),
        );
        doc.attributes.insert(
            "degeneracy".to_string(),
            self.global_degeneracy.unwrap_or(0).to_string(),
        );
        doc.attributes.insert(
            "vertex_cover_size".to_string(),
            self.vertex_cover_ids.len().to_string(),
        );
        if let Some(diam) = self.global_diameter {
            doc.attributes
                .insert("diameter".to_string(), diam.to_string());
        }
        if let Some(clust) = self.global_clustering {
            doc.attributes
                .insert("average_clustering".to_string(), format!("{clust:.4}"));
        }
        self.append_metric_nodes_and_edges(&mut doc);
        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    fn append_metric_nodes_and_edges(&self, doc: &mut GraphTopologyDocument) {
        let deg_map = degree_centrality(&self.graph);
        let close_map = closeness_centrality(&self.graph);
        let clust_map = clustering_coefficients(&self.graph);
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let pos = self.node_positions[&id];
            let label = self
                .node_labels
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string());
            let in_vc = self.vertex_cover_ids.contains(&id);
            let mut node = TopologyNode::new(id.to_string(), label)
                .with_position(f64::from(pos.x), f64::from(pos.y));
            if in_vc {
                node.shape = TopologyNodeShape::DoubleCircle;
                node.stroke_hex = Some("#F59E0B".to_string());
                node.fill_hex = Some("#B45309".to_string());
            } else {
                node.fill_hex = Some("#1D4ED8".to_string());
            }
            node.attributes
                .insert("in_vertex_cover".to_string(), in_vc.to_string());
            if let Some(&idx) = self.node_indices.get(&id) {
                for (key, map) in [
                    ("degree_centrality", &deg_map),
                    ("closeness_centrality", &close_map),
                    ("clustering_coefficient", &clust_map),
                ] {
                    if let Some(&val) = map.get(&idx) {
                        node.attributes.insert(key.to_string(), format!("{val:.4}"));
                    }
                }
            }
            doc.nodes.push(node);
        }
        for &(u, v, w) in &self.edges {
            if self.node_positions.contains_key(&u) && self.node_positions.contains_key(&v) {
                doc.edges
                    .push(TopologyEdge::new(u.to_string(), v.to_string(), w));
            }
        }
    }

    /// Replaces the active network with an imported or synchronized [`GraphTopologyDocument`].
    pub fn load_topology_document(&mut self, doc: &GraphTopologyDocument) -> Result<(), String> {
        doc.validate().map_err(|e| e.to_string())?;
        if doc.nodes.is_empty() {
            return Ok(());
        }
        let (labels, positions, edges) = extract_indexed_canvas_graph(doc);
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.node_labels = labels;
        self.edges.clear();

        for (&id, &pos) in &positions {
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
            self.node_positions.insert(id, pos);
        }
        let mut seen_undirected = std::collections::HashSet::new();
        for (u, v, w) in edges {
            let pair = if u <= v { (u, v) } else { (v, u) };
            if !seen_undirected.insert(pair) {
                continue;
            }
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u), self.node_indices.get(&v))
            {
                self.edges.push((u, v, w));
                self.graph.add_edge(u_idx, v_idx, w);
            }
        }
        self.compute_metrics();
        Ok(())
    }

    /// Restores the default 7-node wheel graph.
    pub fn restore_sample_wheel(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.node_labels.clear();
        self.edges.clear();
        self.build_sample_graph();
    }
}

impl MetricType {
    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::DegreeCentrality => "DegreeCentrality",
            Self::ClosenessCentrality => "ClosenessCentrality",
            Self::ClusteringCoefficient => "ClusteringCoefficient",
            Self::VertexCover => "VertexCover",
        }
    }
}
