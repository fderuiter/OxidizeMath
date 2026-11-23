//! `GraphTopologyDocument` import, reset, and export adapter for [`super::AlgorithmVisualizerTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Algorithm, AlgorithmVisualizerTool};
use crate::tabs::graph_theory::topology_sync::extract_indexed_canvas_graph;
use math_explorer::pure_math::graph_theory::graph::Graph;
use math_explorer::pure_math::graph_theory::graph_io::{
    GraphTopologyDocument, TopologyEdge, TopologyEdgeStyle, TopologyNode, TopologyNodeShape,
};

impl AlgorithmVisualizerTool {
    /// Builds a [`GraphTopologyDocument`] annotated with traversal order and shortest-path distances.
    pub fn build_topology_document(&self) -> Result<GraphTopologyDocument, String> {
        let algo_name = match self.selected_algorithm {
            Algorithm::Dijkstra => "Dijkstra",
            Algorithm::Bfs => "BFS",
            Algorithm::Dfs => "DFS",
        };
        let mut doc = GraphTopologyDocument::new("AlgorithmVisualizer", false);
        doc.attributes
            .insert("algorithm".to_string(), algo_name.to_string());
        if let Some(start) = self.start_node {
            doc.attributes
                .insert("start_node".to_string(), start.to_string());
        }
        doc.attributes.insert(
            "animation_step".to_string(),
            self.animation_step.to_string(),
        );
        self.append_traversal_nodes(&mut doc);
        self.append_traversal_edges(&mut doc);
        doc.validate().map_err(|e| e.to_string())?;
        Ok(doc)
    }

    fn append_traversal_nodes(&self, doc: &mut GraphTopologyDocument) {
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let pos = self.node_positions[&id];
            let label = self
                .node_labels
                .get(&id)
                .cloned()
                .unwrap_or_else(|| id.to_string());
            let mut node = TopologyNode::new(id.to_string(), label)
                .with_position(f64::from(pos.x), f64::from(pos.y));
            let visit_rank = self.visit_order.iter().position(|&v| v == id);
            let is_visited = visit_rank.is_some_and(|r| r < self.animation_step);
            if self.start_node == Some(id) {
                node.shape = TopologyNodeShape::DoubleCircle;
                node.fill_hex = Some("#D97706".to_string());
            } else if is_visited {
                node.fill_hex = Some("#16A34A".to_string());
            } else {
                node.fill_hex = Some("#1E40AF".to_string());
            }
            node.attributes
                .insert("visited".to_string(), is_visited.to_string());
            if let Some(rank) = visit_rank {
                node.attributes
                    .insert("visit_rank".to_string(), rank.to_string());
            }
            if let Some(&dist) = self.distances.get(&id) {
                if dist.is_finite() {
                    node.attributes
                        .insert("distance".to_string(), format!("{dist:.4}"));
                }
            }
            doc.nodes.push(node);
        }
    }

    fn append_traversal_edges(&self, doc: &mut GraphTopologyDocument) {
        for &(u, v, w) in &self.edges {
            if !self.node_positions.contains_key(&u) || !self.node_positions.contains_key(&v) {
                continue;
            }
            let u_vis = self
                .visit_order
                .iter()
                .take(self.animation_step)
                .any(|&x| x == u);
            let v_vis = self
                .visit_order
                .iter()
                .take(self.animation_step)
                .any(|&x| x == v);
            let mut edge = TopologyEdge::new(u.to_string(), v.to_string(), w);
            if u_vis && v_vis {
                edge.style = TopologyEdgeStyle::Thick;
                edge.color_hex = Some("#16A34A".to_string());
            }
            doc.edges.push(edge);
        }
    }

    /// Replaces the active graph with an imported or synchronized [`GraphTopologyDocument`].
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
        for (u, v, w) in edges {
            let weight = w.max(0.0);
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u), self.node_indices.get(&v))
            {
                self.edges.push((u, v, weight));
                self.graph.add_edge(u_idx, v_idx, weight);
            }
        }
        let first_id = positions.keys().copied().next();
        if self
            .start_node
            .is_none_or(|s| !self.node_indices.contains_key(&s))
        {
            self.start_node = first_id;
        }
        if let Some(fid) = self.start_node {
            self.focused_node = fid;
        }
        self.run_algorithm();
        self.animation_step = self.visit_order.len();
        Ok(())
    }

    /// Restores the default 16-node 4x4 weighted grid topology.
    pub fn restore_sample_grid(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.node_labels.clear();
        self.edges.clear();
        self.build_sample_graph();
    }
}

impl scientific_metadata::theory::TheoryDescribable for AlgorithmVisualizerTool {
    fn theory_description(&self) -> String {
        let algo_str = match self.selected_algorithm {
            Algorithm::Dijkstra => "Dijkstra shortest path algorithm via priority relaxation $d(v) = \\min(d(v), d(u) + w(u, v))$",
            Algorithm::Bfs => "Breadth-First Search level-order queue traversal computing unweighted shortest path distances",
            Algorithm::Dfs => "Depth-First Search recursive stack traversal exploring graph connectivity and cycles",
        };
        format!(
            "Graph traversal and pathfinding analysis executing {} across $|V| = {}$ nodes and $|E| = {}$ edges.",
            algo_str,
            self.node_positions.len(),
            self.edges.len()
        )
    }

    fn phonetic_description(&self) -> String {
        "Graph search and shortest path algorithm theory".into()
    }

    fn theory_citation(&self) -> String {
        "A Note on Two Problems in Connexion with Graphs (Dijkstra, 1959); Introduction to Algorithms (Cormen et al., 2022)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Dijkstra".into(),
            "Greedy shortest path in non-negatively weighted graphs".into(),
        );
        map.insert(
            "BFS".into(),
            "Queue-based breadth traversal finding minimum hop counts".into(),
        );
        map.insert(
            "DFS".into(),
            "Stack-based depth exploration identifying connected components".into(),
        );
        map
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
