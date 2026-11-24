//! Offscreen plot exports, styled `.dot`/`.graphml`/`.csv` serialization, and topology handoff for `AlgorithmVisualizerTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{Algorithm, AlgorithmVisualizerState, AlgorithmVisualizerTool};
use crate::accessibility::AccessibleHoverText;
use crate::tabs::graph_theory::{
    io_helpers::{color32_to_hex, render_handoff_buttons},
    plot_export::{
        build_network_datasets, build_network_plot_primitives, EdgePlotSpec, NodePlotGroup,
    },
    shared_topology::{
        take_pending_handoff_for, CompactTopologyState, SharedGraphTopology, TopologyHandoffTarget,
    },
};
use eframe::egui::{self, Pos2};
use egui_plot::ExportPrimitive;
use math_explorer::pure_math::graph_theory::interchange::{
    to_edge_csv, to_graphml, to_styled_dot, DotEdgeStyle, DotNodeStyle, DotStyleOverlay,
};
use std::collections::BTreeMap;

/// Serializes a [`SharedGraphTopology`] into an `AlgorithmVisualizerState` JSON payload.
#[must_use]
pub fn visualizer_state_json_from_shared(topo: &SharedGraphTopology) -> Option<String> {
    let algo = if topo.directed && topo.sink_node.is_some() {
        Algorithm::MaxFlowMinCut
    } else {
        Algorithm::Dijkstra
    };
    let state = AlgorithmVisualizerState {
        selected_algorithm: algo,
        start_node: topo
            .source_node
            .or_else(|| topo.nodes.keys().next().copied()),
        sink_node: topo.sink_node,
        playback_speed: 2.0,
        custom_topology: Some(topo.to_compact_state()),
    };
    serde_json::to_string(&state).ok()
}

impl AlgorithmVisualizerTool {
    /// Exports the active graph and terminal selections as a [`SharedGraphTopology`].
    #[must_use]
    pub fn export_shared_topology(&self) -> SharedGraphTopology {
        let mut nodes = BTreeMap::new();
        for (&id, &pos) in &self.node_positions {
            nodes.insert(id, (pos.x, pos.y));
        }
        SharedGraphTopology {
            nodes,
            edges: self.edges.clone(),
            directed: self.directed,
            source_node: self.start_node,
            sink_node: self.effective_sink_node(self.start_node),
            edge_latencies: Vec::new(),
            revision: self.revision.max(1),
        }
    }

    /// Replaces the active graph with a custom [`SharedGraphTopology`] and reruns the active algorithm.
    pub fn apply_shared_topology(&mut self, topo: &SharedGraphTopology) {
        if topo.is_empty() {
            return;
        }
        self.load_compact_topology(&topo.to_compact_state());
        if let Some(s) = topo
            .source_node
            .filter(|id| self.node_positions.contains_key(id))
        {
            self.start_node = Some(s);
            self.focused_node = s;
        }
        if let Some(t) = topo
            .sink_node
            .filter(|id| self.node_positions.contains_key(id) && Some(*id) != self.start_node)
        {
            self.sink_node = Some(t);
        } else {
            self.sink_node = self.effective_sink_node(self.start_node);
        }
        self.bump_revision();
        self.run_algorithm();
        self.animation_step = self.total_steps();
    }

    pub(super) fn load_compact_topology(&mut self, compact: &CompactTopologyState) {
        self.graph = math_explorer::pure_math::graph_theory::graph::Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();
        self.directed = compact.directed;
        self.is_custom_topology = true;

        for (&id, &(x, y)) in &compact.nodes {
            let pos = Pos2::new(x, y);
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
            self.node_positions.insert(id, pos);
        }
        for &(u, v, w) in &compact.edges {
            self.edges.push((u, v, w));
            if let (Some(&ui), Some(&vi)) = (self.node_indices.get(&u), self.node_indices.get(&v)) {
                self.graph.add_edge(ui, vi, w);
            }
        }
        let first_id = compact.nodes.keys().next().copied();
        if !self
            .start_node
            .is_some_and(|id| self.node_positions.contains_key(&id))
        {
            self.start_node = first_id;
        }
        if let Some(s) = self.start_node {
            self.focused_node = s;
        }
        self.sink_node = self
            .sink_node
            .filter(|id| self.node_positions.contains_key(id) && Some(*id) != self.start_node)
            .or_else(|| self.effective_sink_node(self.start_node));
    }

    /// Resets the visualizer back to its canonical 4x4 undirected grid sample graph.
    pub fn reset_to_sample_grid(&mut self) {
        self.build_sample_graph();
        self.bump_revision();
    }

    /// Emits styled Graphviz `.dot` reflecting the active predecessor tree, `s-t` min-cut, or MST overlay.
    pub fn to_styled_dot_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let mut overlay = DotStyleOverlay {
            graph_label: Some(self.dot_overlay_caption()),
            ..Default::default()
        };
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let (fill, lbl) = self.classify_node_visual(id, &palette);
            overlay.node_styles.insert(
                id,
                DotNodeStyle {
                    color: Some(color32_to_hex(palette.node_stroke)),
                    fillcolor: Some(color32_to_hex(fill)),
                    penwidth: Some(if self.start_node == Some(id) {
                        2.5
                    } else {
                        1.5
                    }),
                    xlabel: Some(lbl),
                },
            );
        }
        for idx in 0..self.edges.len() {
            let (col, width, lbl) = self.classify_edge_visual(idx, &palette);
            overlay.edge_styles.insert(
                idx,
                DotEdgeStyle {
                    color: Some(color32_to_hex(col)),
                    penwidth: Some(f64::from(width)),
                    xlabel: Some(lbl),
                },
            );
        }
        to_styled_dot(&interchange, &overlay).map_err(|e| e.to_string())
    }

    fn dot_overlay_caption(&self) -> String {
        match self.selected_algorithm {
            Algorithm::Dijkstra => "Dijkstra Predecessor Tree".to_string(),
            Algorithm::Bfs => "BFS Discovery Tree".to_string(),
            Algorithm::Dfs => "DFS Discovery Tree".to_string(),
            Algorithm::MaxFlowMinCut => self.flow_data.as_ref().map_or_else(
                || "Edmonds-Karp Max-Flow Min-Cut".to_string(),
                |f| {
                    format!(
                        "Edmonds-Karp Max-Flow = {:.2} (Min-Cut = {:.2})",
                        f.max_flow, f.min_cut_capacity
                    )
                },
            ),
            Algorithm::MinimumSpanningTree => self.mst_data.as_ref().map_or_else(
                || "Kruskal Minimum Spanning Tree".to_string(),
                |m| format!("Kruskal MST Weight = {:.2}", m.total_weight),
            ),
        }
    }

    /// Serializes the active topology as `.graphml`.
    pub fn to_graphml_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        to_graphml(&interchange).map_err(|e| e.to_string())
    }

    /// Serializes the active topology as weighted `.csv`.
    pub fn to_csv_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        to_edge_csv(&interchange).map_err(|e| e.to_string())
    }

    pub(super) fn render_topology_io_controls(&mut self, ui: &mut egui::Ui) {
        if let Some(pending) =
            take_pending_handoff_for(ui.ctx(), TopologyHandoffTarget::AlgorithmVisualizer)
        {
            self.apply_shared_topology(&pending);
        }
        if let Some(imported) = self.file_io.poll(ui.ctx()) {
            self.apply_shared_topology(&imported);
        }
        ui.label("Topology & Takeaway Exports:");
        let snap = self.export_shared_topology();
        if let Some(shared) = render_handoff_buttons(
            ui,
            TopologyHandoffTarget::AlgorithmVisualizer,
            !self.node_positions.is_empty(),
            || snap,
        ) {
            self.apply_shared_topology(&shared);
        }
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("📂 Import")
                .accessible_hover_text("Import a .dot, .graphml, .csv, or .json network")
                .clicked()
            {
                self.file_io.trigger_import_dialog();
            }
            if self.is_custom_topology
                && ui
                    .button("▦ 4x4 Grid")
                    .accessible_hover_text("Restore default 4x4 sample grid")
                    .clicked()
            {
                self.reset_to_sample_grid();
            }
        });
        self.render_visualizer_download_buttons(ui);
        self.file_io.render_status_banner(ui);
    }

    fn render_visualizer_download_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("⬇ .dot")
                .accessible_hover_text(
                    "Download styled Graphviz .dot with active algorithm overlay",
                )
                .clicked()
            {
                self.file_io
                    .trigger_save_dot("algorithm_overlay.dot", self.to_styled_dot_string());
            }
            if ui
                .button("⬇ .graphml")
                .accessible_hover_text("Download GraphML XML topology")
                .clicked()
            {
                self.file_io
                    .trigger_save_graphml("algorithm_topology.graphml", self.to_graphml_string());
            }
            if ui
                .button("⬇ .csv")
                .accessible_hover_text("Download weighted CSV edge list")
                .clicked()
            {
                self.file_io
                    .trigger_save_csv("algorithm_edges.csv", self.to_csv_string());
            }
        });
    }

    pub(super) fn build_visualizer_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let mut positions_btree = BTreeMap::new();
        for (&id, &pos) in &self.node_positions {
            positions_btree.insert(id, pos);
        }
        let mut edge_specs = Vec::with_capacity(self.edges.len());
        for (idx, &(u, v, _)) in self.edges.iter().enumerate() {
            let (color, width, _) = self.classify_edge_visual(idx, &palette);
            edge_specs.push(EdgePlotSpec {
                name: format!("Edge {u}-{v}"),
                u,
                v,
                color,
                width,
                directed: self.directed,
            });
        }
        let mut groups_by_color: BTreeMap<[u8; 4], Vec<Pos2>> = BTreeMap::new();
        for (&id, &pos) in &positions_btree {
            let (col, _) = self.classify_node_visual(id, &palette);
            groups_by_color
                .entry([col.r(), col.g(), col.b(), col.a()])
                .or_default()
                .push(pos);
        }
        let groups: Vec<NodePlotGroup> = groups_by_color
            .into_iter()
            .enumerate()
            .map(|(gi, (rgba, positions))| NodePlotGroup {
                name: format!("Vertex Group {gi}"),
                positions,
                color: egui::Color32::from_rgba_unmultiplied(rgba[0], rgba[1], rgba[2], rgba[3]),
                radius: 6.5,
            })
            .collect();
        build_network_plot_primitives(&positions_btree, &edge_specs, &groups)
    }

    pub(super) fn build_visualizer_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut positions_btree = BTreeMap::new();
        for (&id, &pos) in &self.node_positions {
            positions_btree.insert(id, pos);
        }
        build_network_datasets(&positions_btree, &self.edges)
    }
}

impl scientific_metadata::theory::TheoryDescribable for AlgorithmVisualizerTool {
    fn theory_description(&self) -> String {
        let algo_str = match self.selected_algorithm {
            Algorithm::Dijkstra => {
                "Dijkstra shortest-path predecessor tree $d(v) = \\min(d(v), d(u) + w(u, v))$"
            }
            Algorithm::Bfs => "Breadth-First Search level-order discovery tree",
            Algorithm::Dfs => "Depth-First Search stack discovery tree",
            Algorithm::MaxFlowMinCut => {
                "Edmonds-Karp shortest augmenting-path $s$-$t$ maximum flow and minimum cut"
            }
            Algorithm::MinimumSpanningTree => "Kruskal disjoint-set minimum spanning tree",
        };
        format!(
            "Graph optimization and traversal executing {} across $|V| = {}$ nodes and $|E| = {}$ edges.",
            algo_str,
            self.node_positions.len(),
            self.edges.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Graph search, network flow, and spanning tree theory".into()
    }
    fn theory_citation(&self) -> String {
        "Dijkstra (1959); Edmonds & Karp (1972); Kruskal (1956); Cormen et al. (2022)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Dijkstra".into(),
            "Greedy shortest-path predecessor tree in non-negative graphs".into(),
        );
        map.insert(
            "BFS".into(),
            "Queue-based breadth traversal finding minimum hop counts".into(),
        );
        map.insert(
            "DFS".into(),
            "Stack-based depth exploration identifying connected components".into(),
        );
        map.insert(
            "MaxFlowMinCut".into(),
            "Edmonds-Karp augmenting paths and s-t minimum-cut partition".into(),
        );
        map.insert(
            "MinimumSpanningTree".into(),
            "Kruskal union-find minimum-weight spanning forest".into(),
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

// [cite:graph_parameters_rust]
