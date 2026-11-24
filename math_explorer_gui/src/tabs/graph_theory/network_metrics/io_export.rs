//! Offscreen plot exports, styled `.dot`/`.graphml`/`.csv` invariant downloads, and handoff for `NetworkMetricsTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{MetricType, NetworkMetricsState, NetworkMetricsTool};
use crate::accessibility::AccessibleHoverText;
use crate::app::{SharedGraphTopology as BusGraphTopology, SharedSimulationBus, TopologySource};
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
use math_explorer::pure_math::graph_theory::{
    interchange::{to_graphml, to_styled_dot, DotNodeStyle, DotStyleOverlay},
    parameters::network_metrics::{
        closeness_centrality, clustering_coefficients, degree_centrality,
    },
};
use std::collections::BTreeMap;

/// Serializes a [`SharedGraphTopology`] into a `NetworkMetricsState` JSON payload.
#[must_use]
pub fn metrics_state_json_from_shared(topo: &SharedGraphTopology) -> Option<String> {
    let state = NetworkMetricsState {
        selected_metric: MetricType::DegreeCentrality,
        highlight_vertex_cover: true,
        custom_topology: Some(topo.to_compact_state()),
        topology_source: TopologySource::LocalPreset,
    };
    serde_json::to_string(&state).ok()
}

impl NetworkMetricsTool {
    /// Synchronizes this tool's graph with `SharedSimulationBus` or restores the sample wheel graph.
    pub fn sync_shared_topology(&mut self, ctx: &egui::Context) {
        match SharedSimulationBus::active_graph_for(ctx, self.topology_source) {
            Some(snap) => {
                if self.synced_bus_revision != Some(snap.revision) {
                    self.apply_shared_graph(&snap);
                    self.synced_bus_revision = Some(snap.revision);
                }
            }
            None => {
                if self.synced_bus_revision.take().is_some() {
                    self.build_sample_graph();
                    self.bump_revision();
                }
            }
        }
    }

    fn apply_shared_graph(&mut self, snap: &BusGraphTopology) {
        if snap.node_count() == 0 {
            return;
        }
        self.graph = math_explorer::pure_math::graph_theory::graph::Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();
        self.directed = snap.directed;
        self.is_custom_topology = false;

        let mapped = snap.map_positions_to_bounds(70.0, 70.0, 380.0, 340.0);
        for (idx_in_snap, &orig_id) in snap.node_ids.iter().enumerate() {
            let [x, y] = mapped[idx_in_snap];
            let pos = Pos2::new(x, y);
            let g_idx = self.graph.add_node(pos);
            self.node_indices.insert(orig_id, g_idx);
            self.node_positions.insert(orig_id, pos);
        }
        for &(u, v, w) in &snap.undirected_edges {
            let u_id = snap.node_ids[u];
            let v_id = snap.node_ids[v];
            if let (Some(&u_idx), Some(&v_idx)) =
                (self.node_indices.get(&u_id), self.node_indices.get(&v_id))
            {
                self.edges.push((u_id, v_id, w));
                self.graph.add_edge(u_idx, v_idx, w);
            }
        }
        self.bump_revision();
        self.compute_metrics();
    }

    /// Returns the currently synced shared bus revision, if active.
    #[cfg(test)]
    #[must_use]
    pub fn synced_bus_revision(&self) -> Option<u64> {
        self.synced_bus_revision
    }

    /// Returns the number of nodes currently analyzed.
    #[cfg(test)]
    #[must_use]
    pub fn active_node_count(&self) -> usize {
        self.node_positions.len()
    }
    /// Exports the active network topology as a [`SharedGraphTopology`].
    #[must_use]
    pub fn export_shared_topology(&self) -> SharedGraphTopology {
        let mut nodes = BTreeMap::new();
        for (&id, &pos) in &self.node_positions {
            nodes.insert(id, (pos.x, pos.y));
        }
        let source_node = nodes.keys().next().copied();
        let sink_node = nodes
            .keys()
            .next_back()
            .copied()
            .filter(|&t| Some(t) != source_node);
        SharedGraphTopology {
            nodes,
            edges: self.edges.clone(),
            directed: self.directed,
            source_node,
            sink_node,
            edge_latencies: Vec::new(),
            revision: self.revision.max(1),
        }
    }

    /// Replaces the active graph with a custom [`SharedGraphTopology`] and recomputes all invariants.
    pub fn apply_shared_topology(&mut self, topo: &SharedGraphTopology) {
        if topo.is_empty() {
            return;
        }
        self.synced_bus_revision = None;
        self.topology_source = TopologySource::LocalPreset;
        self.load_compact_topology(&topo.to_compact_state());
        self.bump_revision();
        self.compute_metrics();
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
    }

    /// Restores the default 7-node wheel graph (`W_7`) and recomputes invariants.
    pub fn reset_to_sample_wheel(&mut self) {
        self.synced_bus_revision = None;
        self.build_sample_graph();
        self.bump_revision();
    }

    /// Emits a color-styled Graphviz `.dot` figure with heat-mapped centrality and modulator styling.
    pub fn to_styled_dot_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let (min_val, range) = self.metric_min_and_range();
        let mut overlay = DotStyleOverlay {
            graph_label: Some(format!(
                "Diameter={} | Degeneracy={} | VertexCover={} | Treewidth={}",
                self.global_diameter.unwrap_or(0),
                self.global_degeneracy,
                self.vertex_cover_nodes.len(),
                self.global_treewidth
            )),
            ..Default::default()
        };
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
            let fill = self.node_metric_color(val, min_val, range, &palette);
            let in_vc = self.vertex_cover_nodes.contains(&id);
            let stroke = if in_vc && self.highlight_vertex_cover {
                palette.warning
            } else {
                palette.node_stroke
            };
            let xl = if in_vc {
                format!("{val:.2} [VC]")
            } else {
                format!("{val:.2}")
            };
            overlay.node_styles.insert(
                id,
                DotNodeStyle {
                    color: Some(color32_to_hex(stroke)),
                    fillcolor: Some(color32_to_hex(fill)),
                    penwidth: Some(if in_vc && self.highlight_vertex_cover {
                        3.2
                    } else {
                        1.4
                    }),
                    xlabel: Some(xl),
                },
            );
        }
        to_styled_dot(&interchange, &overlay).map_err(|e| e.to_string())
    }

    /// Emits `.graphml` XML for the active topology.
    pub fn to_graphml_string(&self) -> Result<String, String> {
        let interchange = self.export_shared_topology().to_interchange_graph()?;
        to_graphml(&interchange).map_err(|e| e.to_string())
    }

    /// Emits a `.csv` invariant summary and per-vertex centrality/modulator table.
    pub fn to_invariants_csv_string(&self) -> Result<String, String> {
        let deg_map = degree_centrality(&self.graph);
        let close_map = closeness_centrality(&self.graph);
        let clust_map = clustering_coefficients(&self.graph);
        let mut out = format!(
            "# diameter={},avg_clustering={:.6},degeneracy={},vertex_cover_size={},treewidth={}\n",
            self.global_diameter.unwrap_or(0),
            self.global_clustering.unwrap_or(0.0),
            self.global_degeneracy,
            self.vertex_cover_nodes.len(),
            self.global_treewidth
        );
        out.push_str(
            "node_id,degree_centrality,closeness_centrality,clustering_coefficient,in_vertex_cover\n",
        );
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        for id in sorted_ids {
            let Some(&idx) = self.node_indices.get(&id) else {
                continue;
            };
            let dc = deg_map.get(&idx).copied().unwrap_or(0.0);
            let cc = close_map.get(&idx).copied().unwrap_or(0.0);
            let lc = clust_map.get(&idx).copied().unwrap_or(0.0);
            let vc = usize::from(self.vertex_cover_nodes.contains(&id));
            out.push_str(&format!("{id},{dc:.6},{cc:.6},{lc:.6},{vc}\n"));
        }
        Ok(out)
    }

    pub(super) fn render_topology_io_controls(&mut self, ui: &mut egui::Ui) {
        if let Some(pending) =
            take_pending_handoff_for(ui.ctx(), TopologyHandoffTarget::NetworkMetrics)
        {
            self.apply_shared_topology(&pending);
        }
        if let Some(imported) = self.file_io.poll(ui.ctx()) {
            self.apply_shared_topology(&imported);
        }
        ui.label("Topology & Artifact Downloads:");
        let snap = self.export_shared_topology();
        if let Some(shared) = render_handoff_buttons(
            ui,
            TopologyHandoffTarget::NetworkMetrics,
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
                    .button("☸ Wheel W_7")
                    .accessible_hover_text("Restore default 7-node wheel graph")
                    .clicked()
            {
                self.reset_to_sample_wheel();
            }
        });
        self.render_metrics_download_buttons(ui);
        self.file_io.render_status_banner(ui);
    }

    fn render_metrics_download_buttons(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("⬇ .dot")
                .accessible_hover_text("Download heat-mapped Graphviz .dot figure")
                .clicked()
            {
                self.file_io
                    .trigger_save_dot("network_metrics.dot", self.to_styled_dot_string());
            }
            if ui
                .button("⬇ .graphml")
                .accessible_hover_text("Download GraphML XML topology")
                .clicked()
            {
                self.file_io
                    .trigger_save_graphml("network_metrics.graphml", self.to_graphml_string());
            }
            if ui
                .button("⬇ .csv")
                .accessible_hover_text("Download per-node centrality and structural invariants CSV")
                .clicked()
            {
                self.file_io
                    .trigger_save_csv("network_invariants.csv", self.to_invariants_csv_string());
            }
        });
    }

    pub(super) fn build_metrics_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::SemanticPalette::for_theme(crate::app::ThemeMode::Dark);
        let (min_val, range) = self.metric_min_and_range();
        let mut positions_btree = BTreeMap::new();
        for (&id, &pos) in &self.node_positions {
            positions_btree.insert(id, pos);
        }
        let edge_specs: Vec<EdgePlotSpec> = self
            .edges
            .iter()
            .map(|&(u, v, _)| EdgePlotSpec {
                name: format!("Edge {u}-{v}"),
                u,
                v,
                color: palette.graph_edge,
                width: 2.0,
                directed: self.directed,
            })
            .collect();
        let mut groups = Vec::new();
        if self.highlight_vertex_cover && !self.vertex_cover_nodes.is_empty() {
            let mod_positions: Vec<Pos2> = self
                .vertex_cover_nodes
                .iter()
                .filter_map(|id| positions_btree.get(id).copied())
                .collect();
            groups.push(NodePlotGroup {
                name: "Vertex Cover Modulator".to_string(),
                positions: mod_positions,
                color: palette.warning,
                radius: 9.5,
            });
        }
        for (&id, &pos) in &positions_btree {
            let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
            let col = self.node_metric_color(val, min_val, range, &palette);
            groups.push(NodePlotGroup {
                name: format!("Node {id} ({val:.2})"),
                positions: vec![pos],
                color: col,
                radius: 6.5,
            });
        }
        build_network_plot_primitives(&positions_btree, &edge_specs, &groups)
    }

    pub(super) fn build_metrics_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut positions_btree = BTreeMap::new();
        for (&id, &pos) in &self.node_positions {
            positions_btree.insert(id, pos);
        }
        let mut ds = build_network_datasets(&positions_btree, &self.edges);
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();
        let profile: Vec<[f64; 2]> = sorted_ids
            .into_iter()
            .map(|id| {
                let v = self.metric_values.get(&id).copied().unwrap_or(0.0);
                [id as f64, v]
            })
            .collect();
        if !profile.is_empty() {
            ds.push(("Node Metric Profile".to_string(), profile));
        }
        ds
    }
}

impl scientific_metadata::theory::TheoryDescribable for NetworkMetricsTool {
    fn theory_description(&self) -> String {
        let metric_str = match self.selected_metric {
            MetricType::DegreeCentrality => "degree centrality $C_D(v) = \\frac{\\deg(v)}{|V|-1}$",
            MetricType::ClosenessCentrality => {
                "closeness centrality $C_C(v) = \\frac{|V|-1}{\\sum_{u \\neq v} d(v, u)}$"
            }
            MetricType::ClusteringCoefficient => {
                "Watts-Strogatz local clustering coefficient $C_v = \\frac{2 e_v}{k_v(k_v-1)}$"
            }
        };
        format!(
            "Network centrality and parameterized graph invariants computing {}, degeneracy $k = {}$, vertex cover $|C| = {}$, and treewidth $\\mathrm{{tw}}(G) = {}$.",
            metric_str,
            self.global_degeneracy,
            self.vertex_cover_nodes.len(),
            self.global_treewidth
        )
    }
    fn phonetic_description(&self) -> String {
        "Network centrality, degeneracy, vertex cover, and treewidth theory".into()
    }
    fn theory_citation(&self) -> String {
        "Watts & Strogatz (1998); Newman (2010); Komusiewicz et al. (2025)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "DegreeCentrality".into(),
            "Normalized count of incident edges per vertex".into(),
        );
        map.insert(
            "ClosenessCentrality".into(),
            "Reciprocal of average geodesic distance to all reachable vertices".into(),
        );
        map.insert(
            "ClusteringCoefficient".into(),
            "Fraction of neighbor pairs that are mutually connected".into(),
        );
        map
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
        description: "Degree centrality, closeness centrality, clustering, degeneracy, vertex cover, and exact treewidth.",
        category: "Network Science",
        icon: "📊",
        domain: "graph_theory",
        tool_name: Some("NetworkMetricsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

// [cite:graph_parameters_rust]
