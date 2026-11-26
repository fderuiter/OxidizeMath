#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[path = "network_metrics/bus_sync.rs"]
mod bus_sync;

use super::shared_topology::CompactTopologyState;
use crate::accessibility::AccessibleHoverText;
use crate::app::data_bus::ui::render_follow_workspace_toggle;
use crate::app::data_bus::SharedGraph;
use crate::app::{render_topology_source_bar, SharedSimulationBus, TopologySource};
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::graph::Graph;
use math_explorer::pure_math::graph_theory::parameters::{
    degree::degeneracy,
    modulator::vertex_cover,
    network_metrics::{
        average_clustering_coefficient, closeness_centrality, clustering_coefficients,
        degree_centrality, diameter,
    },
    treewidth::{treewidth_in_place, TreewidthScratch},
};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

#[path = "network_metrics/io_export.rs"]
mod io_export;
pub use io_export::metrics_state_json_from_shared;

fn default_true() -> bool {
    true
}

fn is_true(val: &bool) -> bool {
    *val
}

/// Per-vertex network centrality metric visualized on the canvas.
#[derive(Debug, PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum MetricType {
    /// Normalized incident degree centrality.
    DegreeCentrality,
    /// Reciprocal geodesic distance closeness centrality.
    ClosenessCentrality,
    /// Local Watts-Strogatz triangle clustering coefficient.
    ClusteringCoefficient,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NetworkMetricsState {
    pub(crate) selected_metric: MetricType,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub(crate) highlight_vertex_cover: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub(crate) follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) frozen_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) custom_topology: Option<CompactTopologyState>,
    #[serde(default, skip_serializing_if = "TopologySource::is_default")]
    pub(crate) topology_source: TopologySource,
}

/// Interactive network centrality and parameterized complexity invariant analyzer.
pub struct NetworkMetricsTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    edges: Vec<(usize, usize, f64)>,
    directed: bool,
    is_custom_topology: bool,

    selected_metric: MetricType,
    pub topology_source: TopologySource,
    synced_bus_revision: Option<u64>,
    highlight_vertex_cover: bool,
    metric_values: HashMap<usize, f64>,
    global_diameter: Option<usize>,
    global_clustering: Option<f64>,
    global_degeneracy: usize,
    global_treewidth: usize,
    vertex_cover_nodes: BTreeSet<usize>,
    tw_scratch: TreewidthScratch,
    revision: u64,
    file_io: super::io_helpers::GraphFileIoState,
    pub(crate) follow_workspace_graph: bool,
    pub(crate) applied_bus_revision: u64,
    active_shared_graph: Option<Arc<SharedGraph>>,
}

impl Default for NetworkMetricsTool {
    fn default() -> Self {
        let mut tool = Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: HashMap::new(),
            edges: Vec::new(),
            directed: false,
            is_custom_topology: false,
            selected_metric: MetricType::DegreeCentrality,
            topology_source: TopologySource::SharedBus,
            synced_bus_revision: None,
            highlight_vertex_cover: true,
            metric_values: HashMap::new(),
            global_diameter: None,
            global_clustering: None,
            global_degeneracy: 0,
            global_treewidth: 0,
            vertex_cover_nodes: BTreeSet::new(),
            tw_scratch: TreewidthScratch::with_capacity(32),
            revision: 1,
            file_io: super::io_helpers::GraphFileIoState::default(),
            follow_workspace_graph: true,
            applied_bus_revision: 0,
            active_shared_graph: None,
        };
        tool.build_sample_graph();
        tool
    }
}

impl NetworkMetricsTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    /// Computed graph degeneracy $k$.
    #[cfg(test)]
    #[must_use]
    pub fn degeneracy_value(&self) -> usize {
        self.global_degeneracy
    }

    /// Computed graph treewidth (exact for $|V| \le 24$, min-fill upper bound for $|V| > 24$).
    #[cfg(test)]
    #[must_use]
    pub fn treewidth_value(&self) -> usize {
        self.global_treewidth
    }

    /// Computed 2-approximation vertex cover modulator vertex IDs.
    #[cfg(test)]
    #[must_use]
    pub fn vertex_cover_nodes(&self) -> &BTreeSet<usize> {
        &self.vertex_cover_nodes
    }

    /// Computed geodesic diameter.
    #[cfg(test)]
    #[must_use]
    pub fn diameter_value(&self) -> Option<usize> {
        self.global_diameter
    }

    fn build_sample_graph(&mut self) {
        self.graph = Graph::new();
        self.node_indices.clear();
        self.node_positions.clear();
        self.edges.clear();
        self.directed = false;
        self.is_custom_topology = false;
        let center = Pos2::new(300.0, 300.0);
        let center_id = 0;
        let center_idx = self.graph.add_node(center);
        self.node_indices.insert(center_id, center_idx);
        self.node_positions.insert(center_id, center);

        let num_outer = 6;
        let radius = 150.0;
        for i in 0..num_outer {
            let angle = 2.0 * std::f32::consts::PI * (i as f32) / (num_outer as f32);
            let pos = Pos2::new(
                center.x + radius * angle.cos(),
                center.y + radius * angle.sin(),
            );
            let id = i + 1;
            let idx = self.graph.add_node(pos);
            self.node_indices.insert(id, idx);
            self.node_positions.insert(id, pos);
            self.edges.push((center_id, id, 1.0));
            self.graph.add_edge(center_idx, idx, 1.0);
            if i > 0 {
                let prev_id = i;
                self.edges.push((prev_id, id, 1.0));
                let prev_idx = self.node_indices[&prev_id];
                self.graph.add_edge(prev_idx, idx, 1.0);
            }
        }
        self.edges.push((num_outer, 1, 1.0));
        self.graph
            .add_edge(self.node_indices[&num_outer], self.node_indices[&1], 1.0);
        self.compute_metrics();
    }

    pub(crate) fn compute_metrics(&mut self) {
        self.metric_values.clear();
        if self.node_indices.is_empty() {
            self.global_diameter = Some(0);
            self.global_clustering = Some(0.0);
            self.global_degeneracy = 0;
            self.global_treewidth = 0;
            self.vertex_cover_nodes.clear();
            return;
        }
        let raw_metrics = match self.selected_metric {
            MetricType::DegreeCentrality => degree_centrality(&self.graph),
            MetricType::ClosenessCentrality => closeness_centrality(&self.graph),
            MetricType::ClusteringCoefficient => clustering_coefficients(&self.graph),
        };
        let idx_to_id: HashMap<NodeIndex, usize> = self
            .node_indices
            .iter()
            .map(|(&id, &ni)| (ni, id))
            .collect();
        for (&id, &idx) in &self.node_indices {
            if let Some(&val) = raw_metrics.get(&idx) {
                self.metric_values.insert(id, val);
            }
        }
        self.global_diameter = Some(diameter(&self.graph));
        self.global_clustering = Some(average_clustering_coefficient(&self.graph));
        self.global_degeneracy = degeneracy(&self.graph);
        self.global_treewidth = treewidth_in_place(&self.graph, &mut self.tw_scratch);
        self.vertex_cover_nodes = vertex_cover(&self.graph)
            .into_iter()
            .filter_map(|ni| idx_to_id.get(&ni).copied())
            .collect();
    }

    pub(crate) fn metric_min_and_range(&self) -> (f64, f64) {
        if self.metric_values.is_empty() {
            return (0.0, 0.0);
        }
        let mut max_val = f64::NEG_INFINITY;
        let mut min_val = f64::INFINITY;
        for &val in self.metric_values.values() {
            max_val = max_val.max(val);
            min_val = min_val.min(val);
        }
        (min_val, (max_val - min_val).max(0.0))
    }

    pub(crate) fn node_metric_color(
        &self,
        val: f64,
        min_val: f64,
        range: f64,
        palette: &crate::app::SemanticPalette,
    ) -> egui::Color32 {
        let t = if range > 1e-12 {
            ((val - min_val) / range) as f32
        } else {
            0.5
        }
        .clamp(0.0, 1.0);
        let low = palette.axis_z;
        let high = palette.axis_x;
        let lerp = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * t).round() as u8;
        egui::Color32::from_rgb(
            lerp(low.r(), high.r()),
            lerp(low.g(), high.g()),
            lerp(low.b(), high.b()),
        )
    }

    fn render_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "network_metrics_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Network Metrics");
            if render_topology_source_bar(
                ui,
                "network_metrics",
                &mut self.topology_source,
                self.synced_bus_revision,
                None,
            ) {
                self.bump_revision();
                self.sync_shared_topology(&ctx);
                if !SharedSimulationBus::is_link_topology_enabled(&ctx) {
                    self.sync_from_bus(&ctx);
                }
            }
            ui.separator();
            let rev_opt = (self.applied_bus_revision > 0).then_some(self.applied_bus_revision);
            if render_follow_workspace_toggle(
                ui,
                &mut self.follow_workspace_graph,
                "Network Metrics",
                rev_opt,
            ) && self.follow_workspace_graph
            {
                self.is_custom_topology = false;
                self.applied_bus_revision = 0;
                self.sync_from_bus(ui.ctx());
            }
            ui.separator();
            self.render_metric_selector(ui);
            ui.separator();
            self.render_invariants_summary(ui);
            ui.separator();
            self.render_topology_io_controls(ui);
            ui.separator();
            ui.label(
                "Nodes are heat-mapped by centrality; gold rings mark the vertex-cover modulator.",
            );
        });
    }

    fn render_canvas(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::focusable_noninteractive());
            let label = "Network Metrics Visualization";
            let _ = response.clone().accessible_hover_text(label);
            let summary = format!(
                "{} nodes, {} edges, treewidth {}, degeneracy {}",
                self.node_positions.len(),
                self.edges.len(),
                self.global_treewidth,
                self.global_degeneracy
            );
            crate::accessibility::register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                label,
                Some(&summary),
            );
            crate::accessibility::paint_canvas_focus_ring(ui, &response);

            let to_screen = eframe::egui::emath::RectTransform::from_to(
                eframe::egui::Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            self.paint_edges(&painter, &to_screen, &palette);
            self.paint_nodes(&painter, &to_screen, &palette);
        });
    }

    fn paint_edges(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        palette: &crate::app::SemanticPalette,
    ) {
        for &(u, v, w) in &self.edges {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let edge_color = if w < 0.0 {
                    palette.error
                } else {
                    palette.graph_edge
                };
                let su = to_screen.transform_pos(pos_u);
                let sv = to_screen.transform_pos(pos_v);
                painter.line_segment([su, sv], (2.0, edge_color));
            }
        }
    }

    fn paint_nodes(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        palette: &crate::app::SemanticPalette,
    ) {
        let (min_val, range) = self.metric_min_and_range();
        for (&id, &pos) in &self.node_positions {
            let screen_pos = to_screen.transform_pos(pos);
            let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
            let fill_color = self.node_metric_color(val, min_val, range, palette);
            painter.circle(screen_pos, 20.0, fill_color, (1.2, palette.node_stroke));
            if self.highlight_vertex_cover && self.vertex_cover_nodes.contains(&id) {
                painter.circle_stroke(screen_pos, 24.5, (2.8, palette.warning));
            }
            painter.text(
                screen_pos,
                egui::Align2::CENTER_CENTER,
                format!("{val:.2}"),
                egui::FontId::proportional(12.0),
                crate::app::contrasting_text_color(fill_color),
            );
        }
    }
}

impl InteractiveTool for NetworkMetricsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Network Metrics"
    }
    fn domain(&self) -> &'static str {
        "graph_theory"
    }
    fn observables(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("node_count", self.node_positions.len() as f64),
            ("edge_count", self.edges.len() as f64),
            (
                "synced_bus_revision",
                self.synced_bus_revision.unwrap_or(0) as f64,
            ),
        ]
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::plot()
    }
    fn export_plot_primitives(&self) -> Vec<egui_plot::ExportPrimitive> {
        self.build_metrics_plot_primitives()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.build_metrics_datasets()
    }
    fn state_revision(&self) -> u64 {
        self.revision.max(1)
    }
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn save_state(&self) -> Option<String> {
        self.save_metrics_state()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.load_metrics_state(state)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        self.sync_shared_topology(&ctx);
        if !SharedSimulationBus::is_link_topology_enabled(&ctx) {
            self.sync_from_bus(&ctx);
        }
        self.render_side_panel(parent_ui);
        if !SharedSimulationBus::is_link_topology_enabled(&ctx) {
            self.sync_from_bus(&ctx);
        }
        self.render_canvas(parent_ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkMetricsTool",
        display_name: "Network Metrics",
        domain: "graph_theory",
        tags: &["centrality", "degeneracy", "vertex-cover", "treewidth", "invariants"],
        build: || Box::new(NetworkMetricsTool::default()),
    }
}
