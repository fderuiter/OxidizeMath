#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::shared_topology::CompactTopologyState;
use crate::accessibility::AccessibleHoverText;
use crate::app::workspace_bus::SharedWorkspaceBus;
use crate::app::{render_topology_source_bar, TopologySource};
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

#[path = "network_metrics/bus_sync.rs"]
mod bus_sync;
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

    synced_ws_bus_revision: Option<u64>,
    using_bus_topology: bool,
    latest_linked_coherence: Option<f64>,
    recompute_count: u64,
    revision: u64,
    file_io: super::io_helpers::GraphFileIoState,
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
            synced_ws_bus_revision: None,
            using_bus_topology: false,
            latest_linked_coherence: None,
            recompute_count: 0,
            revision: 1,
            file_io: super::io_helpers::GraphFileIoState::default(),
        };
        tool.build_sample_graph();
        tool
    }
}

impl NetworkMetricsTool {
    pub(crate) fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1).max(1);
    }

    pub(crate) fn compute_metrics(&mut self) {
        self.metric_values.clear();
        self.recompute_count = self.recompute_count.saturating_add(1);
        if self.node_positions.is_empty() {
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
            if self.using_bus_topology {
                self.render_ws_bus_badge(ui);
            } else if render_topology_source_bar(
                ui,
                "network_metrics",
                &mut self.topology_source,
                self.synced_bus_revision,
                None,
            ) {
                self.bump_revision();
                self.sync_shared_topology(&ctx);
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

    fn render_metric_selector(&mut self, ui: &mut egui::Ui) {
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

    fn render_invariants_summary(&mut self, ui: &mut egui::Ui) {
        ui.heading("Structural Invariants");
        ui.label(format!(
            "Nodes: {} | Edges: {}",
            self.node_positions.len(),
            self.edges.len()
        ));
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
        if let Some(coh) = self.latest_linked_coherence {
            ui.label(format!("Linked Coherence R(t): {coh:.3}"));
        }
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
        for &(u, v, _) in &self.edges {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                let su = to_screen.transform_pos(pos_u);
                let sv = to_screen.transform_pos(pos_v);
                painter.line_segment([su, sv], (2.0, palette.graph_edge));
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
                self.synced_bus_revision
                    .or(self.synced_ws_bus_revision)
                    .unwrap_or(0) as f64,
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
    fn sync_workspace_bus(&mut self, bus: &mut SharedWorkspaceBus) {
        self.sync_bus_channels(bus);
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
        let on_bus = self.using_bus_topology || self.synced_bus_revision.is_some();
        let custom_topology = (!on_bus && self.is_custom_topology)
            .then(|| self.export_shared_topology().to_compact_state());
        serde_json::to_string(&NetworkMetricsState {
            selected_metric: self.selected_metric,
            highlight_vertex_cover: self.highlight_vertex_cover,
            custom_topology,
            topology_source: self.topology_source,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: NetworkMetricsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid NetworkMetrics state: {e}"))?;
        if let Some(compact) = &loaded.custom_topology {
            compact.validate()?;
            self.synced_bus_revision = None;
            self.synced_ws_bus_revision = None;
            self.load_compact_topology(compact);
        } else if self.is_custom_topology
            || self.using_bus_topology
            || self.synced_bus_revision.take().is_some()
        {
            self.using_bus_topology = false;
            self.synced_ws_bus_revision = None;
            self.build_sample_graph();
        }
        self.selected_metric = loaded.selected_metric;
        self.highlight_vertex_cover = loaded.highlight_vertex_cover;
        self.topology_source = loaded.topology_source;
        self.bump_revision();
        self.compute_metrics();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        if crate::app::workspace_bus::with_linked_bus(|bus| {
            self.sync_workspace_bus(bus);
        })
        .is_none()
        {
            self.on_bus_unlinked();
        }
        if !self.using_bus_topology {
            let ctx = parent_ui.ctx().clone();
            self.sync_shared_topology(&ctx);
        }
        self.render_side_panel(parent_ui);
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

#[cfg(test)]
#[path = "network_metrics/tests.rs"]
mod tests;
