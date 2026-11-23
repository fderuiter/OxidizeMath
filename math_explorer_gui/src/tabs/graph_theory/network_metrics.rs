use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::graph::Graph;
use math_explorer::pure_math::graph_theory::parameters::degree::degeneracy;
use math_explorer::pure_math::graph_theory::parameters::modulator::vertex_cover;
use math_explorer::pure_math::graph_theory::parameters::network_metrics::{
    average_clustering_coefficient, closeness_centrality, clustering_coefficients,
    degree_centrality, diameter,
};
use petgraph::graph::NodeIndex;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[path = "network_metrics/topology_io.rs"]
mod topology_io;

#[derive(PartialEq, Eq, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum MetricType {
    DegreeCentrality,
    ClosenessCentrality,
    ClusteringCoefficient,
    VertexCover,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkMetricsState {
    selected_metric: MetricType,
    #[serde(default = "default_true")]
    highlight_vertex_cover: bool,
}

const fn default_true() -> bool {
    true
}

pub struct NetworkMetricsTool {
    pub(crate) graph: Graph<Pos2, f64>,
    pub(crate) node_indices: HashMap<usize, NodeIndex>,
    pub(crate) node_positions: HashMap<usize, Pos2>,
    pub(crate) node_labels: BTreeMap<usize, String>,
    pub(crate) edges: Vec<(usize, usize, f64)>,

    pub(crate) selected_metric: MetricType,
    highlight_vertex_cover: bool,
    metric_values: HashMap<usize, f64>,
    pub(crate) global_diameter: Option<usize>,
    pub(crate) global_clustering: Option<f64>,
    pub(crate) global_degeneracy: Option<usize>,
    pub(crate) vertex_cover_ids: BTreeSet<usize>,
}

impl Default for NetworkMetricsTool {
    fn default() -> Self {
        let mut tool = Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: HashMap::new(),
            node_labels: BTreeMap::new(),
            edges: Vec::new(),
            selected_metric: MetricType::DegreeCentrality,
            highlight_vertex_cover: true,
            metric_values: HashMap::new(),
            global_diameter: None,
            global_clustering: None,
            global_degeneracy: None,
            vertex_cover_ids: BTreeSet::new(),
        };
        tool.build_sample_graph();
        tool
    }
}

impl NetworkMetricsTool {
    /// Returns the computed $k$-degeneracy of the active graph.
    #[must_use]
    pub fn degeneracy(&self) -> usize {
        self.global_degeneracy.unwrap_or(0)
    }

    pub(crate) fn build_sample_graph(&mut self) {
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
                self.graph.add_edge(self.node_indices[&prev_id], idx, 1.0);
            }
        }
        let (first_id, last_id) = (1, num_outer);
        self.edges.push((last_id, first_id, 1.0));
        self.graph.add_edge(
            self.node_indices[&last_id],
            self.node_indices[&first_id],
            1.0,
        );

        self.compute_metrics();
    }

    pub(crate) fn compute_metrics(&mut self) {
        self.metric_values.clear();
        self.global_degeneracy = Some(degeneracy(&self.graph));
        let cover_indices = vertex_cover(&self.graph);
        self.vertex_cover_ids.clear();
        for (&id, &idx) in &self.node_indices {
            if cover_indices.contains(&idx) {
                self.vertex_cover_ids.insert(id);
            }
        }

        match self.selected_metric {
            MetricType::VertexCover => {
                for &id in self.node_indices.keys() {
                    let v = if self.vertex_cover_ids.contains(&id) {
                        1.0
                    } else {
                        0.0
                    };
                    self.metric_values.insert(id, v);
                }
            }
            MetricType::DegreeCentrality => self.populate_from_map(degree_centrality(&self.graph)),
            MetricType::ClosenessCentrality => {
                self.populate_from_map(closeness_centrality(&self.graph));
            }
            MetricType::ClusteringCoefficient => {
                self.populate_from_map(clustering_coefficients(&self.graph));
            }
        }

        self.global_diameter = Some(diameter(&self.graph));
        self.global_clustering = Some(average_clustering_coefficient(&self.graph));
    }

    fn populate_from_map(&mut self, raw_metrics: HashMap<NodeIndex, f64>) {
        for (&id, &idx) in &self.node_indices {
            if let Some(&val) = raw_metrics.get(&idx) {
                self.metric_values.insert(id, val);
            }
        }
    }

    fn render_sidebar(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "network_metrics_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Network Metrics");
            ui.separator();
            ui.label("Select Metric to Visualize:");
            let mut changed = false;
            for (metric, label) in [
                (MetricType::DegreeCentrality, "Degree Centrality"),
                (MetricType::ClosenessCentrality, "Closeness Centrality"),
                (
                    MetricType::ClusteringCoefficient,
                    "Local Clustering Coefficient",
                ),
                (MetricType::VertexCover, "2-Approx Vertex Cover"),
            ] {
                changed |= ui
                    .radio_value(&mut self.selected_metric, metric, label)
                    .clicked();
            }
            if changed {
                self.compute_metrics();
            }
            ui.checkbox(
                &mut self.highlight_vertex_cover,
                "Highlight 2-Approx Vertex Cover Rings",
            )
            .accessible_hover_text(
                "Draw a highlighted outer ring around vertices in the 2-approximate vertex cover",
            );

            ui.separator();
            self.render_global_parameters(ui);
            ui.separator();
            ui.label(
                "Nodes are shaded from blue (low) to red (high); gold double-rings mark the 2-approximate vertex cover.",
            );
        });
    }

    fn render_global_parameters(&self, ui: &mut egui::Ui) {
        ui.heading("Global Parameters");
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
        if let Some(k) = self.global_degeneracy {
            ui.label(format!("Degeneracy (k-core): {k}"));
        }
        let cover_list: Vec<String> = self
            .vertex_cover_ids
            .iter()
            .take(12)
            .map(usize::to_string)
            .collect();
        let suffix = if self.vertex_cover_ids.len() > 12 {
            ", ..."
        } else {
            ""
        };
        ui.label(format!(
            "2-Approx Vertex Cover (|C| = {}): {{{}{suffix}}}",
            self.vertex_cover_ids.len(),
            cover_list.join(", ")
        ));
    }

    fn render_metrics_canvas(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::focusable_noninteractive());
            let label = "Network Metrics Visualization";
            let _ = response.clone().accessible_hover_text(label);
            let summary = format!(
                "{} nodes, {} edges, degeneracy {}, vertex cover size {}",
                self.node_positions.len(),
                self.edges.len(),
                self.degeneracy(),
                self.vertex_cover_ids.len()
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
            self.paint_metric_edges(&painter, &to_screen, &palette);
            self.paint_metric_nodes(&painter, &to_screen, &palette);
        });
    }

    fn paint_metric_edges(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        palette: &crate::app::SemanticPalette,
    ) {
        for &(u, v, _) in &self.edges {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                painter.line_segment(
                    [
                        to_screen.transform_pos(pos_u),
                        to_screen.transform_pos(pos_v),
                    ],
                    (2.0, palette.graph_edge),
                );
            }
        }
    }

    fn paint_metric_nodes(
        &self,
        painter: &egui::Painter,
        to_screen: &eframe::egui::emath::RectTransform,
        palette: &crate::app::SemanticPalette,
    ) {
        let mut max_val = 0.0_f64;
        let mut min_val = f64::MAX;
        for &val in self.metric_values.values() {
            max_val = max_val.max(val);
            min_val = min_val.min(val);
        }
        let range = max_val - min_val;

        for (&id, &pos) in &self.node_positions {
            let screen_pos = to_screen.transform_pos(pos);
            let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
            let t = if range > 0.0 {
                ((val - min_val) / range) as f32
            } else {
                0.5
            };
            let (low, high) = (palette.axis_z, palette.axis_x);
            let lerp = |a: u8, b: u8| {
                (f32::from(a) + (f32::from(b) - f32::from(a)) * t.clamp(0.0, 1.0)).round() as u8
            };
            let fill = egui::Color32::from_rgb(
                lerp(low.r(), high.r()),
                lerp(low.g(), high.g()),
                lerp(low.b(), high.b()),
            );
            let in_vc = self.vertex_cover_ids.contains(&id);
            if in_vc && self.highlight_vertex_cover {
                painter.circle_stroke(screen_pos, 24.0, (2.5, palette.warning));
            }
            painter.circle(screen_pos, 19.0, fill, (1.2, palette.node_stroke));
            let caption = if self.selected_metric == MetricType::VertexCover {
                if in_vc {
                    format!("{id}:C")
                } else {
                    id.to_string()
                }
            } else {
                format!("{val:.2}")
            };
            painter.text(
                screen_pos,
                egui::Align2::CENTER_CENTER,
                caption,
                egui::FontId::proportional(11.5),
                crate::app::contrasting_text_color(fill),
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
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::topology_only()
    }
    fn export_topology_document(
        &self,
    ) -> Option<
        Result<math_explorer::pure_math::graph_theory::graph_io::GraphTopologyDocument, String>,
    > {
        Some(self.build_topology_document())
    }
    fn apply_topology_document(
        &mut self,
        doc: &math_explorer::pure_math::graph_theory::graph_io::GraphTopologyDocument,
    ) -> Result<(), String> {
        self.load_topology_document(doc)
    }
    fn reset_default_topology(&mut self) {
        self.restore_sample_wheel();
    }
    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&NetworkMetricsState {
            selected_metric: self.selected_metric,
            highlight_vertex_cover: self.highlight_vertex_cover,
        })
        .ok()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: NetworkMetricsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid NetworkMetrics state: {e}"))?;
        self.selected_metric = loaded.selected_metric;
        self.highlight_vertex_cover = loaded.highlight_vertex_cover;
        self.compute_metrics();
        Ok(())
    }
    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.render_sidebar(parent_ui);
        self.render_metrics_canvas(parent_ui);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkMetricsTool",
        display_name: "Network Metrics",
        domain: "graph_theory",
        tags: &[
            "centrality",
            "clustering",
            "watts-strogatz",
            "degree",
            "closeness",
            "degeneracy",
            "vertex_cover",
        ],
        build: || Box::new(NetworkMetricsTool::default()),
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
            MetricType::VertexCover => {
                "maximal-matching 2-approximate vertex cover $|C| \\le 2 \\cdot \\text{vc}(G)$"
            }
        };
        format!(
            "Parameterized and complex network analysis computing {}, $k$-degeneracy ($k = {}$), and 2-approximate vertex cover ($|C| = {}$).",
            metric_str,
            self.degeneracy(),
            self.vertex_cover_ids.len()
        )
    }
    fn phonetic_description(&self) -> String {
        "Network centrality, degeneracy, and vertex cover theory".into()
    }
    fn theory_citation(&self) -> String {
        "Parameterized Algorithms (Cygan et al., 2015); Collective Dynamics of 'Small-World' Networks (Watts & Strogatz, 1998); Networks (Newman, 2010)".into()
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
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
        map.insert(
            "VertexCover".into(),
            "Greedy maximal-matching 2-approximation to minimum vertex cover alongside k-degeneracy".into(),
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
        description: "Degree centrality, closeness centrality, clustering, k-degeneracy, and 2-approximate vertex cover.",
        category: "Network Science",
        icon: "📊",
        domain: "graph_theory",
        tool_name: Some("NetworkMetricsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
