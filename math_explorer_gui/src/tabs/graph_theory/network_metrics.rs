use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use eframe::egui::Pos2;
use math_explorer::pure_math::graph_theory::graph::Graph;
use math_explorer::pure_math::graph_theory::parameters::network_metrics::{
    average_clustering_coefficient, closeness_centrality, clustering_coefficients,
    degree_centrality, diameter,
};
use petgraph::graph::NodeIndex;
use std::collections::HashMap;

#[path = "network_metrics/export.rs"]
mod export;

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum MetricType {
    DegreeCentrality,
    ClosenessCentrality,
    ClusteringCoefficient,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NetworkMetricsState {
    selected_metric: MetricType,
}

pub struct NetworkMetricsTool {
    graph: Graph<Pos2, f64>,
    node_indices: HashMap<usize, NodeIndex>,
    node_positions: HashMap<usize, Pos2>,
    edges: Vec<(usize, usize, f64)>,

    selected_metric: MetricType,
    metric_values: HashMap<usize, f64>,
    global_diameter: Option<usize>,
    global_clustering: Option<f64>,
}

impl Default for NetworkMetricsTool {
    fn default() -> Self {
        let mut tool = Self {
            graph: Graph::new(),
            node_indices: HashMap::new(),
            node_positions: HashMap::new(),
            edges: Vec::new(),
            selected_metric: MetricType::DegreeCentrality,
            metric_values: HashMap::new(),
            global_diameter: None,
            global_clustering: None,
        };
        tool.build_sample_graph();
        tool
    }
}

impl NetworkMetricsTool {
    fn build_sample_graph(&mut self) {
        // Create a star-like graph to show clear centrality differences
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

            // Connect to center
            self.edges.push((center_id, id, 1.0));
            self.graph.add_edge(center_idx, idx, 1.0);

            // Connect outer nodes forming a ring
            if i > 0 {
                let prev_id = i;
                self.edges.push((prev_id, id, 1.0));
                let prev_idx = self.node_indices[&prev_id];
                self.graph.add_edge(prev_idx, idx, 1.0);
            }
        }

        // Connect last outer node to first outer node to close the ring
        let first_id = 1;
        let last_id = num_outer;
        self.edges.push((last_id, first_id, 1.0));
        let first_idx = self.node_indices[&first_id];
        let last_idx = self.node_indices[&last_id];
        self.graph.add_edge(last_idx, first_idx, 1.0);

        self.compute_metrics();
    }

    fn compute_metrics(&mut self) {
        self.metric_values.clear();

        let raw_metrics = match self.selected_metric {
            MetricType::DegreeCentrality => degree_centrality(&self.graph),
            MetricType::ClosenessCentrality => closeness_centrality(&self.graph),
            MetricType::ClusteringCoefficient => clustering_coefficients(&self.graph),
        };

        for (&id, &idx) in &self.node_indices {
            if let Some(&val) = raw_metrics.get(&idx) {
                self.metric_values.insert(id, val);
            }
        }

        self.global_diameter = Some(diameter(&self.graph));
        self.global_clustering = Some(average_clustering_coefficient(&self.graph));
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
        crate::framework::ExportCapabilities::plot()
    }
    fn export_plot_primitives(&self) -> Vec<egui_plot::export_primitives::ExportPrimitive> {
        self.cold_export_primitives()
    }
    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        self.cold_export_datasets()
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&NetworkMetricsState {
            selected_metric: self.selected_metric,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: NetworkMetricsState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid NetworkMetrics state: {e}"))?;
        self.selected_metric = loaded.selected_metric;
        self.compute_metrics();
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("network_metrics_controls").show(ctx, |ui| {
            ui.heading("Network Metrics");
            ui.separator();

            ui.label("Select Metric to Visualize:");
            let mut changed = false;

            if ui.radio_value(&mut self.selected_metric, MetricType::DegreeCentrality, "Degree Centrality").clicked() {
                changed = true;
            }
            if ui.radio_value(&mut self.selected_metric, MetricType::ClosenessCentrality, "Closeness Centrality").clicked() {
                changed = true;
            }
            if ui.radio_value(&mut self.selected_metric, MetricType::ClusteringCoefficient, "Local Clustering Coefficient").clicked() {
                changed = true;
            }

            if changed {
                self.compute_metrics();
            }

            ui.separator();

            ui.heading("Global Metrics");
            if let Some(d) = self.global_diameter {
                ui.label(format!("Diameter: {}", d));
            }
            if let Some(c) = self.global_clustering {
                ui.label(format!("Avg Clustering Coefficient: {:.3}", c));
            }

            ui.separator();
            ui.label("Nodes are color-coded based on the selected metric value. Red indicates higher values.");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::hover());
            let _ = response
                .clone()
                .accessible_hover_text("Network Metrics Visualization");

            let to_screen = eframe::egui::emath::RectTransform::from_to(
                eframe::egui::Rect::from_min_size(Pos2::ZERO, response.rect.size()),
                response.rect,
            );

            // Draw edges
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            for &(u, v, _) in &self.edges {
                if let (Some(&pos_u), Some(&pos_v)) =
                    (self.node_positions.get(&u), self.node_positions.get(&v))
                {
                    let screen_pos_u = to_screen.transform_pos(pos_u);
                    let screen_pos_v = to_screen.transform_pos(pos_v);
                    painter.line_segment([screen_pos_u, screen_pos_v], (2.0, palette.graph_edge));
                }
            }

            // Find max/min metric values for color scaling
            let mut max_val = 0.0_f64;
            let mut min_val = f64::MAX;
            for &val in self.metric_values.values() {
                if val > max_val {
                    max_val = val;
                }
                if val < min_val {
                    min_val = val;
                }
            }
            let range = max_val - min_val;

            // Draw nodes
            for (&id, &pos) in &self.node_positions {
                let screen_pos = to_screen.transform_pos(pos);
                let val = self.metric_values.get(&id).copied().unwrap_or(0.0);

                // Color mapping: low metric (blue) to high metric (red)
                let t = if range > 0.0 {
                    ((val - min_val) / range) as f32
                } else {
                    0.5
                };

                let low = palette.axis_z;
                let high = palette.axis_x;
                let lerp = |a: u8, b: u8| {
                    (f32::from(a) + (f32::from(b) - f32::from(a)) * t.clamp(0.0, 1.0)).round() as u8
                };
                let fill_color =
                    egui::Color32::from_rgb(lerp(low.r(), high.r()), lerp(low.g(), high.g()), lerp(low.b(), high.b()));

                painter.circle(screen_pos, 20.0, fill_color, (1.0, palette.node_stroke));

                painter.text(
                    screen_pos,
                    egui::Align2::CENTER_CENTER,
                    format!("{:.2}", val),
                    egui::FontId::proportional(12.0),
                    crate::app::contrasting_text_color(fill_color),
                );
            }
        });
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "NetworkMetricsTool",
        display_name: "Network Metrics",
        domain: "graph_theory",
        tags: &[],
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
        };
        format!(
            "Complex network topological parameter analysis computing {}.",
            metric_str
        )
    }
    fn phonetic_description(&self) -> String {
        "Network metrics and graph centrality theory".into()
    }
    fn theory_citation(&self) -> String {
        "Collective Dynamics of 'Small-World' Networks (Watts & Strogatz, 1998); Networks (Newman, 2010)".into()
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
        description: "Degree centrality, closeness centrality, local clustering coefficient, and graph diameter.",
        category: "Network Science",
        icon: "📊",
        domain: "graph_theory",
        tool_name: Some("NetworkMetricsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
