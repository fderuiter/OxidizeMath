#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkMetricsTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::{
    build_graph_primitives, ExportPrimitive, GraphEdgeSpec, GraphNodeSpec,
};
use math_explorer::pure_math::graph_theory::parameters::network_metrics::{
    closeness_centrality, clustering_coefficients, degree_centrality,
};

fn interpolate_metric_color(t: f32, palette: &SemanticPalette) -> egui::Color32 {
    let low = palette.axis_z;
    let high = palette.axis_x;
    let clamped = t.clamp(0.0, 1.0);
    let lerp =
        |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * clamped).round() as u8;
    egui::Color32::from_rgb(
        lerp(low.r(), high.r()),
        lerp(low.g(), high.g()),
        lerp(low.b(), high.b()),
    )
}

impl NetworkMetricsTool {
    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();

        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &val in self.metric_values.values() {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
        let range = max_val - min_val;

        let nodes: Vec<GraphNodeSpec> = sorted_ids
            .into_iter()
            .map(|id| {
                let pos = self.node_positions[&id];
                let val = self.metric_values.get(&id).copied().unwrap_or(0.0);
                let t = if range > 1e-12 {
                    ((val - min_val) / range) as f32
                } else {
                    0.5
                };
                GraphNodeSpec {
                    id,
                    pos: [f64::from(pos.x), f64::from(pos.y)],
                    fill: interpolate_metric_color(t, &palette),
                    stroke_color: palette.node_stroke,
                    radius: 11.0,
                    label: format!("N{id}: {val:.2}"),
                }
            })
            .collect();

        let edges: Vec<GraphEdgeSpec> = self
            .edges
            .iter()
            .filter(|&&(u, v, _)| {
                self.node_positions.contains_key(&u) && self.node_positions.contains_key(&v)
            })
            .map(|&(u, v, _)| GraphEdgeSpec {
                from_id: u,
                to_id: v,
                color: palette.graph_edge,
                width: 1.8,
                directed: false,
                label: format!("{u}-{v}"),
            })
            .collect();

        build_graph_primitives(&nodes, &edges, true)
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let deg_map = degree_centrality(&self.graph);
        let close_map = closeness_centrality(&self.graph);
        let clust_map = clustering_coefficients(&self.graph);

        let mut sorted_ids: Vec<usize> = self.node_indices.keys().copied().collect();
        sorted_ids.sort_unstable();
        let node_rows: Vec<Vec<ExportCell>> = sorted_ids
            .into_iter()
            .map(|id| {
                let idx = self.node_indices[&id];
                vec![
                    ExportCell::UInt(id as u64),
                    ExportCell::Float(deg_map.get(&idx).copied().unwrap_or(0.0)),
                    ExportCell::Float(close_map.get(&idx).copied().unwrap_or(0.0)),
                    ExportCell::Float(clust_map.get(&idx).copied().unwrap_or(0.0)),
                ]
            })
            .collect();

        let summary_rows = vec![vec![
            ExportCell::UInt(self.node_positions.len() as u64),
            ExportCell::UInt(self.edges.len() as u64),
            ExportCell::UInt(self.global_diameter.unwrap_or(0) as u64),
            ExportCell::Float(self.global_clustering.unwrap_or(0.0)),
        ]];

        vec![
            ExportTable::new(
                "Node Metrics",
                vec![
                    "node_id".into(),
                    "degree_centrality".into(),
                    "closeness_centrality".into(),
                    "clustering_coefficient".into(),
                ],
                node_rows,
            ),
            ExportTable::new(
                "Global Summary",
                vec![
                    "nodes".into(),
                    "edges".into(),
                    "diameter".into(),
                    "avg_clustering_coefficient".into(),
                ],
                summary_rows,
            ),
        ]
    }
}
