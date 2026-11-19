#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GraphEditorTool;
use egui_plot::export_primitives::ExportPrimitive;
use math_explorer::pure_math::graph_theory::layout::force_directed_layout;

impl GraphEditorTool {
    pub(super) fn step_export_layout(&mut self) {
        if self.node_positions.is_empty() {
            return;
        }
        let bounds = (100.0, 100.0, 700.0, 500.0);
        let layout = force_directed_layout(&self.graph, bounds, 8);
        self.apply_layout(&layout);
    }

    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut primitives = Vec::with_capacity(self.edges.len() + 2);

        for (idx, &(u, v, _)) in self.edges.iter().enumerate() {
            if let (Some(&pos_u), Some(&pos_v)) =
                (self.node_positions.get(&u), self.node_positions.get(&v))
            {
                primitives.push(ExportPrimitive::Polyline {
                    name: format!("edge_{idx}"),
                    points: vec![
                        [f64::from(pos_u.x), -f64::from(pos_u.y)],
                        [f64::from(pos_v.x), -f64::from(pos_v.y)],
                    ],
                    color: palette.graph_edge,
                    stroke_width: 2.0,
                    closed: false,
                    fill: None,
                });
            }
        }

        let mut regular_nodes = Vec::new();
        let mut selected_nodes = Vec::new();
        for (&id, &pos) in &self.node_positions {
            let pt = [f64::from(pos.x), -f64::from(pos.y)];
            if self.selected_node == Some(id) {
                selected_nodes.push(pt);
            } else {
                regular_nodes.push(pt);
            }
        }

        if !regular_nodes.is_empty() {
            primitives.push(ExportPrimitive::Points {
                name: "graph_nodes".to_string(),
                points: regular_nodes,
                color: palette.series_color(0),
                radius: 8.0,
            });
        }
        if !selected_nodes.is_empty() {
            primitives.push(ExportPrimitive::Points {
                name: "selected_node".to_string(),
                points: selected_nodes,
                color: palette.success,
                radius: 9.0,
            });
        }

        primitives
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let mut datasets = Vec::new();
        if !self.node_positions.is_empty() {
            let nodes = self
                .node_positions
                .values()
                .map(|pos| [f64::from(pos.x), f64::from(pos.y)])
                .collect();
            datasets.push(("graph_nodes".to_string(), nodes));
        }
        if !self.edges.is_empty() {
            let edges = self
                .edges
                .iter()
                .map(|&(u, v, _)| [u as f64, v as f64])
                .collect();
            datasets.push(("graph_edges".to_string(), edges));
        }
        datasets
    }
}
