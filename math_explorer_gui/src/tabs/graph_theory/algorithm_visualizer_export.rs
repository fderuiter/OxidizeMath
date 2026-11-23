#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::AlgorithmVisualizerTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::{
    build_graph_primitives, ExportPrimitive, GraphEdgeSpec, GraphNodeSpec,
};

impl AlgorithmVisualizerTool {
    pub(super) fn step_export_simulation(&mut self) {
        if self.visit_order.is_empty() {
            self.run_algorithm();
        }
        if !self.visit_order.is_empty() {
            self.animation_step = (self.animation_step + 1) % (self.visit_order.len() + 1);
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let mut sorted_ids: Vec<usize> = self.node_positions.keys().copied().collect();
        sorted_ids.sort_unstable();

        let upto = self.animation_step.min(self.visit_order.len());
        let visited = &self.visit_order[..upto];
        let active = visited.last().copied();

        let nodes: Vec<GraphNodeSpec> = sorted_ids
            .into_iter()
            .map(|id| {
                let pos = self.node_positions[&id];
                let fill = if active == Some(id) {
                    palette.warning
                } else if self.start_node == Some(id) {
                    palette.success
                } else if visited.contains(&id) {
                    palette.accent
                } else {
                    palette.info
                };
                GraphNodeSpec {
                    id,
                    pos: [f64::from(pos.x), f64::from(pos.y)],
                    fill,
                    stroke_color: palette.node_stroke,
                    radius: 10.0,
                    label: format!("Node {id}"),
                }
            })
            .collect();

        let edges: Vec<GraphEdgeSpec> = self
            .edges
            .iter()
            .filter(|&&(u, v, _)| {
                self.node_positions.contains_key(&u) && self.node_positions.contains_key(&v)
            })
            .map(|&(u, v, w)| {
                let both_visited = visited.contains(&u) && visited.contains(&v);
                GraphEdgeSpec {
                    from_id: u,
                    to_id: v,
                    color: if both_visited {
                        palette.accent
                    } else {
                        palette.graph_edge
                    },
                    width: if both_visited { 2.4 } else { 1.5 },
                    directed: false,
                    label: format!("{u}-{v} ({w:.0})"),
                }
            })
            .collect();

        build_graph_primitives(&nodes, &edges, true)
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let traversal_rows = self.build_traversal_rows();
        let edge_rows: Vec<Vec<ExportCell>> = self
            .edges
            .iter()
            .map(|&(u, v, w)| {
                vec![
                    ExportCell::UInt(u as u64),
                    ExportCell::UInt(v as u64),
                    ExportCell::Float(w),
                ]
            })
            .collect();

        vec![
            ExportTable::new(
                "Traversal Order",
                vec![
                    "visit_rank".into(),
                    "node_id".into(),
                    "distance".into(),
                    "visited_at_step".into(),
                ],
                traversal_rows,
            ),
            ExportTable::new(
                "Graph Edges",
                vec!["source".into(), "target".into(), "weight".into()],
                edge_rows,
            ),
        ]
    }

    fn build_traversal_rows(&self) -> Vec<Vec<ExportCell>> {
        let upto = self.animation_step.min(self.visit_order.len());
        if !self.visit_order.is_empty() {
            return self
                .visit_order
                .iter()
                .enumerate()
                .map(|(rank, &node_id)| {
                    let dist = self
                        .distances
                        .get(&node_id)
                        .copied()
                        .unwrap_or(rank as f64);
                    vec![
                        ExportCell::UInt(rank as u64),
                        ExportCell::UInt(node_id as u64),
                        ExportCell::Float(dist),
                        ExportCell::Bool(rank < upto),
                    ]
                })
                .collect();
        }
        let mut ids: Vec<usize> = self.node_positions.keys().copied().collect();
        ids.sort_unstable();
        ids.into_iter()
            .enumerate()
            .map(|(rank, node_id)| {
                vec![
                    ExportCell::UInt(rank as u64),
                    ExportCell::UInt(node_id as u64),
                    ExportCell::Float(0.0),
                    ExportCell::Bool(false),
                ]
            })
            .collect()
    }
}
