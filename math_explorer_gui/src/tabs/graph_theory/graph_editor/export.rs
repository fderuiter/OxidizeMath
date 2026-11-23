#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AffineLatency, GraphEditorTool};
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use crate::framework::ExportCapabilities;
use eframe::egui;
use egui_plot::export_primitives::{
    build_graph_primitives, ExportPrimitive, GraphEdgeSpec, GraphNodeSpec,
};

impl GraphEditorTool {
    pub(super) fn current_export_capabilities(&self) -> ExportCapabilities {
        if self.node_positions.is_empty() && !self.braess.config.enabled {
            ExportCapabilities::unpopulated(
                "Add nodes on the canvas or load a starter preset before exporting the graph.",
            )
        } else {
            ExportCapabilities::plot()
        }
    }

    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let nodes: Vec<GraphNodeSpec> = self
            .node_positions
            .iter()
            .map(|(&id, &pos)| {
                let fill = if self.selected_node == Some(id) {
                    palette.accent
                } else if self.braess.config.enabled && id == self.braess.config.source_node {
                    palette.success
                } else if self.braess.config.enabled && id == self.braess.config.sink_node {
                    palette.warning
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
            .enumerate()
            .filter(|(_, &(u, v, _))| {
                self.node_positions.contains_key(&u) && self.node_positions.contains_key(&v)
            })
            .map(|(idx, &(u, v, w))| {
                let flow = self
                    .braess
                    .equilibrium_flows
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0);
                let active_flow = self.braess.config.enabled && flow > 1e-6;
                let label = if self.braess.config.enabled {
                    format!("{u}->{v} (f={flow:.2})")
                } else {
                    format!("{u}-{v} ({w:.1})")
                };
                GraphEdgeSpec {
                    from_id: u,
                    to_id: v,
                    color: if active_flow {
                        palette.warning
                    } else {
                        palette.graph_edge
                    },
                    width: if active_flow { 2.6 } else { 1.6 },
                    directed: self.braess.config.enabled,
                    label,
                }
            })
            .collect();

        build_graph_primitives(&nodes, &edges, true)
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let node_rows: Vec<Vec<ExportCell>> = self
            .node_positions
            .iter()
            .map(|(&id, &pos)| {
                vec![
                    ExportCell::UInt(id as u64),
                    ExportCell::Float(f64::from(pos.x)),
                    ExportCell::Float(f64::from(pos.y)),
                    ExportCell::Bool(self.selected_node == Some(id)),
                ]
            })
            .collect();

        let mut tables = vec![ExportTable::new(
            "Nodes",
            vec!["node_id".into(), "x".into(), "y".into(), "selected".into()],
            node_rows,
        )];
        let edge_rows = self.build_edge_export_rows();
        if !edge_rows.is_empty() {
            tables.push(ExportTable::new(
                "Edges",
                [
                    "edge_index",
                    "source",
                    "target",
                    "weight",
                    "latency_a",
                    "latency_b",
                    "equilibrium_flow",
                    "equilibrium_latency",
                ]
                .into_iter()
                .map(String::from)
                .collect(),
                edge_rows,
            ));
        }
        tables
    }

    fn build_edge_export_rows(&self) -> Vec<Vec<ExportCell>> {
        self.edges
            .iter()
            .enumerate()
            .map(|(idx, &(u, v, w))| {
                let lat = self
                    .braess
                    .edge_latencies
                    .get(idx)
                    .copied()
                    .unwrap_or_else(|| AffineLatency::clamped(0.0, w));
                let flow = self
                    .braess
                    .equilibrium_flows
                    .get(idx)
                    .copied()
                    .unwrap_or(0.0);
                let eq_lat = self
                    .braess
                    .equilibrium_latencies
                    .get(idx)
                    .copied()
                    .unwrap_or_else(|| lat.eval(flow));
                vec![
                    ExportCell::UInt(idx as u64),
                    ExportCell::UInt(u as u64),
                    ExportCell::UInt(v as u64),
                    ExportCell::Float(w),
                    ExportCell::Float(lat.a),
                    ExportCell::Float(lat.b),
                    ExportCell::Float(flow),
                    ExportCell::Float(eq_lat),
                ]
            })
            .collect()
    }
}
