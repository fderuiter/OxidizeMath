#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::{
    build_graph_primitives, ExportPrimitive, GraphEdgeSpec, GraphNodeSpec,
};
use math_explorer::epidemiology::networks::NodeState;

const fn state_label(state: NodeState) -> &'static str {
    match state {
        NodeState::Susceptible => "Susceptible",
        NodeState::Infected => "Infected",
        NodeState::Recovered => "Recovered",
        NodeState::Vaccinated => "Vaccinated",
    }
}

impl NetworkPropagationTool {
    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let n = self.model.num_nodes();
        let positions = self.model.positions();
        let states = self.model.states();

        let nodes: Vec<GraphNodeSpec> = (0..n)
            .map(|i| {
                let (fill, radius) = match states[i] {
                    NodeState::Susceptible => (palette.axis_z, 5.5),
                    NodeState::Infected => (palette.axis_x, 6.7),
                    NodeState::Recovered => (palette.axis_y, 5.5),
                    NodeState::Vaccinated => (palette.accent, 6.5),
                };
                GraphNodeSpec {
                    id: i,
                    pos: [f64::from(positions[i][0]), f64::from(positions[i][1])],
                    fill,
                    stroke_color: palette.node_stroke,
                    radius,
                    label: format!("Node {i}"),
                }
            })
            .collect();

        let edges: Vec<GraphEdgeSpec> = self
            .model
            .edges()
            .iter()
            .filter(|&&[u, v]| u < n && v < n)
            .map(|&[u, v]| {
                let quarantined = self.model.is_edge_quarantined(u, v);
                GraphEdgeSpec {
                    from_id: u,
                    to_id: v,
                    color: if quarantined {
                        palette.warning
                    } else {
                        palette.graph_edge
                    },
                    width: if quarantined { 1.8 } else { 1.2 },
                    directed: false,
                    label: format!("{u}-{v}"),
                }
            })
            .collect();

        build_graph_primitives(&nodes, &edges, true)
    }

    pub(super) fn build_export_tables(&self) -> Vec<ExportTable> {
        let n = self.model.num_nodes();
        let positions = self.model.positions();
        let states = self.model.states();
        let adj = self.model.adjacency();

        let node_rows: Vec<Vec<ExportCell>> = (0..n)
            .map(|i| {
                vec![
                    ExportCell::UInt(i as u64),
                    ExportCell::Float(f64::from(positions[i][0])),
                    ExportCell::Float(f64::from(positions[i][1])),
                    ExportCell::Text(state_label(states[i]).into()),
                    ExportCell::UInt(self.model.active_degree(i) as u64),
                    ExportCell::UInt(adj[i].len() as u64),
                ]
            })
            .collect();

        let edge_rows: Vec<Vec<ExportCell>> = self
            .model
            .edges()
            .iter()
            .filter(|&&[u, v]| u < n && v < n)
            .map(|&[u, v]| {
                vec![
                    ExportCell::UInt(u as u64),
                    ExportCell::UInt(v as u64),
                    ExportCell::Bool(self.model.is_edge_quarantined(u, v)),
                ]
            })
            .collect();

        let mut tables = vec![ExportTable::new(
            "Nodes",
            vec![
                "node_id".into(),
                "x".into(),
                "y".into(),
                "state".into(),
                "active_degree".into(),
                "total_degree".into(),
            ],
            node_rows,
        )];
        if !edge_rows.is_empty() {
            tables.push(ExportTable::new(
                "Contact Edges",
                vec!["source".into(), "target".into(), "quarantined".into()],
                edge_rows,
            ));
        }
        tables
    }
}
