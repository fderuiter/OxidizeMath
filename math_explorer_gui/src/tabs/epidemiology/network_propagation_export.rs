#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use crate::app::SemanticPalette;
use crate::framework::export_table::{ExportCell, ExportTable};
use eframe::egui;
use egui_plot::export_primitives::{
    build_graph_primitives, ExportPrimitive, GraphEdgeSpec, GraphNodeSpec,
};
use math_explorer::epidemiology::networks::NodeState;

fn state_label(state: NodeState) -> &'static str {
    match state {
        NodeState::Susceptible => "Susceptible",
        NodeState::Infected => "Infected",
        NodeState::Recovered => "Recovered",
    }
}

impl NetworkPropagationTool {
    pub(super) fn build_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = SemanticPalette::from_visuals(&egui::Visuals::dark());
        let n = self.model.num_nodes();
        let positions = self.model.positions();
        let states = self.model.states();
        let adj = self.model.adjacency();

        let nodes: Vec<GraphNodeSpec> = (0..n)
            .map(|i| {
                let fill = match states[i] {
                    NodeState::Susceptible => palette.axis_z,
                    NodeState::Infected => palette.axis_x,
                    NodeState::Recovered => palette.axis_y,
                };
                GraphNodeSpec {
                    id: i,
                    pos: [f64::from(positions[i][0]), f64::from(positions[i][1])],
                    fill,
                    stroke_color: palette.node_stroke,
                    radius: 6.0,
                    label: format!("Node {i}"),
                }
            })
            .collect();

        let mut edges = Vec::new();
        for (i, neighbors) in adj.iter().enumerate().take(n) {
            for &j in neighbors {
                if i < j && j < n {
                    edges.push(GraphEdgeSpec {
                        from_id: i,
                        to_id: j,
                        color: palette.graph_edge,
                        width: 1.2,
                        directed: false,
                        label: format!("{i}-{j}"),
                    });
                }
            }
        }

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
                    ExportCell::UInt(adj[i].len() as u64),
                ]
            })
            .collect();

        let mut edge_rows = Vec::new();
        for (i, neighbors) in adj.iter().enumerate().take(n) {
            for &j in neighbors {
                if i < j && j < n {
                    edge_rows.push(vec![
                        ExportCell::UInt(i as u64),
                        ExportCell::UInt(j as u64),
                    ]);
                }
            }
        }

        let mut tables = vec![ExportTable::new(
            "Nodes",
            vec![
                "node_id".into(),
                "x".into(),
                "y".into(),
                "state".into(),
                "degree".into(),
            ],
            node_rows,
        )];
        if !edge_rows.is_empty() {
            tables.push(ExportTable::new(
                "Contact Edges",
                vec!["source".into(), "target".into()],
                edge_rows,
            ));
        }
        tables
    }
}
