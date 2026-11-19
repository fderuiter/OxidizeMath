#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use egui_plot::export_primitives::ExportPrimitive;
use math_explorer::epidemiology::networks::NodeState;

impl NetworkPropagationTool {
    pub(super) fn cold_export_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = crate::app::palette::DARK_PALETTE;
        let mut primitives = Vec::new();
        let positions = self.model.positions();
        let adjacency = self.model.adjacency();
        let states = self.model.states();

        let mut edge_idx = 0usize;
        for (i, neighbors) in adjacency.iter().enumerate() {
            for &j in neighbors {
                if i < j {
                    primitives.push(ExportPrimitive::Polyline {
                        name: format!("contact_edge_{edge_idx}"),
                        points: vec![
                            [f64::from(positions[i][0]), -f64::from(positions[i][1])],
                            [f64::from(positions[j][0]), -f64::from(positions[j][1])],
                        ],
                        color: palette.graph_edge,
                        stroke_width: 1.2,
                        closed: false,
                        fill: None,
                    });
                    edge_idx += 1;
                }
            }
        }

        let mut susceptible = Vec::new();
        let mut infected = Vec::new();
        let mut recovered = Vec::new();
        for (i, &state) in states.iter().enumerate() {
            let pt = [f64::from(positions[i][0]), -f64::from(positions[i][1])];
            match state {
                NodeState::Susceptible => susceptible.push(pt),
                NodeState::Infected => infected.push(pt),
                NodeState::Recovered => recovered.push(pt),
            }
        }

        for (label, pts, color) in [
            ("susceptible_nodes", susceptible, palette.axis_z),
            ("infected_nodes", infected, palette.axis_x),
            ("recovered_nodes", recovered, palette.axis_y),
        ] {
            if !pts.is_empty() {
                primitives.push(ExportPrimitive::Points {
                    name: label.to_string(),
                    points: pts,
                    color,
                    radius: 5.5,
                });
            }
        }

        primitives
    }

    pub(super) fn cold_export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        let positions = self.model.positions();
        let states = self.model.states();
        let mut s_pts = Vec::new();
        let mut i_pts = Vec::new();
        let mut r_pts = Vec::new();

        for (idx, &state) in states.iter().enumerate() {
            let pt = [f64::from(positions[idx][0]), f64::from(positions[idx][1])];
            match state {
                NodeState::Susceptible => s_pts.push(pt),
                NodeState::Infected => i_pts.push(pt),
                NodeState::Recovered => r_pts.push(pt),
            }
        }

        let counts = vec![
            [0.0, s_pts.len() as f64],
            [1.0, i_pts.len() as f64],
            [2.0, r_pts.len() as f64],
        ];
        let mut datasets = vec![("compartment_counts".to_string(), counts)];
        if !s_pts.is_empty() {
            datasets.push(("susceptible_nodes".to_string(), s_pts));
        }
        if !i_pts.is_empty() {
            datasets.push(("infected_nodes".to_string(), i_pts));
        }
        if !r_pts.is_empty() {
            datasets.push(("recovered_nodes".to_string(), r_pts));
        }
        datasets
    }
}
