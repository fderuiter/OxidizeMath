#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use egui_plot::{ExportPrimitive, SharedPoints};
use math_explorer::epidemiology::networks::NodeState;

use crate::app::ThemeMode;
use crate::framework::{ExportCell, ExportTable};

use super::presets::NetPropState;
use super::NetworkPropagationTool;

const fn state_name(state: NodeState) -> &'static str {
    match state {
        NodeState::Susceptible => "Susceptible",
        NodeState::Infected => "Infected",
        NodeState::Recovered => "Recovered",
        NodeState::Vaccinated => "Vaccinated",
    }
}

impl NetworkPropagationTool {
    pub(super) fn serialize_net_prop_state(&self) -> Option<String> {
        let (num_nodes, vaccinated_nodes, quarantined_edges, frozen_graph) =
            if self.last_upstream_graph_rev != 0 {
                (
                    self.model.num_nodes().clamp(10, 200),
                    Vec::new(),
                    Vec::new(),
                    None,
                )
            } else if self.synced_bus_revision.is_some() {
                (self.local_num_nodes, Vec::new(), Vec::new(), None)
            } else {
                let frozen = (!self.follow_workspace_graph)
                    .then(|| self.active_shared_graph.as_deref().cloned())
                    .flatten();
                let nodes = if self.active_shared_graph.is_some()
                    && self.seeded_graph.is_none()
                    && frozen.is_none()
                {
                    self.local_num_nodes
                } else {
                    self.model.num_nodes()
                };
                let (vax, quar) = if self.active_shared_graph.is_some()
                    && self.seeded_graph.is_none()
                    && frozen.is_none()
                {
                    (Vec::new(), Vec::new())
                } else {
                    (
                        self.vaccinated_nodes.clone(),
                        self.quarantined_edges.clone(),
                    )
                };
                (nodes, vax, quar, frozen)
            };
        serde_json::to_string(&NetPropState {
            num_nodes,
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            seed: self.seed,
            budget: self.budget,
            vaccinated_nodes,
            quarantined_edges,
            is_running: self.is_running,
            topology_source: self.topology_source,
            follow_workspace_graph: self.follow_workspace_graph,
            seeded_graph: self.seeded_graph.clone(),
            frozen_graph,
        })
        .ok()
    }

    pub(super) fn build_epi_plot_primitives(&self) -> Vec<ExportPrimitive> {
        let palette = ThemeMode::Dark.palette();
        let positions = self.model.positions();
        let states = self.model.states();
        let edges = self.model.edges();
        let mut prims = Vec::with_capacity(edges.len() + 4);

        for (edge_idx, &[u, v]) in edges.iter().enumerate() {
            if u >= positions.len() || v >= positions.len() {
                continue;
            }
            let p0 = [f64::from(positions[u][0]), -f64::from(positions[u][1])];
            let p1 = [f64::from(positions[v][0]), -f64::from(positions[v][1])];
            let quarantined = self.quarantined_edges.binary_search(&edge_idx).is_ok();
            let (color, width) = if quarantined {
                (palette.warning, 2.0)
            } else {
                (palette.graph_edge, 1.2)
            };
            prims.push(ExportPrimitive::Polyline {
                name: if quarantined {
                    "Quarantined Edge".to_string()
                } else {
                    "Contact Edge".to_string()
                },
                points: SharedPoints::from(vec![p0, p1]),
                color,
                stroke_width: width,
                closed: false,
                fill: None,
            });
        }

        let groups = [
            (NodeState::Susceptible, "Susceptible", palette.axis_z, 5.5),
            (NodeState::Infected, "Infected", palette.axis_x, 6.5),
            (NodeState::Recovered, "Recovered", palette.axis_y, 5.5),
            (NodeState::Vaccinated, "Vaccinated", palette.accent, 6.0),
        ];
        for (target_state, label, color, radius) in groups {
            let pts: Vec<[f64; 2]> = states
                .iter()
                .enumerate()
                .filter(|&(_, &st)| st == target_state)
                .filter_map(|(idx, _)| positions.get(idx))
                .map(|p| [f64::from(p[0]), -f64::from(p[1])])
                .collect();
            if !pts.is_empty() {
                prims.push(ExportPrimitive::Points {
                    name: label.to_string(),
                    points: SharedPoints::from(pts),
                    color,
                    radius,
                });
            }
        }
        prims
    }

    fn build_epi_nodes_table(&self) -> ExportTable {
        let positions = self.model.positions();
        let node_rows = self
            .model
            .states()
            .iter()
            .enumerate()
            .map(|(idx, &st)| {
                let pos = positions.get(idx).copied().unwrap_or([0.0, 0.0]);
                vec![
                    ExportCell::Int(idx as i64),
                    ExportCell::Float(f64::from(pos[0])),
                    ExportCell::Float(f64::from(pos[1])),
                    ExportCell::Text(state_name(st).to_string()),
                    ExportCell::Bool(st == NodeState::Vaccinated),
                ]
            })
            .collect();
        ExportTable::new(
            "Network Nodes",
            vec![
                "node_id".into(),
                "x".into(),
                "y".into(),
                "state".into(),
                "vaccinated".into(),
            ],
            node_rows,
        )
    }

    pub(super) fn build_epi_export_tables(&self) -> Vec<ExportTable> {
        let mut tables = vec![self.build_epi_nodes_table()];
        let edge_rows: Vec<Vec<ExportCell>> = self
            .model
            .edges()
            .iter()
            .enumerate()
            .map(|(idx, &[u, v])| {
                let quar = self.quarantined_edges.binary_search(&idx).is_ok();
                vec![
                    ExportCell::Int(idx as i64),
                    ExportCell::Int(u as i64),
                    ExportCell::Int(v as i64),
                    ExportCell::Bool(quar),
                ]
            })
            .collect();
        if !edge_rows.is_empty() {
            tables.push(ExportTable::new(
                "Contact Edges",
                vec![
                    "edge_id".into(),
                    "source".into(),
                    "target".into(),
                    "quarantined".into(),
                ],
                edge_rows,
            ));
        }
        let trace_len = self
            .trace_s
            .len()
            .min(self.trace_i.len())
            .min(self.trace_r.len());
        if trace_len > 0 {
            let trace_rows = (0..trace_len)
                .map(|k| {
                    vec![
                        ExportCell::Float(self.trace_s[k][0]),
                        ExportCell::Float(self.trace_s[k][1]),
                        ExportCell::Float(self.trace_i[k][1]),
                        ExportCell::Float(self.trace_r[k][1]),
                    ]
                })
                .collect();
            tables.push(ExportTable::new(
                "SIR Trace",
                vec![
                    "step".into(),
                    "susceptible".into(),
                    "infected".into(),
                    "recovered".into(),
                ],
                trace_rows,
            ));
        }
        tables
    }
}
