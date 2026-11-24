//! Live `GraphTopologyPayload` ingestion for `NetworkPropagationTool` via `NetworkEpidemicModel::builder()`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use crate::app::dataflow_bus::{poll_pane_graph, GraphTopologyPayload, MAX_DATAFLOW_GRAPH_NODES};
use eframe::egui;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};

fn build_symmetric_adjacency(n: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<usize>> {
    let mut adj = vec![Vec::new(); n];
    for &(u, v, w) in edges {
        if u < n && v < n && u != v && w > 0.0 {
            adj[u].push(v);
            adj[v].push(u);
        }
    }
    for nbrs in &mut adj {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adj
}

impl NetworkPropagationTool {
    /// Polls `PaneDataflowBus` and rebuilds the contact graph only when `revision` advances.
    pub(super) fn poll_upstream_graph(&mut self, ctx: &egui::Context) {
        if self.is_challenge_locked() {
            self.upstream_linked = false;
            return;
        }
        let Some(payload) = poll_pane_graph(ctx) else {
            if self.upstream_linked {
                self.upstream_linked = false;
                self.last_upstream_graph_rev = 0;
                let _ = self.model.set_num_nodes(self.local_num_nodes);
                self.vaccinated_nodes.clear();
                self.quarantined_edges.clear();
                self.rebuild_seeded_graph();
            }
            return;
        };
        self.upstream_linked = true;
        if payload.revision != self.last_upstream_graph_rev {
            let _ = self.ingest_graph_topology(&payload);
        }
    }

    /// Rebuilds [`NetworkEpidemicModel`] from a live [`GraphTopologyPayload`] via `builder()`.
    pub fn ingest_graph_topology(&mut self, payload: &GraphTopologyPayload) -> Result<(), String> {
        let n = payload.node_positions.len().min(MAX_DATAFLOW_GRAPH_NODES);
        if n == 0 {
            self.last_upstream_graph_rev = payload.revision;
            return Ok(());
        }
        let positions = payload.normalized_positions(n, 165.0);
        let adjacency = build_symmetric_adjacency(n, &payload.edges);

        let prev_quarantined_pairs: Vec<[usize; 2]> = self
            .quarantined_edges
            .iter()
            .filter_map(|&idx| self.model.edges().get(idx).copied())
            .collect();
        let states = self.seed_states_for_topology(n, &adjacency);
        let (next_model, unmitigated_twin) =
            self.build_topology_models(n, positions, adjacency, states)?;

        self.apply_rebuilt_topology_model(
            next_model,
            unmitigated_twin,
            &prev_quarantined_pairs,
            payload.revision,
        );
        Ok(())
    }

    fn build_topology_models(
        &self,
        n: usize,
        positions: Vec<[f32; 2]>,
        adjacency: Vec<Vec<usize>>,
        states: Vec<NodeState>,
    ) -> Result<(NetworkEpidemicModel, NetworkEpidemicModel), String> {
        let unmitigated_states: Vec<NodeState> = states
            .iter()
            .map(|&st| {
                if st == NodeState::Vaccinated {
                    NodeState::Susceptible
                } else {
                    st
                }
            })
            .collect();
        let unmitigated_twin = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(self.model.beta(), self.model.gamma())
            .states(unmitigated_states)
            .positions(positions.clone())
            .adjacency(adjacency.clone())
            .build()
            .map_err(|e| e.to_string())?;
        let next_model = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(self.model.beta(), self.model.gamma())
            .states(states)
            .positions(positions)
            .adjacency(adjacency)
            .build()
            .map_err(|e| e.to_string())?;
        Ok((next_model, unmitigated_twin))
    }

    fn seed_states_for_topology(&self, n: usize, adjacency: &[Vec<usize>]) -> Vec<NodeState> {
        let prev_states = self.model.states();
        let mut states = vec![NodeState::Susceptible; n];
        if self.last_upstream_graph_rev != 0 {
            for (i, slot) in states.iter_mut().enumerate() {
                if let Some(&st) = prev_states.get(i) {
                    *slot = st;
                }
            }
        }
        if !states.contains(&NodeState::Infected) {
            let seed_node = (0..n).max_by_key(|&i| adjacency[i].len()).unwrap_or(0);
            if states[seed_node] == NodeState::Susceptible {
                states[seed_node] = NodeState::Infected;
            } else if let Some(first_s) = states.iter().position(|&s| s == NodeState::Susceptible) {
                states[first_s] = NodeState::Infected;
            }
        }
        states
    }

    fn apply_rebuilt_topology_model(
        &mut self,
        mut next_model: NetworkEpidemicModel,
        unmitigated_twin: NetworkEpidemicModel,
        prev_quarantined_pairs: &[[usize; 2]],
        revision: u64,
    ) {
        self.vaccinated_nodes.clear();
        for (idx, &st) in next_model.states().iter().enumerate() {
            if st == NodeState::Vaccinated {
                self.vaccinated_nodes.push(idx);
            }
        }
        self.quarantined_edges.clear();
        let new_edges = next_model.edges().to_vec();
        for &pair in prev_quarantined_pairs {
            if let Ok(edge_idx) = new_edges.binary_search(&pair) {
                if next_model
                    .set_edge_quarantined(pair[0], pair[1], true)
                    .is_ok()
                {
                    self.quarantined_edges.push(edge_idx);
                }
            }
        }
        self.quarantined_edges.sort_unstable();
        self.quarantined_edges.dedup();
        self.baseline_model = unmitigated_twin;
        self.model = next_model;
        self.enforce_budget_cap();
        self.clamp_focus_indices();
        self.synced_bus_revision = None;
        self.last_upstream_graph_rev = revision;
        self.bump_revision();
    }

    #[cfg(test)]
    pub(crate) fn sim_steps(&self) -> u64 {
        self.sim_steps
    }

    #[cfg(test)]
    pub(super) fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.canvas_id
    }

    #[cfg(all(test, feature = "physics", feature = "biology"))]
    pub(crate) fn current_effective_r0(&self) -> f64 {
        super::compute_network_analytics(&self.model).effective_r0
    }
}
