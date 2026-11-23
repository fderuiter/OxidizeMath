#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NetworkPropagationTool, DEFAULT_NETWORK_SEED};
use crate::app::workspace_bus::GraphTopologyChannel;
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};

fn initial_seeded_states(num_nodes: usize) -> Vec<NodeState> {
    let mut states = vec![NodeState::Susceptible; num_nodes];
    if !states.is_empty() {
        states[0] = NodeState::Infected;
    }
    states
}

fn build_channel_adjacency(n: usize, edges: &[(usize, usize, f64)]) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); n];
    for &(u, v, _) in edges {
        if u < n && v < n && u != v {
            adjacency[u].push(v);
            adjacency[v].push(u);
        }
    }
    for nbrs in &mut adjacency {
        nbrs.sort_unstable();
        nbrs.dedup();
    }
    adjacency
}

impl NetworkPropagationTool {
    /// Computes live residual heterogeneous reproduction number $R_0$ for the active network.
    #[cfg(test)]
    #[must_use]
    pub fn current_heterogeneous_r0(&self) -> f64 {
        super::compute_network_analytics(&self.model).effective_r0
    }

    pub(crate) fn ingest_topology_channel(&mut self, channel: &GraphTopologyChannel) {
        let rev = channel.revision();
        if rev == 0 || self.synced_bus_revision == Some(rev) {
            return;
        }
        self.synced_bus_revision = Some(rev);
        if channel.positions().is_empty() {
            return;
        }
        let n = channel.positions().len();
        let positions: Vec<[f32; 2]> = channel
            .positions()
            .iter()
            .map(|&[nx, ny]| [nx * 165.0, ny * 165.0])
            .collect();
        let adjacency = build_channel_adjacency(n, channel.edges());
        let states = if self.using_bus_topology {
            (0..n)
                .map(|i| {
                    self.model
                        .states()
                        .get(i)
                        .copied()
                        .unwrap_or(NodeState::Susceptible)
                })
                .collect()
        } else {
            initial_seeded_states(n)
        };
        if let Ok(mut built) = NetworkEpidemicModel::builder()
            .num_nodes(n)
            .parameters(self.model.beta(), self.model.gamma())
            .states(states)
            .positions(positions)
            .adjacency(adjacency)
            .build()
        {
            self.reconcile_bus_interventions(&mut built);
            self.model = built;
            self.using_bus_topology = true;
            self.clamp_focus_indices();
            self.bump_revision();
        }
    }

    fn reconcile_bus_interventions(&mut self, built: &mut NetworkEpidemicModel) {
        if self.using_bus_topology {
            self.vaccinated_nodes
                .retain(|&node| built.set_node_vaccinated(node, true).is_ok());
            let edges = built.edges().to_vec();
            self.quarantined_edges.retain(|&edge_idx| {
                edges
                    .get(edge_idx)
                    .is_some_and(|&[u, v]| built.set_edge_quarantined(u, v, true).is_ok())
            });
        } else {
            self.vaccinated_nodes.clear();
            self.quarantined_edges.clear();
        }
    }

    pub(super) fn reset_bus_topology_states(&mut self) {
        self.rng = oxidize_core::rng::OxidizeRng::new(self.seed);
        let edges = self.model.edges().to_vec();
        for &[u, v] in &edges {
            let _ = self.model.set_edge_quarantined(u, v, false);
        }
        self.vaccinated_nodes.clear();
        self.quarantined_edges.clear();
        let n = self.model.num_nodes();
        *self.model.states_mut() = initial_seeded_states(n);
        self.is_running = false;
        self.clamp_focus_indices();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
    }

    pub(super) fn on_bus_unlinked(&mut self) {
        if self.using_bus_topology {
            self.using_bus_topology = false;
            self.synced_bus_revision = None;
            self.vaccinated_nodes.clear();
            self.quarantined_edges.clear();
            let _ = self.model.set_num_nodes(self.standalone_num_nodes);
            if self.seed == 0 {
                self.seed = DEFAULT_NETWORK_SEED;
            }
            self.rebuild_seeded_graph();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_topology_ingestion_r0_and_unlink_restore() {
        let mut tool = NetworkPropagationTool::default();
        let _ = tool.model.set_beta(0.5);
        let _ = tool.model.set_gamma(0.1);

        let mut ch = GraphTopologyChannel::default();
        ch.publish(
            &[[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            &[(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0)],
        );
        tool.ingest_topology_channel(&ch);
        assert!(tool.using_bus_topology);
        assert_eq!(tool.model.num_nodes(), 4);
        let r0_star = tool.current_heterogeneous_r0();
        assert!(
            r0_star > 4.0,
            "Star hub should produce high R0, got {r0_star}"
        );

        // Quarantining an edge on the linked star graph reduces residual R0
        assert_eq!(tool.toggle_edge_quarantine(0), Ok(true));
        assert!(tool.current_heterogeneous_r0() < r0_star);

        // Cutting all edges drops residual R0 to 0.0
        ch.publish(&[[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], &[]);
        tool.ingest_topology_channel(&ch);
        assert_eq!(tool.current_heterogeneous_r0(), 0.0);

        // Unlinking restores standalone 50-node seeded geometric graph
        tool.on_bus_unlinked();
        assert!(!tool.using_bus_topology);
        assert_eq!(tool.model.num_nodes(), 50);
    }
}
