//! Workspace data bus synchronization and empirical trace streaming for [`super::NetworkPropagationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::NetworkPropagationTool;
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    adjacency_degree_moments, publish_epidemic_trace, read_shared_graph,
    seed_shared_graph_if_empty, EpidemicTraceSignal, SharedGraph, MAX_EPIDEMIC_NODE_BUDGET,
    SESSION_MAX_TRACE_POINTS,
};
use eframe::egui;
use math_explorer::epidemiology::networks::{
    heterogeneous_r0, NetworkEpidemicModelBuilder, NodeState,
};
use std::sync::Arc;

impl NetworkPropagationTool {
    /// Computes analytical degree moments $(\langle k \rangle, \operatorname{Var}(k))$ and $R_0^{\text{het}}$.
    #[must_use]
    pub fn degree_moments_and_r0(&self) -> (f64, f64, f64) {
        let (mean_k, var_k) = adjacency_degree_moments(self.model.adjacency());
        let r0_het = heterogeneous_r0(self.model.beta(), self.model.gamma(), mean_k, var_k);
        (mean_k, var_k, r0_het)
    }

    pub(super) fn count_compartment(&self, target: NodeState) -> usize {
        self.model
            .states()
            .iter()
            .filter(|&&s| s == target)
            .count()
    }

    pub(super) fn reset_empirical_trace(&mut self) {
        self.trace_step = 0;
        self.trace_s.clear();
        self.trace_i.clear();
        self.trace_r.clear();
        self.record_trace_sample();
    }

    pub(super) fn record_trace_sample(&mut self) {
        if self.trace_s.len() >= SESSION_MAX_TRACE_POINTS {
            self.trace_s.remove(0);
            self.trace_i.remove(0);
            self.trace_r.remove(0);
        }
        let t = self.trace_step as f64;
        self.trace_s
            .push([t, self.count_compartment(NodeState::Susceptible) as f64]);
        self.trace_i
            .push([t, self.count_compartment(NodeState::Infected) as f64]);
        self.trace_r
            .push([t, self.count_compartment(NodeState::Recovered) as f64]);
        self.trace_dirty = true;
    }

    pub(super) fn flush_trace_to_bus(&mut self, ctx: &egui::Context) {
        if !self.trace_dirty {
            return;
        }
        let (_, _, r0_het) = self.degree_moments_and_r0();
        let signal = EpidemicTraceSignal {
            revision: 1,
            population: self.model.num_nodes().max(1),
            beta: self.model.beta(),
            gamma: self.model.gamma(),
            heterogeneous_r0: r0_het,
            susceptible: self.trace_s.clone(),
            infected: self.trace_i.clone(),
            recovered: self.trace_r.clone(),
        };
        let _ = publish_epidemic_trace(ctx, signal);
        self.trace_dirty = false;
    }

    pub(crate) fn apply_shared_graph(&mut self, shared: &Arc<SharedGraph>) {
        let (clamped, orig_count) = shared.clamped_to_budget(MAX_EPIDEMIC_NODE_BUDGET);
        self.clamped_source_nodes = orig_count;
        self.applied_bus_revision = shared.revision;
        self.active_shared_graph = Some(Arc::clone(&clamped));

        let beta = self.model.beta();
        let gamma = self.model.gamma();
        let n = clamped.nodes.len();
        if n == 0 {
            if let Ok(empty_fallback) = NetworkEpidemicModelBuilder::new()
                .num_nodes(1)
                .parameters(beta, gamma)
                .states(vec![NodeState::Susceptible])
                .positions(vec![[0.0, 0.0]])
                .adjacency(vec![Vec::new()])
                .build()
            {
                self.model = empty_fallback;
            }
            self.reset_empirical_trace();
            return;
        }

        let adj = clamped.positive_adjacency();
        let hub_idx = adj
            .iter()
            .enumerate()
            .max_by_key(|(_, nb)| nb.len())
            .map_or(0, |(idx, _)| idx);
        let mut states = vec![NodeState::Susceptible; n];
        states[hub_idx] = NodeState::Infected;
        let positions: Vec<[f32; 2]> = clamped
            .nodes
            .iter()
            .map(|nd| {
                [
                    (nd.norm_pos[0].clamp(0.0, 1.0) - 0.5) * 260.0,
                    (nd.norm_pos[1].clamp(0.0, 1.0) - 0.5) * 220.0,
                ]
            })
            .collect();

        if let Ok(built) = NetworkEpidemicModelBuilder::new()
            .num_nodes(n)
            .parameters(beta, gamma)
            .states(states)
            .positions(positions)
            .adjacency(adj)
            .build()
        {
            self.model = built;
            self.rng = oxidize_core::rng::OxidizeRng::default();
            self.reset_empirical_trace();
        }
    }

    pub(super) fn sync_from_bus(&mut self, ctx: &egui::Context) {
        if sync_consumer_follow_flag(ctx, &mut self.follow_workspace_graph, "Network Propagation")
            && self.follow_workspace_graph
        {
            self.applied_bus_revision = 0;
        }
        if self.pending_seed_publish {
            self.pending_seed_publish = false;
            if let Some(ref seeded) = self.seeded_graph {
                let arc = seed_shared_graph_if_empty(ctx, seeded);
                if self.follow_workspace_graph {
                    self.apply_shared_graph(&arc);
                }
            }
        }
        if !self.follow_workspace_graph {
            return;
        }
        if let Some(shared) = read_shared_graph(ctx) {
            if shared.revision != self.applied_bus_revision {
                self.apply_shared_graph(&shared);
            }
        }
    }
}
