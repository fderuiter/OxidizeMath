//! Workspace data bus synchronization, state validation, and empirical trace streaming for [`super::NetworkPropagationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NetPropState, NetworkPropagationTool, MAX_CONTAINMENT_BUDGET};
use crate::app::data_bus::ui::sync_consumer_follow_flag;
use crate::app::data_bus::{
    adjacency_degree_moments, publish_epidemic_trace, read_shared_graph,
    seed_shared_graph_if_empty, EpidemicTraceSignal, SharedGraph, MAX_EPIDEMIC_NODE_BUDGET,
    SESSION_MAX_BUS_EDGES, SESSION_MAX_BUS_NODES, SESSION_MAX_TRACE_POINTS,
};
use eframe::egui;
use math_explorer::epidemiology::networks::{
    heterogeneous_r0, NetworkEpidemicModel, NetworkEpidemicModelBuilder, NodeState,
};
use std::collections::HashSet;
use std::sync::Arc;

fn validate_unique_indices(items: &[usize], label: &str) -> Result<Vec<usize>, String> {
    let mut seen = HashSet::with_capacity(items.len());
    for &idx in items {
        if !seen.insert(idx) {
            return Err(format!("Duplicate index {idx} in {label}"));
        }
    }
    let mut sorted = items.to_vec();
    sorted.sort_unstable();
    Ok(sorted)
}

fn build_model_from_shared_graph(
    clamped: &SharedGraph,
    beta: f64,
    gamma: f64,
) -> Result<NetworkEpidemicModel, String> {
    let n = clamped.nodes.len();
    if n == 0 {
        return NetworkEpidemicModelBuilder::new()
            .num_nodes(1)
            .parameters(beta, gamma)
            .states(vec![NodeState::Susceptible])
            .positions(vec![[0.0, 0.0]])
            .adjacency(vec![Vec::new()])
            .build()
            .map_err(|e| e.to_string());
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
    NetworkEpidemicModelBuilder::new()
        .num_nodes(n)
        .parameters(beta, gamma)
        .states(states)
        .positions(positions)
        .adjacency(adj)
        .build()
        .map_err(|e| e.to_string())
}

fn apply_validated_interventions(
    candidate: &mut NetworkEpidemicModel,
    vax_sorted: &[usize],
    edge_sorted: &[usize],
) -> Result<(), String> {
    for &node in vax_sorted {
        if node >= candidate.num_nodes() {
            return Err(format!("Vaccinated node index {node} out of bounds"));
        }
        if candidate.patient_zero() == Some(node)
            || candidate.states()[node] != NodeState::Susceptible
        {
            return Err(format!("Cannot vaccinate patient zero (Node {node})"));
        }
        candidate
            .set_node_vaccinated(node, true)
            .map_err(|e| e.to_string())?;
    }
    for &edge_idx in edge_sorted {
        let &[u, v] = candidate
            .edges()
            .get(edge_idx)
            .ok_or_else(|| format!("Quarantined edge index {edge_idx} out of bounds"))?;
        candidate
            .set_edge_quarantined(u, v, true)
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

impl NetworkPropagationTool {
    /// Computes structural degree moments $(\langle k \rangle, \operatorname{Var}(k))$ and $R_0^{\text{het}}$.
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
            .take(self.model.num_nodes())
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

        if let Ok(built) =
            build_model_from_shared_graph(&clamped, self.model.beta(), self.model.gamma())
        {
            self.model = built;
            self.vaccinated_nodes.clear();
            self.quarantined_edges.clear();
            self.clamp_focus_indices();
            self.rng = oxidize_core::rng::OxidizeRng::new(self.seed);
            self.reset_empirical_trace();
            self.bump_revision();
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

    pub(super) fn load_validated_state_json(&mut self, state: &str) -> Result<(), String> {
        let s: NetPropState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid network propagation state: {e}"))?;
        let (vax_sorted, edge_sorted) = validate_net_prop_bounds(&s)?;

        let mut candidate_rng = oxidize_core::rng::OxidizeRng::new(s.seed);
        let graph_opt = s.frozen_graph.or_else(|| s.seeded_graph.clone());
        let (mut candidate, active_shared, clamped_orig, applied_rev) =
            if let Some(graph) = graph_opt {
                let shared = Arc::new(graph);
                let (clamped, orig) = shared.clamped_to_budget(MAX_EPIDEMIC_NODE_BUDGET);
                let built = build_model_from_shared_graph(&clamped, s.beta, s.gamma)?;
                (built, Some(clamped), orig, shared.revision)
            } else {
                let built = NetworkEpidemicModel::new_with_rng(
                    s.num_nodes,
                    s.beta,
                    s.gamma,
                    &mut candidate_rng,
                );
                (built, None, None, 0)
            };
        apply_validated_interventions(&mut candidate, &vax_sorted, &edge_sorted)?;

        self.model = candidate;
        self.rng = candidate_rng;
        self.seed = s.seed;
        self.budget = s.budget;
        self.vaccinated_nodes = vax_sorted;
        self.quarantined_edges = edge_sorted;
        self.is_running = s.is_running;
        self.follow_workspace_graph = s.follow_workspace_graph;
        self.seeded_graph = s.seeded_graph;
        self.active_shared_graph = active_shared;
        self.clamped_source_nodes = clamped_orig;
        self.applied_bus_revision = applied_rev;
        self.pending_seed_publish = self.seeded_graph.is_some();
        self.clamp_focus_indices();
        self.reset_empirical_trace();
        #[cfg(test)]
        {
            self.sim_steps = 0;
        }
        self.bump_revision();
        Ok(())
    }
}

fn validate_net_prop_bounds(s: &NetPropState) -> Result<(Vec<usize>, Vec<usize>), String> {
    let has_graph = s.seeded_graph.is_some() || s.frozen_graph.is_some();
    let nodes_ok = if has_graph {
        (1..=MAX_EPIDEMIC_NODE_BUDGET).contains(&s.num_nodes)
    } else {
        (10..=200).contains(&s.num_nodes)
    };
    if !nodes_ok
        || !s.beta.is_finite()
        || !(0.0..=1.0).contains(&s.beta)
        || !s.gamma.is_finite()
        || !(0.0..=1.0).contains(&s.gamma)
        || s.budget > MAX_CONTAINMENT_BUDGET
    {
        return Err("Network propagation parameters out of bounds".into());
    }
    if s.vaccinated_nodes
        .len()
        .saturating_add(s.quarantined_edges.len())
        > s.budget
    {
        return Err("Containment interventions exceed configured budget".into());
    }
    if let Some(ref g) = s.seeded_graph {
        g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
    }
    if let Some(ref g) = s.frozen_graph {
        g.validate(SESSION_MAX_BUS_NODES, SESSION_MAX_BUS_EDGES)?;
    }
    let vax_sorted = validate_unique_indices(&s.vaccinated_nodes, "vaccinated_nodes")?;
    let edge_sorted = validate_unique_indices(&s.quarantined_edges, "quarantined_edges")?;
    Ok((vax_sorted, edge_sorted))
}
