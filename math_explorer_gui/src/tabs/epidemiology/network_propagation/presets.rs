#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{DEFAULT_CONTAINMENT_BUDGET, DEFAULT_NETWORK_SEED, MAX_CONTAINMENT_BUDGET};
use math_explorer::epidemiology::networks::{NetworkEpidemicModel, NodeState};
use std::collections::HashSet;

const fn default_seed() -> u64 {
    DEFAULT_NETWORK_SEED
}

const fn is_default_seed(seed: &u64) -> bool {
    *seed == DEFAULT_NETWORK_SEED
}

const fn default_budget() -> usize {
    DEFAULT_CONTAINMENT_BUDGET
}

const fn is_default_budget(budget: &usize) -> bool {
    *budget == DEFAULT_CONTAINMENT_BUDGET
}

const fn is_false(value: &bool) -> bool {
    !*value
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NetPropState {
    pub(super) num_nodes: usize,
    pub(super) beta: f64,
    pub(super) gamma: f64,
    #[serde(default = "default_seed", skip_serializing_if = "is_default_seed")]
    pub(super) seed: u64,
    #[serde(default = "default_budget", skip_serializing_if = "is_default_budget")]
    pub(super) budget: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) vaccinated_nodes: Vec<usize>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(super) quarantined_edges: Vec<usize>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub(super) is_running: bool,
}

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

pub(super) fn build_validated_candidate(
    s: &NetPropState,
) -> Result<
    (
        NetworkEpidemicModel,
        oxidize_core::rng::OxidizeRng,
        Vec<usize>,
        Vec<usize>,
    ),
    String,
> {
    if !(10..=200).contains(&s.num_nodes)
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
    let vax_sorted = validate_unique_indices(&s.vaccinated_nodes, "vaccinated_nodes")?;
    let edge_sorted = validate_unique_indices(&s.quarantined_edges, "quarantined_edges")?;

    let mut candidate_rng = oxidize_core::rng::OxidizeRng::new(s.seed);
    let mut candidate =
        NetworkEpidemicModel::new_with_rng(s.num_nodes, s.beta, s.gamma, &mut candidate_rng);

    for &node in &vax_sorted {
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
    for &edge_idx in &edge_sorted {
        let &[u, v] = candidate
            .edges()
            .get(edge_idx)
            .ok_or_else(|| format!("Quarantined edge index {edge_idx} out of bounds"))?;
        candidate
            .set_edge_quarantined(u, v, true)
            .map_err(|e| e.to_string())?;
    }
    Ok((candidate, candidate_rng, vax_sorted, edge_sorted))
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Graph Epidemic Spread",
    "domain": "epidemiology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "num_nodes": 60,
    "beta": 0.15,
    "gamma": 0.05
  }
}"#;

pub(super) const HUB_VACCINATION_CHALLENGE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Super-Spreader Hub Vaccination",
    "domain": "epidemiology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "num_nodes": 72,
    "beta": 0.16,
    "gamma": 0.05,
    "seed": 20261121
  }
}"#;

pub(super) const BRIDGE_CUT_CHALLENGE_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Community Bridge Quarantine",
    "domain": "epidemiology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "num_nodes": 56,
    "beta": 0.22,
    "gamma": 0.04,
    "seed": 20261122
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_propagation_starter",
        title: "Graph Epidemic Spread",
        description: "Stochastic SIR contagion propagation across a random geometric contact network.",
        category: "Network Epidemiology",
        icon: "🕸️",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_propagation_hub_vaccination",
        title: "Super-Spreader Hub Vaccination",
        description: "Immunize up to 5 high-degree hubs on a seeded geometric contact network to collapse R₀.",
        category: "Containment Challenges",
        icon: "💉",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || HUB_VACCINATION_CHALLENGE_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "network_propagation_bridge_cut",
        title: "Community Bridge Quarantine",
        description: "Quarantine up to 5 inter-cluster bridge edges on a seeded contact graph to halt contagion spread.",
        category: "Containment Challenges",
        icon: "✂️",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || BRIDGE_CUT_CHALLENGE_JSON,
    }
}
