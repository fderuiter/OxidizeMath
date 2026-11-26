//! Starter and containment challenge presets for [`super::NetworkPropagationTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{DEFAULT_CONTAINMENT_BUDGET, DEFAULT_NETWORK_SEED};
use crate::app::data_bus::SharedGraph;
use crate::app::TopologySource;

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

const fn default_true() -> bool {
    true
}

const fn is_false(value: &bool) -> bool {
    !*value
}

const fn is_true(value: &bool) -> bool {
    *value
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
    #[serde(default, skip_serializing_if = "TopologySource::is_default")]
    pub(super) topology_source: TopologySource,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub(super) follow_workspace_graph: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) seeded_graph: Option<SharedGraph>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) frozen_graph: Option<SharedGraph>,
}

pub(super) const STARTER_PRESET_JSON: &str = r#"{
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

pub(super) const HUB_SPOKE_SUPERSPREADERS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Hub-and-Spoke Superspreaders",
    "domain": "epidemiology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "num_nodes": 13,
    "beta": 0.08,
    "gamma": 0.10,
    "follow_workspace_graph": true,
    "seeded_graph": {
      "revision": 1,
      "nodes": [
        { "id": 0, "norm_pos": [0.50, 0.50] },
        { "id": 1, "norm_pos": [0.88, 0.50] },
        { "id": 2, "norm_pos": [0.83, 0.69] },
        { "id": 3, "norm_pos": [0.69, 0.83] },
        { "id": 4, "norm_pos": [0.50, 0.88] },
        { "id": 5, "norm_pos": [0.31, 0.83] },
        { "id": 6, "norm_pos": [0.17, 0.69] },
        { "id": 7, "norm_pos": [0.12, 0.50] },
        { "id": 8, "norm_pos": [0.17, 0.31] },
        { "id": 9, "norm_pos": [0.31, 0.17] },
        { "id": 10, "norm_pos": [0.50, 0.12] },
        { "id": 11, "norm_pos": [0.69, 0.17] },
        { "id": 12, "norm_pos": [0.83, 0.31] }
      ],
      "edges": [
        [0, 1, 1.0],
        [0, 2, 1.0],
        [0, 3, 1.0],
        [0, 4, 1.0],
        [0, 5, 1.0],
        [0, 6, 1.0],
        [0, 7, 1.0],
        [0, 8, 1.0],
        [0, 9, 1.0],
        [0, 10, 1.0],
        [0, 11, 1.0],
        [0, 12, 1.0],
        [1, 2, 1.0],
        [7, 8, 1.0]
      ]
    }
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
        id: "hub_spoke_superspreaders",
        title: "Hub-and-Spoke Superspreaders",
        description: "High degree-variance star contact network driving R₀_het > 1 despite subcritical mean-field β/γ.",
        category: "Network Epidemiology",
        icon: "💥",
        domain: "epidemiology",
        tool_name: Some("NetworkPropagationTool"),
        preset_generator: || HUB_SPOKE_SUPERSPREADERS_JSON,
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
