#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

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
