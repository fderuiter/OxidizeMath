//! Starter presets for [`super::NeuralNetworkVizTool`], including balanced E/I neural rings.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

pub(super) const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Spiking Neural Circuit",
    "domain": "neuroscience",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "external_input": [15.0, 0.0, 0.0, 0.0, 0.0],
    "weights": [
      [0.0, -1.0, 0.0, 0.0, 2.0],
      [2.0, 0.0, -1.0, 0.0, 0.0],
      [0.0, 2.0, 0.0, -1.0, 0.0],
      [0.0, 0.0, 2.0, 0.0, -1.0],
      [-1.0, 0.0, 0.0, 2.0, 0.0]
    ],
    "dt": 0.05,
    "is_running": false
  }
}"#;

pub(super) const EI_NEURAL_RINGS_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "E/I Balanced Neural Ring",
    "domain": "neuroscience",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "external_input": [18.0, 4.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
    "weights": [],
    "dt": 0.05,
    "is_running": false,
    "follow_workspace_graph": true,
    "seeded_graph": {
      "revision": 1,
      "nodes": [
        { "id": 0, "norm_pos": [0.85, 0.50] },
        { "id": 1, "norm_pos": [0.75, 0.75] },
        { "id": 2, "norm_pos": [0.50, 0.85] },
        { "id": 3, "norm_pos": [0.25, 0.75] },
        { "id": 4, "norm_pos": [0.15, 0.50] },
        { "id": 5, "norm_pos": [0.25, 0.25] },
        { "id": 6, "norm_pos": [0.50, 0.15] },
        { "id": 7, "norm_pos": [0.75, 0.25] }
      ],
      "edges": [
        [0, 1, 2.2],
        [1, 2, 2.2],
        [2, 3, 2.2],
        [3, 4, 2.2],
        [4, 5, 2.2],
        [5, 6, 2.2],
        [6, 7, 2.2],
        [7, 0, 2.2],
        [0, 4, -1.8],
        [1, 5, -1.5],
        [2, 6, -1.8],
        [3, 7, -1.5]
      ]
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "neural_network_viz_starter",
        title: "Spiking Neural Circuit",
        description: "Coupled 5-neuron Hodgkin-Huxley ring circuit with excitatory and inhibitory synaptic currents.",
        category: "Neural Circuits",
        icon: "⚡",
        domain: "neuroscience",
        tool_name: Some("NeuralNetworkVizTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "ei_neural_rings",
        title: "E/I Balanced Neural Ring",
        description: "Excitatory (+w) ring propagation regulated by cross-ring inhibitory (-w) Hodgkin-Huxley synapses.",
        category: "Neural Circuits",
        icon: "🧠",
        domain: "neuroscience",
        tool_name: Some("NeuralNetworkVizTool"),
        preset_generator: || EI_NEURAL_RINGS_JSON,
    }
}
