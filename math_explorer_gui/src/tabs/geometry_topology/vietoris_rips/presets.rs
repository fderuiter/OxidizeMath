#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

const STARTER_PRESET_2D_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Vietoris-Rips Filtration",
    "domain": "geometry_topology",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "radius": 2.2,
    "points": [
      [5.0, 8.0], [6.5, 7.6], [7.6, 6.5], [8.0, 5.0],
      [7.6, 3.5], [6.5, 2.4], [5.0, 2.0], [3.5, 2.4],
      [2.4, 3.5], [2.0, 5.0], [2.4, 6.5], [3.5, 7.6]
    ]
  }
}"#;

const STARTER_PRESET_LORENZ_3D_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "3D Lorenz Strange Attractor Homology",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mode": "lorenz_3d",
    "radius": 7.5,
    "chaos": {
      "sigma": 10.0,
      "rho": 28.0,
      "beta": 2.6666666666666665,
      "dt": 0.01,
      "landmark_count": 64,
      "takens_lag": 8
    },
    "camera": {
      "pitch": 0.35,
      "yaw": 0.55,
      "zoom": 1.0
    }
  }
}"#;

const STARTER_PRESET_TAKENS_3D_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Takens Delay Attractor Reconstruction",
    "domain": "geometry_topology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mode": "takens_3d",
    "radius": 7.0,
    "chaos": {
      "sigma": 10.0,
      "rho": 28.0,
      "beta": 2.6666666666666665,
      "dt": 0.01,
      "landmark_count": 64,
      "takens_lag": 10
    },
    "camera": {
      "pitch": 0.45,
      "yaw": -0.6,
      "zoom": 1.0
    }
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_starter",
        title: "Vietoris-Rips Filtration",
        description: "Simplicial complex filtration of vertices, edges, and triangles across distance radius ε.",
        category: "Algebraic Topology",
        icon: "🕸️",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_2D_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_lorenz_3d",
        title: "3D Lorenz Strange Attractor Homology",
        description: "64-landmark 3D Lorenz butterfly filtration locking onto β₁ = 2 persistent loops alongside λ₁ and D₂.",
        category: "Algebraic Topology",
        icon: "🦋",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_LORENZ_3D_JSON,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vietoris_rips_takens_3d",
        title: "Takens Delay Attractor Reconstruction",
        description: "3D delay-coordinate embedding (x(t), x(t-τ), x(t-2τ)) reconstructing strange-attractor topology from scalar observations.",
        category: "Algebraic Topology",
        icon: "🌀",
        domain: "geometry_topology",
        tool_name: Some("VietorisRipsTool"),
        preset_generator: || STARTER_PRESET_TAKENS_3D_JSON,
    }
}
