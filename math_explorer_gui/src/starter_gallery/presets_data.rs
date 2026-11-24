//! Embedded JSON preset envelopes re-exported by [`crate::starter_gallery`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// JSON preset envelope for On-Attractor Lorenz Butterfly Twins.
pub const PRESET_BUTTERFLY_TWINS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz Butterfly Twins (ε = 1e-9)",
    "domain": "chaos",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.01,
    "simulation_speed": 8,
    "execution_state": "Running",
    "show_annotations": true,
    "camera": { "pitch": 0.25, "yaw": -0.35, "zoom": 1.0 },
    "system_kind": "Lorenz",
    "initial_state": [-4.385295763939413, -2.064285615514713, 25.90419269166981],
    "shadow_enabled": true,
    "epsilon": 1e-9,
    "perturbation_expr": "epsilon"
  }
}"#;

/// JSON preset envelope for Rössler Spiral-Fold Strange Attractor.
pub const PRESET_ROSSLER_ATTRACTOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Rössler Spiral-Fold Attractor",
    "domain": "chaos",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.02,
    "simulation_speed": 8,
    "execution_state": "Running",
    "show_annotations": false,
    "camera": { "pitch": 0.55, "yaw": -0.6, "zoom": 1.0 },
    "system_kind": "Rossler",
    "initial_state": [0.1, 0.0, 0.0],
    "shadow_enabled": true,
    "epsilon": 1e-8,
    "perturbation_expr": "epsilon",
    "rossler_params": [0.2, 0.2, 5.7]
  }
}"#;

/// JSON preset envelope for Lagrangian Chaotic Double Pendulum.
pub const PRESET_DOUBLE_PENDULUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Chaotic Double Pendulum Twins",
    "domain": "chaos",
    "timestamp": "2026-11-22T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.005,
    "simulation_speed": 6,
    "execution_state": "Running",
    "show_annotations": false,
    "camera": { "pitch": 0.3, "yaw": -0.4, "zoom": 1.0 },
    "system_kind": "DoublePendulum",
    "initial_state": [2.4, 2.4, 0.0],
    "shadow_enabled": true,
    "epsilon": 1e-9,
    "perturbation_expr": "epsilon",
    "pendulum_params": [1.0, 1.0, 1.0, 1.0, 9.81]
  }
}"#;
