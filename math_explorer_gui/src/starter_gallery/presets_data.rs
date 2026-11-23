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

/// JSON preset envelope for Glicko-2 Swiss Upset & Volatility Spike.
pub const PRESET_GLICKO2_SWISS_UPSET: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Glicko-2 Tournament - Swiss Upset & Volatility Spike",
    "domain": "game_theory",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "tau": 0.6,
    "draw_probability": 0.15,
    "pairing_mode": "Swiss",
    "target_rounds": 5,
    "rng_seed": 42,
    "players": [
      {"name": "GM Vega", "rating": 1820.0, "rd": 65.0, "volatility": 0.06, "inactive": false},
      {"name": "IM Kestrel", "rating": 1710.0, "rd": 80.0, "volatility": 0.06, "inactive": false},
      {"name": "FM Orion", "rating": 1615.0, "rd": 95.0, "volatility": 0.06, "inactive": false},
      {"name": "CM Lyra", "rating": 1520.0, "rd": 140.0, "volatility": 0.07, "inactive": false},
      {"name": "Nova Prodigy", "rating": 1440.0, "rd": 240.0, "volatility": 0.09, "inactive": false}
    ],
    "periods": [
      {"matches": [{"player_a": 0, "player_b": 1, "score_a": 0.0}, {"player_a": 2, "player_b": 3, "score_a": 0.5}], "bye_player": 4},
      {"matches": [{"player_a": 1, "player_b": 4, "score_a": 0.0}, {"player_a": 0, "player_b": 2, "score_a": 0.0}], "bye_player": 3},
      {"matches": [{"player_a": 4, "player_b": 0, "score_a": 1.0}, {"player_a": 2, "player_b": 3, "score_a": 0.5}], "bye_player": 1}
    ]
  }
}"#;

/// JSON preset envelope for Glicko-2 Round-Robin Frozen vs. Sequential Drift.
pub const PRESET_GLICKO2_ROUND_ROBIN_DRIFT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Glicko-2 Tournament - Round-Robin Frozen vs. Sequential Drift",
    "domain": "game_theory",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "tau": 0.5,
    "draw_probability": 0.1,
    "pairing_mode": "RoundRobin",
    "target_rounds": 3,
    "rng_seed": 108,
    "players": [
      {"name": "Alpha", "rating": 1500.0, "rd": 260.0, "volatility": 0.06, "inactive": false},
      {"name": "Bravo", "rating": 1500.0, "rd": 260.0, "volatility": 0.06, "inactive": false},
      {"name": "Charlie", "rating": 1500.0, "rd": 260.0, "volatility": 0.06, "inactive": false},
      {"name": "Delta", "rating": 1500.0, "rd": 260.0, "volatility": 0.06, "inactive": false}
    ],
    "periods": [
      {
        "matches": [
          {"player_a": 0, "player_b": 1, "score_a": 1.0},
          {"player_a": 0, "player_b": 2, "score_a": 1.0},
          {"player_a": 0, "player_b": 3, "score_a": 0.0},
          {"player_a": 1, "player_b": 2, "score_a": 1.0},
          {"player_a": 1, "player_b": 3, "score_a": 0.5},
          {"player_a": 2, "player_b": 3, "score_a": 0.0}
        ]
      }
    ]
  }
}"#;

/// JSON preset envelope for MRI Physics Bloch Precession & Relaxation.
pub const PRESET_MRI_BLOCH_RELAXATION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "MRI Bloch Relaxation - Off-Resonance Precession",
    "domain": "mri",
    "timestamp": "2026-11-20T00:00:00Z"
  },
  "config": {
    "t1": 1.2,
    "t2": 0.35,
    "bx": 0.00000002,
    "by": 0.0,
    "bz": 0.0000001,
    "time_scale": 5.0
  }
}"#;

/// JSON preset envelope for Favoritism Sibling Rivalry Strategy.
pub const PRESET_FAVORITISM_SIBLING_RIVALRY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Unified Favoritism - Crisis & Proximity Showdown",
    "domain": "favoritism",
    "timestamp": "2026-11-20T00:00:00Z"
  },
  "config": {
    "children": [
      {
        "name": "Crisis Hero",
        "inputs": {
          "days_integrated": 1825.0,
          "distance_km": 15.0,
          "g_practical": 9.0,
          "g_emotional": 8.5,
          "wealth": 6.0,
          "talent": 7.0,
          "intelligence": 8.0,
          "emotional_sensitivity": 7.5,
          "helped_during_crisis": true,
          "active_on_social_media": true,
          "birth_order_weight": 1.2,
          "time_since_last_contact": 2.0,
          "f_initial": 15.0
        }
      },
      {
        "name": "Distant Heir",
        "inputs": {
          "days_integrated": 1825.0,
          "distance_km": 450.0,
          "g_practical": 9.5,
          "g_emotional": 3.0,
          "wealth": 9.5,
          "talent": 8.0,
          "intelligence": 8.5,
          "emotional_sensitivity": 3.5,
          "helped_during_crisis": false,
          "active_on_social_media": false,
          "birth_order_weight": 1.0,
          "time_since_last_contact": 45.0,
          "f_initial": 4.0
        }
      }
    ],
    "selected_child_index": 0,
    "new_child_name": "Youngest Sibling"
  }
}"#;
