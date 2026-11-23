//! Built-in JSON preset envelopes shared across starter gallery registrations and tests.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

/// JSON preset envelope for Harmonic Oscillator ODE.
pub const PRESET_HARMONIC_OSCILLATOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y'' = -k*y (Harmonic Oscillator)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "HarmonicOscillator",
    "dt": 0.05,
    "total_time": 20.0,
    "param_k": 2.0,
    "param_r": 1.0,
    "param_cap_k": 10.0,
    "ic_y0": 1.5,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Logistic Growth ODE.
pub const PRESET_LOGISTIC_GROWTH: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y' = r*y*(1 - y/K) (Logistic Growth)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "LogisticGrowth",
    "dt": 0.05,
    "total_time": 20.0,
    "param_k": 1.0,
    "param_r": 0.8,
    "param_cap_k": 10.0,
    "ic_y0": 0.5,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Exponential Growth/Decay ODE.
pub const PRESET_EXPONENTIAL_DECAY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "ODE Solver - y' = k*y (Exponential)",
    "domain": "analysis",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "preset": "Exponential",
    "dt": 0.05,
    "total_time": 10.0,
    "param_k": -0.5,
    "param_r": 1.0,
    "param_cap_k": 10.0,
    "ic_y0": 5.0,
    "ic_v0": 0.0
  }
}"#;

/// JSON preset envelope for Turing Spots reaction-diffusion.
pub const PRESET_TURING_SPOTS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis Simulation - Spots",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 1.0,
    "d_v": 100.0,
    "dt": 0.05,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Turing Stripes reaction-diffusion.
pub const PRESET_TURING_STRIPES: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis Simulation - Stripes",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.14,
    "b": 0.86,
    "d_u": 1.0,
    "d_v": 50.0,
    "dt": 0.02,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Cylinder Vector Field.
pub const PRESET_VECTOR_FIELD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Cylinder Vector Field",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Uniform",
        "strength": 10.0,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Doublet",
        "strength": 251.32741228718345,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 20,
    "view_range": 10.0
  }
}"#;

/// JSON preset envelope for Fluid Dynamics Source-Sink Pair.
pub const PRESET_SOURCE_SINK: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Potential Flow - Source-Sink Singularity Pair",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "elements": [
      {
        "element_type": "Source",
        "strength": 10.0,
        "x": -3.0,
        "y": 0.0,
        "angle": 0.0
      },
      {
        "element_type": "Sink",
        "strength": 10.0,
        "x": 3.0,
        "y": 0.0,
        "angle": 0.0
      }
    ],
    "grid_size": 25,
    "view_range": 12.0
  }
}"#;

/// JSON preset envelope for 3D Chaotic Attractor.
pub const PRESET_3D_ATTRACTOR: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Lorenz 3D Attractor - Butterfly",
    "domain": "chaos",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "sigma": 10.0,
    "rho": 28.0,
    "beta": 2.6666666666666665,
    "dt": 0.01,
    "simulation_speed": 5,
    "execution_state": "Running",
    "show_annotations": true,
    "camera": {
      "pitch": 0.35,
      "yaw": -0.45,
      "zoom": 1.0
    }
  }
}"#;

/// JSON preset envelope for Replicator Hawk-Dove Dynamics.
pub const PRESET_HAWK_DOVE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Replicator Dynamics - Hawk-Dove",
    "domain": "game_theory",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "payoff_matrix": [[-1.0, 2.0], [0.0, 1.0]],
    "initial_population": [0.5, 0.5],
    "time_horizon": 20.0,
    "dt": 0.05,
    "strategy_names": ["Hawk", "Dove"]
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
