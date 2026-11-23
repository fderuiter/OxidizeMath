//! Built-in configurations and starter presets for [`super::NashShapleyTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{NashShapleyConfig, NashShapleyViewMode};

pub fn default_config() -> NashShapleyConfig {
    NashShapleyConfig {
        payoff_probe: 1.0,
        mode: NashShapleyViewMode::BimatrixNash,
        rows: 2,
        cols: 2,
        row_payoffs: vec![2.0, 0.0, 0.0, 1.0],
        col_payoffs: vec![1.0, 0.0, 0.0, 2.0],
        zero_sum_lock: false,
        weak_dominance: false,
        initial_label: 0,
        coalition_players: 3,
        characteristic_values: vec![0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 6.0],
    }
}

pub fn rps_zero_sum_config() -> NashShapleyConfig {
    NashShapleyConfig {
        payoff_probe: 1.5,
        mode: NashShapleyViewMode::BimatrixNash,
        rows: 3,
        cols: 3,
        row_payoffs: vec![0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0],
        col_payoffs: vec![0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0],
        zero_sum_lock: true,
        weak_dominance: false,
        initial_label: 0,
        coalition_players: 3,
        characteristic_values: vec![0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    }
}

pub fn convex_synergy_config() -> NashShapleyConfig {
    NashShapleyConfig {
        payoff_probe: 2.0,
        mode: NashShapleyViewMode::CoalitionalShapley,
        rows: 2,
        cols: 2,
        row_payoffs: vec![3.0, 0.0, 5.0, 1.0],
        col_payoffs: vec![3.0, 5.0, 0.0, 1.0],
        zero_sum_lock: false,
        weak_dominance: false,
        initial_label: 0,
        coalition_players: 4,
        characteristic_values: vec![
            0.0, 1.0, 1.0, 3.0, 2.0, 4.0, 4.0, 7.0, 2.0, 4.0, 4.0, 7.0, 5.0, 8.0, 8.0, 12.0,
        ],
    }
}

pub const PRESET_RPS_ZERO_SUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Zero-Sum RPS Minimax & Glove Market",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "payoff_probe": 1.5,
    "mode": "BimatrixNash",
    "rows": 3,
    "cols": 3,
    "row_payoffs": [0.0, -1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 0.0],
    "col_payoffs": [0.0, 1.0, -1.0, -1.0, 0.0, 1.0, 1.0, -1.0, 0.0],
    "zero_sum_lock": true,
    "weak_dominance": false,
    "initial_label": 0,
    "coalition_players": 3,
    "characteristic_values": [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
  }
}"#;

pub const PRESET_CONVEX_SYNERGY_SHAPLEY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Airport Runway Shapley & Core Allocation",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "payoff_probe": 2.0,
    "mode": "CoalitionalShapley",
    "rows": 2,
    "cols": 2,
    "row_payoffs": [3.0, 0.0, 5.0, 1.0],
    "col_payoffs": [3.0, 5.0, 0.0, 1.0],
    "zero_sum_lock": false,
    "weak_dominance": false,
    "initial_label": 0,
    "coalition_players": 4,
    "characteristic_values": [0.0, 1.0, 1.0, 3.0, 2.0, 4.0, 4.0, 7.0, 2.0, 4.0, 4.0, 7.0, 5.0, 8.0, 8.0, 12.0]
  }
}"#;
