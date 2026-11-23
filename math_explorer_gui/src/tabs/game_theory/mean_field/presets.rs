//! Built-in configurations and starter presets for [`super::MeanFieldGameTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::MeanFieldConfig;

pub fn default_config() -> MeanFieldConfig {
    MeanFieldConfig {
        congestion_weight: 1.5,
        viscosity: 0.08,
        time_horizon: 0.35,
        terminal_attraction: 1.5,
        target_center: 0.0,
        bimodal_initial: false,
        grid_points: 31,
        iterations: 6,
        slice_fraction: 0.5,
    }
}

pub fn high_congestion_config() -> MeanFieldConfig {
    MeanFieldConfig {
        congestion_weight: 6.0,
        viscosity: 0.10,
        time_horizon: 0.40,
        terminal_attraction: 0.8,
        target_center: 0.0,
        bimodal_initial: false,
        grid_points: 31,
        iterations: 7,
        slice_fraction: 0.5,
    }
}

pub fn bimodal_coalescence_config() -> MeanFieldConfig {
    MeanFieldConfig {
        congestion_weight: 0.8,
        viscosity: 0.06,
        time_horizon: 0.40,
        terminal_attraction: 2.8,
        target_center: 0.25,
        bimodal_initial: true,
        grid_points: 33,
        iterations: 7,
        slice_fraction: 0.5,
    }
}

pub const PRESET_HIGH_CONGESTION: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Crowd-Averse Congestion Dispersion",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "congestion_weight": 6.0,
    "viscosity": 0.1,
    "time_horizon": 0.4,
    "terminal_attraction": 0.8,
    "target_center": 0.0,
    "bimodal_initial": false,
    "grid_points": 31,
    "iterations": 7,
    "slice_fraction": 0.5
  }
}"#;

pub const PRESET_BIMODAL_COALESCENCE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Bimodal Crowd Coalescence to Target",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "congestion_weight": 0.8,
    "viscosity": 0.06,
    "time_horizon": 0.4,
    "terminal_attraction": 2.8,
    "target_center": 0.25,
    "bimodal_initial": true,
    "grid_points": 33,
    "iterations": 7,
    "slice_fraction": 0.5
  }
}"#;
