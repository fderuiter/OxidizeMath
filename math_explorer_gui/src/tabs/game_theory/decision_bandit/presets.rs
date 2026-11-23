//! Built-in configurations and starter presets for [`super::DecisionBanditTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{DecisionBanditConfig, DecisionWorkbenchMode, RiskMode};

pub fn default_config() -> DecisionBanditConfig {
    DecisionBanditConfig {
        ucb_c: std::f64::consts::SQRT_2,
        mode: DecisionWorkbenchMode::DecisionTreeVoi,
        risk_mode: RiskMode::Neutral,
        risk_param: 0.01,
        safe_payoff: 20.0,
        success_payoff: 120.0,
        failure_payoff: -40.0,
        prior_good: 0.45,
        signal_sensitivity: 0.80,
        signal_specificity: 0.80,
        arm_probs: vec![0.20, 0.75, 0.45],
        horizon: 300,
        rng_seed: 2026,
    }
}

pub fn cara_seismic_config() -> DecisionBanditConfig {
    DecisionBanditConfig {
        ucb_c: 1.2,
        mode: DecisionWorkbenchMode::DecisionTreeVoi,
        risk_mode: RiskMode::Cara,
        risk_param: 0.015,
        safe_payoff: 15.0,
        success_payoff: 140.0,
        failure_payoff: -50.0,
        prior_good: 0.40,
        signal_sensitivity: 0.85,
        signal_specificity: 0.82,
        arm_probs: vec![0.25, 0.80, 0.50],
        horizon: 250,
        rng_seed: 42,
    }
}

pub fn thompson_vs_ucb1_config() -> DecisionBanditConfig {
    DecisionBanditConfig {
        ucb_c: 1.414,
        mode: DecisionWorkbenchMode::BanditRegret,
        risk_mode: RiskMode::Crra,
        risk_param: 1.0,
        safe_payoff: 50.0,
        success_payoff: 150.0,
        failure_payoff: 20.0,
        prior_good: 0.50,
        signal_sensitivity: 0.80,
        signal_specificity: 0.80,
        arm_probs: vec![0.15, 0.45, 0.82, 0.60],
        horizon: 400,
        rng_seed: 777,
    }
}

pub const PRESET_CARA_SEISMIC_VOI: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CARA Risk-Averse Drilling & EVSI",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "ucb_c": 1.2,
    "mode": "DecisionTreeVoi",
    "risk_mode": "Cara",
    "risk_param": 0.015,
    "safe_payoff": 15.0,
    "success_payoff": 140.0,
    "failure_payoff": -50.0,
    "prior_good": 0.4,
    "signal_sensitivity": 0.85,
    "signal_specificity": 0.82,
    "arm_probs": [0.25, 0.8, 0.5],
    "horizon": 250,
    "rng_seed": 42
  }
}"#;

pub const PRESET_THOMPSON_VS_UCB1: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Thompson vs. UCB1 Regret Race",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "ucb_c": 1.414,
    "mode": "BanditRegret",
    "risk_mode": "Crra",
    "risk_param": 1.0,
    "safe_payoff": 50.0,
    "success_payoff": 150.0,
    "failure_payoff": 20.0,
    "prior_good": 0.5,
    "signal_sensitivity": 0.8,
    "signal_specificity": 0.8,
    "arm_probs": [0.15, 0.45, 0.82, 0.6],
    "horizon": 400,
    "rng_seed": 777
  }
}"#;
