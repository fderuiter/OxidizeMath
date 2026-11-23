//! Built-in configurations and starter presets for [`super::OptimalAuctionTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{AuctionConfig, AuctionDistKind};

pub fn default_config() -> AuctionConfig {
    AuctionConfig {
        upper_bound: 100.0,
        lower_bound: 0.0,
        exp_rate: 0.2,
        dist_kind: AuctionDistKind::Uniform,
        n_bidders: 2,
        custom_reserve: 50.0,
        mc_trials: 4_000,
        seed: 2026,
    }
}

pub fn monopoly_config() -> AuctionConfig {
    AuctionConfig {
        upper_bound: 200.0,
        lower_bound: 0.0,
        exp_rate: 0.2,
        dist_kind: AuctionDistKind::Uniform,
        n_bidders: 1,
        custom_reserve: 100.0,
        mc_trials: 5_000,
        seed: 77,
    }
}

pub fn exponential_config() -> AuctionConfig {
    AuctionConfig {
        upper_bound: 16.0,
        lower_bound: 0.0,
        exp_rate: 0.25,
        dist_kind: AuctionDistKind::Exponential,
        n_bidders: 3,
        custom_reserve: 4.0,
        mc_trials: 5_000,
        seed: 314,
    }
}

pub const PRESET_SINGLE_BIDDER_MONOPOLY: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Single-Bidder Monopoly Reserve Surge",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "upper_bound": 200.0,
    "lower_bound": 0.0,
    "exp_rate": 0.2,
    "dist_kind": "Uniform",
    "n_bidders": 1,
    "custom_reserve": 100.0,
    "mc_trials": 5000,
    "seed": 77
  }
}"#;

pub const PRESET_EXPONENTIAL_HAZARD: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Constant-Hazard Exponential Auction",
    "domain": "game_theory",
    "timestamp": "2026-11-23T00:00:00Z"
  },
  "config": {
    "upper_bound": 16.0,
    "lower_bound": 0.0,
    "exp_rate": 0.25,
    "dist_kind": "Exponential",
    "n_bidders": 3,
    "custom_reserve": 4.0,
    "mc_trials": 5000,
    "seed": 314
  }
}"#;
