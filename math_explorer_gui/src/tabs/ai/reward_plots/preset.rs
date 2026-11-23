//! Compact RLE URL/preset serialization for [`super::RewardPlotsTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::RewardPlotsTool;
use crate::tabs::ai::rl_sculpt::{
    apply_rle_to_mdp, encode_custom_cells_rle, is_zero_f64, RlSolverMode,
};
use math_explorer::ai::reinforcement_learning::SculptedGridMdp;

fn default_dim() -> i32 {
    5
}

fn is_default_dim(val: &i32) -> bool {
    *val == 5
}

fn default_gamma() -> f64 {
    0.9
}

fn is_default_gamma(val: &f64) -> bool {
    (*val - 0.9).abs() <= 1e-12
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RewardPlotsState {
    training_steps: usize,
    #[serde(default = "default_dim", skip_serializing_if = "is_default_dim")]
    width: i32,
    #[serde(default = "default_dim", skip_serializing_if = "is_default_dim")]
    height: i32,
    #[serde(default = "default_gamma", skip_serializing_if = "is_default_gamma")]
    gamma: f64,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    wind_slip_prob: f64,
    #[serde(default, skip_serializing_if = "RlSolverMode::is_default")]
    solver_mode: RlSolverMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
}

impl RewardPlotsTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&RewardPlotsState {
            training_steps: self.training_steps,
            width: self.mdp.width,
            height: self.mdp.height,
            gamma: self.mdp.gamma,
            wind_slip_prob: self.mdp.wind_slip_prob,
            solver_mode: self.solver_mode,
            rle_mask: encode_custom_cells_rle(&self.mdp),
        })
        .ok()
    }

    pub(super) fn try_load_preset_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: RewardPlotsState =
            serde_json::from_str(state).map_err(|e| format!("Invalid reward plots state: {e}"))?;
        if !(10..=1000).contains(&parsed.training_steps)
            || !(2..=20).contains(&parsed.width)
            || !(2..=20).contains(&parsed.height)
            || !parsed.gamma.is_finite()
            || !(0.0..=1.0).contains(&parsed.gamma)
            || !parsed.wind_slip_prob.is_finite()
            || !(0.0..=1.0).contains(&parsed.wind_slip_prob)
        {
            return Err(
                "RewardPlots training_steps, dimensions, gamma, or wind_slip_prob out of bounds"
                    .into(),
            );
        }
        let mut mdp =
            SculptedGridMdp::default_with_traps(parsed.width, parsed.height, parsed.gamma);
        mdp.wind_slip_prob = parsed.wind_slip_prob;
        apply_rle_to_mdp(&mut mdp, &parsed.rle_mask)?;

        self.training_steps = parsed.training_steps;
        self.mdp = mdp;
        self.solver_mode = parsed.solver_mode;
        self.reset();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Reward Trajectory",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "training_steps": 200
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "reward_plots_starter",
        title: "Reward Trajectory",
        description: "Episodic cumulative return trajectory across tabular Q-learning Grid World training episodes.",
        category: "Reinforcement Learning",
        icon: "🎯",
        domain: "ai",
        tool_name: Some("RewardPlotsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
