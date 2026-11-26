#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HeatmapView, QTableInspectorTool, SculptedQTableRunner};
use crate::async_sim::SimulationRunner;
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

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QTableInspectorState {
    view: HeatmapView,
    gamma: f64,
    pretrain_episodes: u32,
    #[serde(default = "default_dim", skip_serializing_if = "is_default_dim")]
    width: i32,
    #[serde(default = "default_dim", skip_serializing_if = "is_default_dim")]
    height: i32,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    wind_slip_prob: f64,
    #[serde(default, skip_serializing_if = "RlSolverMode::is_default")]
    solver_mode: RlSolverMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
}

impl QTableInspectorTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&QTableInspectorState {
            view: self.view,
            gamma: self.mdp.gamma,
            pretrain_episodes: self.episodes_trained,
            width: self.mdp.width,
            height: self.mdp.height,
            wind_slip_prob: self.mdp.wind_slip_prob,
            solver_mode: self.solver_mode,
            rle_mask: encode_custom_cells_rle(&self.mdp),
        })
        .ok()
    }

    pub(super) fn try_load_preset_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: QTableInspectorState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Q-table state: {e}"))?;
        if !(2..=20).contains(&parsed.width)
            || !(2..=20).contains(&parsed.height)
            || !parsed.gamma.is_finite()
            || !(0.0..=1.0).contains(&parsed.gamma)
            || !parsed.wind_slip_prob.is_finite()
            || !(0.0..=1.0).contains(&parsed.wind_slip_prob)
            || parsed.pretrain_episodes > 5_000
        {
            return Err(
                "Q-table dimensions, gamma, wind_slip_prob, or pretrain_episodes out of bounds"
                    .into(),
            );
        }
        let _ = math_commons::primitives::UnitInterval::new(parsed.gamma)
            .map_err(|e| format!("Invalid discount factor: {e}"))?;

        let mut mdp =
            SculptedGridMdp::default_with_traps(parsed.width, parsed.height, parsed.gamma);
        mdp.wind_slip_prob = parsed.wind_slip_prob;
        apply_rle_to_mdp(&mut mdp, &parsed.rle_mask)?;

        let mut runner = SculptedQTableRunner::from_mdp(mdp.clone(), parsed.solver_mode);
        runner.target_episodes = parsed.pretrain_episodes;
        runner.is_training = parsed.pretrain_episodes > 0;
        while !runner.is_complete() {
            runner.step();
        }

        self.mdp = mdp;
        self.solver_mode = parsed.solver_mode;
        self.latest_snap = runner.build_q_snapshot();
        self.episodes_trained = runner.episodes_trained;
        self.latest_audio = runner.latest_audio;
        self.view = parsed.view;
        self.worker.replace_runner(runner);
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Q-Table Inspector",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "view": "MaxQ",
    "gamma": 0.95,
    "pretrain_episodes": 200
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "q_table_inspector_starter",
        title: "Q-Table Inspector",
        description: "Tabular temporal difference state-action value matrix visualization and convergence tracking.",
        category: "Reinforcement Learning",
        icon: "📊",
        domain: "ai",
        tool_name: Some("QTableInspectorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
