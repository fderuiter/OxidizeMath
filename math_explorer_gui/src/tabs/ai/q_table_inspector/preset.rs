#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HeatmapView, QTableInspectorTool, SculptedQTableRunner};
use crate::async_sim::{BackgroundWorker, SimulationRunner};
use crate::tabs::ai::rl_sculpt::{
    apply_rle_to_mdp, encode_custom_cells_rle, is_zero_f64, RlSolverMode,
};
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::SculptedGridMdp;

fn default_dim() -> i32 {
    5
}

fn is_default_dim(val: &i32) -> bool {
    *val == 5
}

fn default_zero_unit() -> UnitInterval {
    UnitInterval::new(0.0).expect("0.0 is in [0, 1]")
}

fn is_zero_unit(val: &UnitInterval) -> bool {
    is_zero_f64(&val.value())
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QTableInspectorState {
    view: HeatmapView,
    gamma: UnitInterval,
    pretrain_episodes: u32,
    #[serde(default = "default_dim", skip_serializing_if = "is_default_dim")]
    width: i32,
    #[serde(default = "default_dim", skip_serializing_if = "is_default_dim")]
    height: i32,
    #[serde(default = "default_zero_unit", skip_serializing_if = "is_zero_unit")]
    wind_slip_prob: UnitInterval,
    #[serde(default, skip_serializing_if = "RlSolverMode::is_default")]
    solver_mode: RlSolverMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
}

impl QTableInspectorTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&QTableInspectorState {
            view: self.view,
            gamma: UnitInterval::new(self.mdp.gamma).ok()?,
            pretrain_episodes: self.episodes_trained,
            width: self.mdp.width,
            height: self.mdp.height,
            wind_slip_prob: UnitInterval::new(self.mdp.wind_slip_prob).ok()?,
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
            || parsed.pretrain_episodes > 5_000
        {
            return Err(
                "Q-table dimensions, gamma, wind_slip_prob, or pretrain_episodes out of bounds"
                    .into(),
            );
        }

        let mut mdp =
            SculptedGridMdp::default_with_traps(parsed.width, parsed.height, parsed.gamma.value());
        mdp.wind_slip_prob = parsed.wind_slip_prob.value();
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
        self.worker = BackgroundWorker::new(runner);
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
