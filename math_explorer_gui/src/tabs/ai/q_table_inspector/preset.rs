#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HeatmapView, QTableInspectorTool, QTableRunner};
use crate::async_sim::{BackgroundWorker, SimulationRunner};
use math_explorer::ai::reinforcement_learning::algorithms::TabularQAgent;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct QTableInspectorState {
    view: HeatmapView,
    gamma: f64,
    pretrain_episodes: u32,
}

fn build_env(gamma: f64) -> GridWorldEnv {
    GridWorldEnv {
        width: 5,
        height: 5,
        start: GridState { x: 0, y: 0 },
        goal: GridState { x: 4, y: 4 },
        traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
        gamma,
    }
}

impl QTableInspectorTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&QTableInspectorState {
            view: self.view,
            gamma: self.env.gamma,
            pretrain_episodes: self.episodes_trained,
        })
        .ok()
    }

    pub(super) fn try_load_preset_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: QTableInspectorState =
            serde_json::from_str(state).map_err(|e| format!("Invalid Q-table state: {e}"))?;
        if !parsed.gamma.is_finite()
            || !(0.0..=1.0).contains(&parsed.gamma)
            || parsed.pretrain_episodes > 5_000
        {
            return Err("Q-table gamma or pretrain_episodes out of bounds".into());
        }
        let gamma_unit = math_commons::primitives::UnitInterval::new(parsed.gamma)
            .map_err(|e| format!("Invalid discount factor: {e}"))?;
        let alpha = math_commons::primitives::UnitInterval::new(0.1).unwrap();
        let eps = math_commons::primitives::UnitInterval::new(0.1).unwrap();

        let mut runner = QTableRunner {
            env: build_env(parsed.gamma),
            agent: TabularQAgent::new(alpha, gamma_unit, eps, None),
            episodes_trained: 0,
            target_episodes: parsed.pretrain_episodes,
            is_training: parsed.pretrain_episodes > 0,
        };
        while !runner.is_complete() {
            runner.step();
        }
        self.env = build_env(parsed.gamma);
        self.cached_q_values.clear();
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let st = GridState { x, y };
                for action in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    self.cached_q_values
                        .insert((st, action), runner.agent.get_q_value(&st, &action));
                }
            }
        }
        self.episodes_trained = runner.episodes_trained;
        self.view = parsed.view;
        self.worker = BackgroundWorker::new(runner);
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
