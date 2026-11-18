#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{HeatmapView, QTableInspectorTool, QTableRunner};
use crate::async_sim::BackgroundWorker;
use math_explorer::ai::reinforcement_learning::algorithms::TabularQAgent;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv};

#[derive(serde::Serialize, serde::Deserialize)]
struct QTableInspectorState {
    width: i32,
    height: i32,
    gamma: f64,
    view: HeatmapView,
}

impl QTableInspectorTool {
    pub(super) fn serialize_state(&self) -> Option<String> {
        serde_json::to_string(&QTableInspectorState {
            width: self.env.width,
            height: self.env.height,
            gamma: self.env.gamma,
            view: self.view,
        })
        .ok()
    }

    pub(super) fn deserialize_state(&mut self, state: &str) -> Result<(), String> {
        let s: QTableInspectorState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid QTableInspectorTool state: {e}"))?;
        let gamma_iv = math_commons::primitives::UnitInterval::new(s.gamma)
            .map_err(|e| format!("Invalid discount factor in QTableInspectorTool: {e}"))?;
        if s.width < 2 || s.height < 2 {
            return Err("Invalid grid dimensions in QTableInspectorTool".to_string());
        }
        let build_env = || {
            let traps = vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }]
                .into_iter()
                .filter(|t| {
                    t.x < s.width && t.y < s.height && !(t.x == s.width - 1 && t.y == s.height - 1)
                })
                .collect();
            GridWorldEnv {
                width: s.width,
                height: s.height,
                start: GridState { x: 0, y: 0 },
                goal: GridState {
                    x: s.width - 1,
                    y: s.height - 1,
                },
                traps,
                gamma: s.gamma,
            }
        };
        let build_agent = || {
            TabularQAgent::new(
                math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                gamma_iv,
                math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                None,
            )
        };
        self.env = build_env();
        self.cached_q_values.clear();
        self.episodes_trained = 0;
        self.view = s.view;
        self.worker = BackgroundWorker::new(QTableRunner {
            env: build_env(),
            agent: build_agent(),
            episodes_trained: 0,
            target_episodes: 0,
            is_training: false,
        });
        self.sync_from_latest();
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
    "width": 5,
    "height": 5,
    "gamma": 0.95,
    "view": "MaxQ"
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
