//! Compact RLE URL/preset serialization for [`super::GridWorldTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::tabs::ai::rl_sculpt::{
    RlOverlayMode, RlSolverMode, apply_rle_to_mdp, encode_custom_cells_rle, is_zero_f64,
    solve_bellman_with_policy,
};
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{DenseRlWorkspace, SculptedGridMdp};

fn default_rl_rate() -> f64 {
    0.1
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GridWorldToolState {
    width: i32,
    height: i32,
    #[serde(alias = "discount_factor")]
    gamma: f64,
    #[serde(default = "default_rl_rate")]
    learning_rate: f64,
    #[serde(default = "default_rl_rate")]
    exploration_rate: f64,
    playback_speed: f32,
    is_playing: bool,
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    wind_slip_prob: f64,
    #[serde(default, skip_serializing_if = "RlSolverMode::is_default")]
    solver_mode: RlSolverMode,
    #[serde(default, skip_serializing_if = "RlOverlayMode::is_default")]
    overlay_mode: RlOverlayMode,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
}

impl GridWorldTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        serde_json::to_string(&GridWorldToolState {
            width: self.mdp.width,
            height: self.mdp.height,
            gamma: self.discount_factor,
            learning_rate: self.learning_rate,
            exploration_rate: self.exploration_rate,
            playback_speed: self.playback_speed,
            is_playing: self.is_playing,
            wind_slip_prob: self.mdp.wind_slip_prob,
            solver_mode: self.solver_mode,
            overlay_mode: self.overlay_mode,
            rle_mask: encode_custom_cells_rle(&self.mdp),
        })
        .ok()
    }

    pub(super) fn try_load_preset_state(&mut self, state: &str) -> Result<(), String> {
        let parsed: GridWorldToolState =
            serde_json::from_str(state).map_err(|e| format!("Invalid GridWorld state: {e}"))?;
        if !(2..=20).contains(&parsed.width)
            || !(2..=20).contains(&parsed.height)
            || !parsed.gamma.is_finite()
            || !(0.0..=1.0).contains(&parsed.gamma)
            || !parsed.learning_rate.is_finite()
            || !(0.01..=1.0).contains(&parsed.learning_rate)
            || !parsed.exploration_rate.is_finite()
            || !(0.0..=1.0).contains(&parsed.exploration_rate)
            || !parsed.playback_speed.is_finite()
            || !(1.0..=20.0).contains(&parsed.playback_speed)
            || !parsed.wind_slip_prob.is_finite()
            || !(0.0..=1.0).contains(&parsed.wind_slip_prob)
        {
            return Err(
                "GridWorld dimensions, RL hyperparameters, wind_slip_prob, or playback_speed out of bounds"
                    .into(),
            );
        }
        let lr_unit = UnitInterval::new(parsed.learning_rate)
            .map_err(|e| format!("Invalid learning rate: {e}"))?;
        let gamma_unit =
            UnitInterval::new(parsed.gamma).map_err(|e| format!("Invalid discount factor: {e}"))?;
        let eps_unit = UnitInterval::new(parsed.exploration_rate)
            .map_err(|e| format!("Invalid exploration rate: {e}"))?;

        let mut mdp =
            SculptedGridMdp::default_with_traps(parsed.width, parsed.height, parsed.gamma);
        mdp.wind_slip_prob = parsed.wind_slip_prob;
        apply_rle_to_mdp(&mut mdp, &parsed.rle_mask)?;

        self.learning_rate = lr_unit.value();
        self.discount_factor = gamma_unit.value();
        self.exploration_rate = eps_unit.value();
        self.solver_mode = parsed.solver_mode;
        self.overlay_mode = parsed.overlay_mode;
        self.workspace =
            DenseRlWorkspace::new(parsed.width as usize, parsed.height as usize, 100, None);
        self.workspace.reset_episode(&mdp);
        self.current_state = mdp.start;
        self.bellman = solve_bellman_with_policy(&mdp, &self.workspace, self.solver_mode);
        self.bellman_dirty = false;
        self.mdp = mdp;
        self.episodes = 0;
        self.total_reward = 0.0;
        self.steps = 0;
        self.last_td_error = 0.0;
        self.last_pg_grad_norm = 0.0;
        self.latest_audio = crate::audio::SonificationFrame::silent();
        self.playback_speed = parsed.playback_speed;
        self.is_playing = parsed.is_playing;
        self.last_tick = crate::web_io::PlatformInstant::now();
        self.revision = self.revision.wrapping_add(1);
        Ok(())
    }
}

const STARTER_PRESET_JSON: &str = r#"{"metadata":{"schema_version":"1.0","title":"Grid World RL","domain":"ai","timestamp":"2026-11-16T00:00:00Z"},"config":{"width":5,"height":5,"gamma":0.95,"playback_speed":8.0,"is_playing":false}}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "grid_world_starter",
        title: "Grid World RL",
        description: "Markov Decision Process navigation with obstacle avoidance and discount reward propagation.",
        category: "Reinforcement Learning",
        icon: "🗺️",
        domain: "ai",
        tool_name: Some("GridWorldTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
