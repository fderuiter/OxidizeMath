//! Compact RLE URL/preset serialization for [`super::GridWorldTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::tabs::ai::rl_sculpt::{
    apply_rle_to_mdp, encode_custom_cells_rle, is_zero_f64, solve_bellman_with_policy,
    RlOverlayMode, RlSolverMode,
};
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{DenseRlWorkspace, SculptedGridMdp};

fn default_rl_rate() -> UnitInterval {
    UnitInterval::new(0.1).expect("0.1 is in [0, 1]")
}

fn default_zero_unit() -> UnitInterval {
    UnitInterval::new(0.0).expect("0.0 is in [0, 1]")
}

fn is_zero_unit(val: &UnitInterval) -> bool {
    is_zero_f64(&val.value())
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct GridWorldToolState {
    width: i32,
    height: i32,
    #[serde(alias = "discount_factor")]
    gamma: UnitInterval,
    #[serde(default = "default_rl_rate")]
    learning_rate: UnitInterval,
    #[serde(default = "default_rl_rate")]
    exploration_rate: UnitInterval,
    playback_speed: f32,
    is_playing: bool,
    #[serde(default = "default_zero_unit", skip_serializing_if = "is_zero_unit")]
    wind_slip_prob: UnitInterval,
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
            gamma: UnitInterval::new(self.discount_factor).ok()?,
            learning_rate: UnitInterval::new(self.learning_rate).ok()?,
            exploration_rate: UnitInterval::new(self.exploration_rate).ok()?,
            playback_speed: self.playback_speed,
            is_playing: self.is_playing,
            wind_slip_prob: UnitInterval::new(self.mdp.wind_slip_prob).ok()?,
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
            || parsed.learning_rate.value() < 0.01
            || !parsed.playback_speed.is_finite()
            || !(1.0..=20.0).contains(&parsed.playback_speed)
        {
            return Err(
                "GridWorld dimensions, RL hyperparameters, wind_slip_prob, or playback_speed out of bounds"
                    .into(),
            );
        }
        let gamma_val = parsed.gamma.value();
        let mut mdp = SculptedGridMdp::default_with_traps(parsed.width, parsed.height, gamma_val);
        mdp.wind_slip_prob = parsed.wind_slip_prob.value();
        apply_rle_to_mdp(&mut mdp, &parsed.rle_mask)?;

        self.learning_rate = parsed.learning_rate.value();
        self.discount_factor = gamma_val;
        self.exploration_rate = parsed.exploration_rate.value();
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
