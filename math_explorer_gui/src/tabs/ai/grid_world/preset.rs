//! Compact RLE URL/preset serialization and starter gallery registrations for [`super::GridWorldTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::GridWorldTool;
use crate::framework::{decode_u8_grid, encode_u8_grid};
use crate::tabs::ai::rl_sculpt::{
    apply_rle_to_mdp, encode_custom_cells_rle, is_zero_f64, solve_bellman_with_policy,
    RlOverlayMode, RlSolverMode,
};
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{
    DenseRlWorkspace, GridCellRole, GridState, SculptedGridMdp,
};

fn default_rl_rate() -> f64 {
    0.1
}

fn is_default_start(pos: &Option<[i32; 2]>) -> bool {
    matches!(pos, None | Some([0, 0]))
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
    #[serde(default, skip_serializing_if = "is_default_start")]
    start: Option<[i32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    goal: Option<[i32; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    layout_rle: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    rle_mask: Vec<[u32; 2]>,
}

fn apply_legacy_layout_and_endpoints(
    mdp: &mut SculptedGridMdp,
    parsed: &GridWorldToolState,
) -> Result<(), String> {
    if let Some(ref rle) = parsed.layout_rle {
        let w = parsed.width as usize;
        let h = parsed.height as usize;
        let cells = decode_u8_grid(rle, w, h, 2)?;
        let wind = mdp.wind_slip_prob;
        *mdp = SculptedGridMdp::new(parsed.width, parsed.height, parsed.gamma, wind);
        for (idx, sym) in cells.into_iter().enumerate() {
            let st = GridState {
                x: (idx % w) as i32,
                y: (idx / w) as i32,
            };
            match sym {
                1 if st != mdp.goal => mdp.set_cell_role(st, GridCellRole::SolidWall),
                2 if st != mdp.goal => mdp.set_cell_role(st, GridCellRole::Trap),
                _ => {}
            }
        }
    }
    if let Some([gx, gy]) = parsed.goal {
        let goal = GridState { x: gx, y: gy };
        if !mdp.in_bounds(goal) {
            return Err("GridWorld goal coordinates out of bounds".into());
        }
        mdp.set_cell_role(goal, GridCellRole::Goal);
    }
    if let Some([sx, sy]) = parsed.start {
        let start = GridState { x: sx, y: sy };
        if !mdp.in_bounds(start) {
            return Err("GridWorld start coordinates out of bounds".into());
        }
        if start == mdp.goal {
            return Err("GridWorld start and goal cannot occupy the same cell".into());
        }
        mdp.set_start(start);
    }
    Ok(())
}

impl GridWorldTool {
    pub(super) fn save_preset_state(&self) -> Option<String> {
        let start = (self.mdp.start != GridState { x: 0, y: 0 })
            .then_some([self.mdp.start.x, self.mdp.start.y]);
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
            start,
            goal: None,
            layout_rle: None,
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
        if !parsed.rle_mask.is_empty() {
            apply_rle_to_mdp(&mut mdp, &parsed.rle_mask)?;
        }
        apply_legacy_layout_and_endpoints(&mut mdp, &parsed)?;

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

fn generate_grid_world_labyrinth_preset() -> &'static str {
    static LABYRINTH_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    LABYRINTH_PRESET
        .get_or_init(|| {
            let w = 8usize;
            let h = 8usize;
            let mut grid = vec![0u8; w * h];
            for &(wx, wy) in &[
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (2, 4),
                (5, 3),
                (5, 4),
                (5, 5),
                (5, 6),
                (5, 7),
            ] {
                grid[flatten_2d_index(wx, wy, w)] = 1;
            }
            grid[flatten_2d_index(5, 2, w)] = 2;
            grid[flatten_2d_index(2, 5, w)] = 2;
            let rle = encode_u8_grid(w, h, &grid, 2).expect("valid GridWorld labyrinth RLE");
            serde_json::json!({
                "metadata": {
                    "schema_version": "1.0",
                    "title": "Grid World Labyrinth Maze",
                    "domain": "ai",
                    "timestamp": "2026-11-21T00:00:00Z"
                },
                "config": {
                    "width": 8,
                    "height": 8,
                    "gamma": 0.95,
                    "learning_rate": 0.25,
                    "exploration_rate": 0.15,
                    "playback_speed": 12.0,
                    "is_playing": false,
                    "start": [0, 0],
                    "goal": [7, 7],
                    "layout_rle": rle
                }
            })
            .to_string()
        })
        .as_str()
}

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

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "grid_world_labyrinth_maze",
        title: "Grid World Labyrinth Maze",
        description: "Sculpted 8x8 switchback wall labyrinth with hazard traps demonstrating Bellman temporal-difference rerouting.",
        category: "Reinforcement Learning",
        icon: "🧱",
        domain: "ai",
        tool_name: Some("GridWorldTool"),
        preset_generator: generate_grid_world_labyrinth_preset,
    }
}
