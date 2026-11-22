use crate::framework::{encode_u8_grid, InputMode, InteractiveTool, RasterBrushState, RasterBrushStroke};
use eframe::egui;
use math_commons::math_kernel::types::flatten_2d_index;
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{
    algorithms::TabularQAgent,
    grid_world::{GridState, GridWorldEnv, Move},
    q_function::TabularQFunction,
    strategies::EpsilonGreedy,
    MarkovDecisionProcess,
};
use std::hash::{Hash, Hasher};

#[path = "grid_world/preset.rs"]
mod preset;
#[path = "grid_world/ui.rs"]
mod ui;

/// Interactive brush modes for sculpting [`GridWorldTool`] environments.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GridBrushMode {
    /// Paints impassable wall cells blocking agent transitions.
    #[default]
    Wall,
    /// Paints terminal `-10.0` penalty trap cells.
    Trap,
    /// Relocates the episode `start` state.
    Start,
    /// Relocates the terminal `+10.0` reward `goal` state.
    Goal,
    /// Clears walls or traps back to open navigable floor.
    Empty,
}

impl GridBrushMode {
    /// All selectable GridWorld sculpting modes in UI display order.
    pub const ALL: [Self; 5] = [
        Self::Wall,
        Self::Trap,
        Self::Start,
        Self::Goal,
        Self::Empty,
    ];

    /// User-facing label for side-panel sculpting controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Wall => "Paint Wall (Impassable)",
            Self::Trap => "Paint Trap (-10 Penalty)",
            Self::Start => "Place Start (S)",
            Self::Goal => "Place Goal (+10 Reward)",
            Self::Empty => "Erase Cell",
        }
    }
}

pub struct GridWorldTool {
    pub(crate) env: GridWorldEnv,
    pub(crate) agent: TabularQAgent<GridState, Move>,
    pub(crate) learning_rate: f64,
    pub(crate) discount_factor: f64,
    pub(crate) exploration_rate: f64,
    pub(crate) current_state: GridState,
    pub(crate) episodes: u32,
    pub(crate) total_reward: f64,
    pub(crate) steps: u32,
    pub(crate) is_playing: bool,
    pub(crate) playback_speed: f32,
    pub(crate) last_tick: crate::web_io::PlatformInstant,
    pub(crate) brush_state: RasterBrushState,
    pub(crate) brush_mode: GridBrushMode,
}

impl Default for GridWorldTool {
    fn default() -> Self {
        let learning_rate = 0.1;
        let discount_factor = 0.9;
        let exploration_rate = 0.1;
        let env = GridWorldEnv {
            width: 5,
            height: 5,
            start: GridState { x: 0, y: 0 },
            goal: GridState { x: 4, y: 4 },
            traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
            walls: Vec::new(),
            gamma: discount_factor,
        };
        let agent = Self::build_agent(learning_rate, discount_factor, exploration_rate);
        Self {
            current_state: env.start,
            env,
            agent,
            learning_rate,
            discount_factor,
            exploration_rate,
            episodes: 0,
            total_reward: 0.0,
            steps: 0,
            is_playing: false,
            playback_speed: 5.0,
            last_tick: crate::web_io::PlatformInstant::now(),
            brush_state: RasterBrushState::with_radius(0),
            brush_mode: GridBrushMode::default(),
        }
    }
}

impl GridWorldTool {
    fn build_agent(
        learning_rate: f64,
        discount_factor: f64,
        exploration_rate: f64,
    ) -> TabularQAgent<GridState, Move> {
        let lr = UnitInterval::new(learning_rate.clamp(0.0, 1.0)).unwrap();
        let gamma = UnitInterval::new(discount_factor.clamp(0.0, 1.0)).unwrap();
        let eps = UnitInterval::new(exploration_rate.clamp(0.0, 1.0)).unwrap();
        TabularQAgent::new(lr, gamma, eps, None)
    }

    pub(crate) fn sync_agent_hyperparameters(&mut self) {
        let Ok(lr) = UnitInterval::new(self.learning_rate.clamp(0.0, 1.0)) else {
            return;
        };
        let Ok(gamma) = UnitInterval::new(self.discount_factor.clamp(0.0, 1.0)) else {
            return;
        };
        let Ok(eps) = UnitInterval::new(self.exploration_rate.clamp(0.0, 1.0)) else {
            return;
        };
        self.env.gamma = gamma.value();
        let q_func = std::mem::replace(&mut self.agent.q_func, TabularQFunction::new());
        let rng = self.agent.rng.clone();
        let mut updated =
            TabularQAgent::new_generic(q_func, lr, gamma, Box::new(EpsilonGreedy::new(eps)), None);
        updated.rng = rng;
        self.agent = updated;
    }

    pub(crate) fn reset_agent(&mut self) {
        self.is_playing = false;
        self.env.gamma = self.discount_factor.clamp(0.0, 1.0);
        self.agent = Self::build_agent(
            self.learning_rate,
            self.discount_factor,
            self.exploration_rate,
        );
        self.reset_episode();
        self.episodes = 0;
    }

    pub(crate) fn step_agent(&mut self) {
        if self.env.is_terminal(&self.current_state) {
            self.reset_episode();
            return;
        }
        let actions = self.env.actions(&self.current_state);
        if actions.is_empty() {
            return;
        }
        if let Some(action) = self.agent.select_action(&self.current_state, &actions) {
            let next_state = self.env.step(&self.current_state, &action);
            let reward = self.env.reward(&self.current_state, &action, &next_state);
            let next_actions = self.env.actions(&next_state);
            self.agent.update(
                &self.current_state,
                &action,
                reward,
                &next_state,
                &next_actions,
            );
            self.current_state = next_state;
            self.total_reward += reward;
            self.steps += 1;
        }
    }

    pub(crate) fn reset_episode(&mut self) {
        self.current_state = self.env.start;
        self.episodes += 1;
        self.total_reward = 0.0;
        self.steps = 0;
    }

    pub(crate) fn train_batch(&mut self) {
        self.is_playing = false;
        for _ in 0..100 {
            for _ in 0..100 {
                if self.env.is_terminal(&self.current_state) {
                    break;
                }
                self.step_agent();
            }
            self.reset_episode();
        }
    }

    /// Applies a [`RasterBrushStroke`] to sculpt walls, traps, start, or goal cells live.
    pub fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke) {
        let mode = if stroke.secondary {
            GridBrushMode::Empty
        } else {
            self.brush_mode
        };
        if matches!(mode, GridBrushMode::Start | GridBrushMode::Goal) {
            let (fx, fy) = stroke.focus_cell;
            let target = GridState {
                x: fx as i32,
                y: fy as i32,
            };
            if target.x >= 0
                && target.x < self.env.width
                && target.y >= 0
                && target.y < self.env.height
            {
                self.env.walls.retain(|&c| c != target);
                self.env.traps.retain(|&c| c != target);
                if mode == GridBrushMode::Start && target != self.env.goal {
                    self.env.start = target;
                    if self.steps == 0 {
                        self.current_state = target;
                    }
                } else if mode == GridBrushMode::Goal && target != self.env.start {
                    self.env.goal = target;
                }
            }
            return;
        }

        for &(cx, cy) in &stroke.cells {
            let cell = GridState {
                x: cx as i32,
                y: cy as i32,
            };
            if cell.x < 0
                || cell.x >= self.env.width
                || cell.y < 0
                || cell.y >= self.env.height
                || cell == self.env.start
                || cell == self.env.goal
            {
                continue;
            }
            self.env.walls.retain(|&c| c != cell);
            self.env.traps.retain(|&c| c != cell);
            match mode {
                GridBrushMode::Wall => self.env.walls.push(cell),
                GridBrushMode::Trap => self.env.traps.push(cell),
                GridBrushMode::Empty | GridBrushMode::Start | GridBrushMode::Goal => {}
            }
        }
        if self.env.walls.contains(&self.current_state) {
            self.current_state = self.env.start;
        }
    }

    /// Clears all sculpted walls and traps while preserving `start` and `goal`.
    pub fn clear_sculpted_obstacles(&mut self) {
        self.env.walls.clear();
        self.env.traps.clear();
    }

    fn encode_layout_rle(&self) -> Option<String> {
        let w = self.env.width.max(1) as usize;
        let h = self.env.height.max(1) as usize;
        let mut grid = vec![0u8; w * h];
        for wall in &self.env.walls {
            if wall.x >= 0 && (wall.x as usize) < w && wall.y >= 0 && (wall.y as usize) < h {
                grid[flatten_2d_index(wall.x as usize, wall.y as usize, w)] = 1;
            }
        }
        for trap in &self.env.traps {
            if trap.x >= 0 && (trap.x as usize) < w && trap.y >= 0 && (trap.y as usize) < h {
                grid[flatten_2d_index(trap.x as usize, trap.y as usize, w)] = 2;
            }
        }
        encode_u8_grid(w, h, &grid, 2).ok()
    }
}

use preset::{decode_walls_and_traps, resolve_endpoints, GridWorldToolState};

impl InteractiveTool for GridWorldTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Grid World (RL)"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "ai"
    }

    fn focused_canvas_id(&self) -> Option<egui::Id> {
        self.brush_state.focused_id
    }

    fn state_revision(&self) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.env.width.hash(&mut h);
        self.env.height.hash(&mut h);
        self.discount_factor.to_bits().hash(&mut h);
        self.learning_rate.to_bits().hash(&mut h);
        self.exploration_rate.to_bits().hash(&mut h);
        self.playback_speed.to_bits().hash(&mut h);
        self.is_playing.hash(&mut h);
        (self.env.start.x, self.env.start.y).hash(&mut h);
        (self.env.goal.x, self.env.goal.y).hash(&mut h);
        for w in &self.env.walls {
            (w.x, w.y).hash(&mut h);
        }
        for t in &self.env.traps {
            (t.x, t.y).hash(&mut h);
        }
        h.finish().max(1)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GridWorldToolState {
            width: self.env.width,
            height: self.env.height,
            gamma: self.discount_factor,
            learning_rate: self.learning_rate,
            exploration_rate: self.exploration_rate,
            playback_speed: self.playback_speed,
            is_playing: self.is_playing,
            start: Some([self.env.start.x, self.env.start.y]),
            goal: Some([self.env.goal.x, self.env.goal.y]),
            layout_rle: self.encode_layout_rle(),
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
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
        {
            return Err(
                "GridWorld dimensions, RL hyperparameters, or playback_speed out of bounds".into(),
            );
        }
        let lr_unit = UnitInterval::new(parsed.learning_rate)
            .map_err(|e| format!("Invalid learning rate: {e}"))?;
        let gamma_unit =
            UnitInterval::new(parsed.gamma).map_err(|e| format!("Invalid discount factor: {e}"))?;
        let eps_unit = UnitInterval::new(parsed.exploration_rate)
            .map_err(|e| format!("Invalid exploration rate: {e}"))?;
        let (start, goal) = resolve_endpoints(&parsed)?;
        let (walls, traps) = decode_walls_and_traps(&parsed, start, goal)?;
        self.learning_rate = lr_unit.value();
        self.discount_factor = gamma_unit.value();
        self.exploration_rate = eps_unit.value();
        self.env = GridWorldEnv {
            width: parsed.width,
            height: parsed.height,
            start,
            goal,
            traps,
            walls,
            gamma: parsed.gamma,
        };
        self.agent = TabularQAgent::new(lr_unit, gamma_unit, eps_unit, None);
        self.current_state = self.env.start;
        self.episodes = 0;
        self.total_reward = 0.0;
        self.steps = 0;
        self.playback_speed = parsed.playback_speed;
        self.is_playing = parsed.is_playing;
        self.last_tick = crate::web_io::PlatformInstant::now();
        Ok(())
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let ctx = parent_ui.ctx().clone();
        if self.is_playing {
            let speed_hz = self.playback_speed.clamp(1.0, 20.0);
            let interval = std::time::Duration::from_secs_f32(1.0 / speed_hz);
            let now = crate::web_io::PlatformInstant::now();
            if now.duration_since(self.last_tick) >= interval {
                self.step_agent();
                self.last_tick = now;
            }
            ctx.request_repaint();
        }
        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        self.draw_side_panel(parent_ui);
        self.draw_central_canvas(parent_ui, input_mode);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "GridWorldTool",
        display_name: "Grid World (RL)",
        domain: "ai",
        tags: &[],
        build: || Box::new(GridWorldTool::default()),
    }
}

#[cfg(test)]
#[path = "grid_world/tests.rs"]
mod tests;
