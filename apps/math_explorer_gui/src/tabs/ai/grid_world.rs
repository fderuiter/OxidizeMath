//! Interactive 2D GridWorld reinforcement learning laboratory with tactile maze sculpting,
//! exact Bellman value-iteration overlays, policy-gradient rollouts, and TD sonification.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::rl_sculpt::{
    apply_rl_brush_stroke, mean_bellman_residual, new_rl_brush_kit, solve_bellman_with_policy,
    RlMazePreset, RlOverlayMode, RlSolverMode,
};
use crate::audio::{mappings::rl_td_surprise_frame, SonificationFrame};
use crate::framework::{
    InputMode, InteractionContext, InteractiveTool, RasterBrushState, RasterBrushStroke,
    SpatialBrushKit,
};
use eframe::egui;
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{
    compute_softmax_pg_delta, step_softmax_pg_into, step_tabular_q_into, BellmanSolution,
    DenseRlWorkspace, GridCellRole, GridState, MarkovDecisionProcess, Move, SculptedGridMdp,
};

#[path = "grid_world/canvas.rs"]
mod canvas;
#[path = "grid_world/preset.rs"]
mod preset;
#[cfg(test)]
#[path = "grid_world/tests.rs"]
mod tests;
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
    pub const ALL: [Self; 5] = [Self::Wall, Self::Trap, Self::Start, Self::Goal, Self::Empty];

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

/// Interactive GridWorld RL tool with live maze sculpting, Bellman overlays, and sonification.
pub struct GridWorldTool {
    pub(super) mdp: SculptedGridMdp,
    pub(super) workspace: DenseRlWorkspace,
    pub(super) bellman: BellmanSolution,
    pub(super) bellman_dirty: bool,
    pub(super) solver_mode: RlSolverMode,
    pub(super) overlay_mode: RlOverlayMode,
    pub(super) maze_preset: RlMazePreset,
    pub(super) brush_kit: SpatialBrushKit,
    pub(crate) brush_state: RasterBrushState,
    pub(crate) brush_mode: GridBrushMode,
    pub(super) learning_rate: f64,
    pub(super) discount_factor: f64,
    pub(super) exploration_rate: f64,
    pub(super) current_state: GridState,
    pub(super) episodes: u32,
    pub(super) total_reward: f64,
    pub(super) steps: u32,
    pub(super) last_td_error: f64,
    pub(super) last_pg_grad_norm: f64,
    pub(super) latest_audio: SonificationFrame,
    pub(super) is_playing: bool,
    pub(super) playback_speed: f32,
    pub(super) last_tick: crate::web_io::PlatformInstant,
    pub(super) revision: u64,
}

impl Default for GridWorldTool {
    fn default() -> Self {
        let learning_rate = 0.1;
        let discount_factor = 0.9;
        let exploration_rate = 0.1;
        let mdp = SculptedGridMdp::default_with_traps(5, 5, discount_factor);
        let workspace = DenseRlWorkspace::new(5, 5, 100, None);
        let bellman = solve_bellman_with_policy(&mdp, &workspace, RlSolverMode::TabularQ);
        Self {
            current_state: mdp.start,
            mdp,
            workspace,
            bellman,
            bellman_dirty: false,
            solver_mode: RlSolverMode::TabularQ,
            overlay_mode: RlOverlayMode::EmpiricalPolicy,
            maze_preset: RlMazePreset::BaselineTraps,
            brush_kit: new_rl_brush_kit(),
            brush_state: RasterBrushState::with_radius(0),
            brush_mode: GridBrushMode::default(),
            learning_rate,
            discount_factor,
            exploration_rate,
            episodes: 0,
            total_reward: 0.0,
            steps: 0,
            last_td_error: 0.0,
            last_pg_grad_norm: 0.0,
            latest_audio: SonificationFrame::silent(),
            is_playing: false,
            playback_speed: 5.0,
            last_tick: crate::web_io::PlatformInstant::now(),
            revision: 0,
        }
    }
}

impl GridWorldTool {
    pub(super) fn ensure_bellman_up_to_date(&mut self) {
        if self.bellman_dirty {
            self.bellman = solve_bellman_with_policy(&self.mdp, &self.workspace, self.solver_mode);
            self.bellman_dirty = false;
        }
    }

    pub(super) fn sync_agent_hyperparameters(&mut self) {
        self.learning_rate = self.learning_rate.clamp(0.01, 1.0);
        self.discount_factor = self.discount_factor.clamp(0.0, 1.0);
        self.exploration_rate = self.exploration_rate.clamp(0.0, 1.0);
        if (self.mdp.gamma - self.discount_factor).abs() > 1e-12 {
            self.mdp.gamma = self.discount_factor;
            self.bellman_dirty = true;
        }
        self.ensure_bellman_up_to_date();
        self.revision = self.revision.wrapping_add(1);
    }

    pub(super) fn reset_agent(&mut self) {
        self.is_playing = false;
        self.mdp.gamma = self.discount_factor.clamp(0.0, 1.0);
        self.workspace.max_episode_steps =
            ((self.mdp.width as usize) * (self.mdp.height as usize) * 4).max(100);
        self.workspace
            .resize_and_clear(self.mdp.width as usize, self.mdp.height as usize);
        self.workspace.reset_episode(&self.mdp);
        self.current_state = self.mdp.start;
        self.episodes = 0;
        self.total_reward = 0.0;
        self.steps = 0;
        self.last_td_error = 0.0;
        self.last_pg_grad_norm = 0.0;
        self.latest_audio = SonificationFrame::silent();
        self.bellman_dirty = true;
        self.ensure_bellman_up_to_date();
    }

    pub(super) fn reset_episode(&mut self) {
        self.workspace.reset_episode(&self.mdp);
        self.current_state = self.mdp.start;
        self.episodes += 1;
        self.total_reward = 0.0;
        self.steps = 0;
        self.bellman_dirty = true;
        self.ensure_bellman_up_to_date();
    }

    pub(super) fn step_agent(&mut self) {
        let max_steps = self.workspace.max_episode_steps.max(4);
        if self.mdp.is_terminal(&self.current_state) || (self.steps as usize) >= max_steps {
            self.reset_episode();
            return;
        }
        let Ok(lr) = UnitInterval::new(self.learning_rate.clamp(0.01, 1.0)) else {
            return;
        };
        let Ok(eps) = UnitInterval::new(self.exploration_rate.clamp(0.0, 1.0)) else {
            return;
        };

        self.workspace.current_state = self.current_state;
        self.workspace.episode_steps = self.steps as usize;
        let outcome = match self.solver_mode {
            RlSolverMode::TabularQ => step_tabular_q_into(&self.mdp, &mut self.workspace, lr, eps),
            RlSolverMode::PolicyGradient => {
                let s_idx = self.mdp.cell_index(self.current_state);
                let logits = self.workspace.theta_logits[s_idx];
                let out = step_softmax_pg_into(&self.mdp, &mut self.workspace, lr);
                let delta = compute_softmax_pg_delta(logits, out.action, out.td_error, lr);
                self.last_pg_grad_norm = delta.iter().map(|d| d * d).sum::<f64>().sqrt();
                out
            }
        };

        self.last_td_error = outcome.td_error;
        self.latest_audio = rl_td_surprise_frame(
            outcome.state_value,
            outcome.td_error,
            outcome.reward,
            outcome.wind_pan,
        );

        let next_steps = self.steps.saturating_add(1);
        let timed_out = !self.mdp.is_terminal(&outcome.next_state)
            && (outcome.episode_finished || (next_steps as usize) >= max_steps);
        if timed_out {
            self.reset_episode();
        } else {
            self.current_state = outcome.next_state;
            self.total_reward += outcome.reward;
            self.steps = next_steps;
            self.bellman_dirty = true;
        }
    }

    pub(super) fn train_batch(&mut self) {
        self.is_playing = false;
        let max_steps = self.workspace.max_episode_steps.max(4);
        for _ in 0..100 {
            let ep_before = self.episodes;
            for _ in 0..max_steps {
                if self.mdp.is_terminal(&self.current_state) {
                    self.reset_episode();
                    break;
                }
                self.step_agent();
                if self.episodes > ep_before {
                    break;
                }
            }
            if self.episodes == ep_before {
                self.reset_episode();
            }
        }
    }

    pub(super) fn greedy_action_for_display(&self, state: GridState) -> Option<Move> {
        if self.mdp.is_terminal(&state) || self.mdp.cell_role(state) == GridCellRole::SolidWall {
            return None;
        }
        match self.overlay_mode {
            RlOverlayMode::BellmanVStar => self.bellman.optimal_action_at(state),
            _ => match self.solver_mode {
                RlSolverMode::TabularQ => self.workspace.greedy_q_action(state),
                RlSolverMode::PolicyGradient => self.workspace.greedy_policy_action(state),
            },
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
            let target = GridState {
                x: stroke.focus_cell.0 as i32,
                y: stroke.focus_cell.1 as i32,
            };
            if self.mdp.in_bounds(target) {
                if mode == GridBrushMode::Start && target != self.mdp.goal {
                    self.mdp.set_start(target);
                    if self.steps == 0 {
                        self.current_state = self.mdp.start;
                    }
                } else if mode == GridBrushMode::Goal && target != self.mdp.start {
                    self.mdp.set_cell_role(target, GridCellRole::Goal);
                }
                self.bellman_dirty = true;
                self.ensure_bellman_up_to_date();
                self.revision = self.revision.wrapping_add(1);
            }
            return;
        }

        let role = match mode {
            GridBrushMode::Wall => GridCellRole::SolidWall,
            GridBrushMode::Trap => GridCellRole::Trap,
            GridBrushMode::Empty | GridBrushMode::Start | GridBrushMode::Goal => GridCellRole::Free,
        };
        for &(cx, cy) in &stroke.cells {
            let cell = GridState {
                x: cx as i32,
                y: cy as i32,
            };
            if !self.mdp.in_bounds(cell) || cell == self.mdp.start || cell == self.mdp.goal {
                continue;
            }
            self.mdp.set_cell_role(cell, role);
        }
        if self.mdp.cell_role(self.current_state) == GridCellRole::SolidWall {
            self.current_state = self.mdp.start;
        }
        self.bellman_dirty = true;
        self.ensure_bellman_up_to_date();
        self.revision = self.revision.wrapping_add(1);
    }

    /// Clears all sculpted non-goal cells while preserving `start` and `goal`.
    pub fn clear_sculpted_obstacles(&mut self) {
        for y in 0..self.mdp.height {
            for x in 0..self.mdp.width {
                let st = GridState { x, y };
                if st != self.mdp.goal {
                    self.mdp.set_cell_role(st, GridCellRole::Free);
                }
            }
        }
        self.bellman_dirty = true;
        self.ensure_bellman_up_to_date();
        self.revision = self.revision.wrapping_add(1);
    }
}

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
        self.revision
    }
    fn on_brush(&mut self, ctx: &InteractionContext) {
        let Some(stroke) = ctx.brush_stroke else {
            return;
        };
        if apply_rl_brush_stroke(&mut self.mdp, &stroke) {
            if self.mdp.cell_role(self.current_state) == GridCellRole::SolidWall {
                self.current_state = self.mdp.start;
            }
            self.bellman_dirty = true;
            self.ensure_bellman_up_to_date();
            self.revision = self.revision.wrapping_add(1);
        }
    }
    fn observables(&self) -> Vec<(&'static str, f64)> {
        let residual =
            mean_bellman_residual(&self.mdp, &self.workspace, self.solver_mode, &self.bellman);
        vec![
            ("episodes", f64::from(self.episodes)),
            ("total_reward", self.total_reward),
            ("bellman_residual", residual),
            ("wind_slip_prob", self.mdp.wind_slip_prob),
        ]
    }
    fn save_state(&self) -> Option<String> {
        self.save_preset_state()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.try_load_preset_state(state)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        self.ensure_bellman_up_to_date();
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
        tags: &[
            "reinforcement learning",
            "rl",
            "q-learning",
            "bellman",
            "mdp",
            "policy",
            "epsilon-greedy",
            "sonification",
        ],
        build: || Box::new(GridWorldTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for GridWorldTool {
    fn theory_description(&self) -> String {
        format!(
            "Sculptable Markov Decision Process with bounce-back walls, wind slip $p = {:.2}$, Bellman optimality $V^*(s) = \\max_a \\sum_{{s'}} P(s'|s,a)[R + \\gamma V^*(s')]$, and {} ($\\alpha = {:.2}, \\gamma = {:.2}, \\epsilon = {:.2}$).",
            self.mdp.wind_slip_prob,
            self.solver_mode.label(),
            self.learning_rate,
            self.discount_factor,
            self.exploration_rate
        )
    }
    fn phonetic_description(&self) -> String {
        "Reinforcement Learning involves an agent navigating a sculptable Markov Decision Process using Bellman optimality value iteration, Q-learning, or policy gradients.".into()
    }
    fn theory_citation(&self) -> String {
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT press."
            .into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "learning_rate".into(),
            "Step size α updating Q-values or softmax policy preferences.".into(),
        );
        map.insert(
            "discount_factor".into(),
            "Discount factor γ weighting future cumulative rewards.".into(),
        );
        map.insert(
            "exploration_rate".into(),
            "Epsilon-greedy probability ε of sampling exploratory moves.".into(),
        );
        map
    }
}
