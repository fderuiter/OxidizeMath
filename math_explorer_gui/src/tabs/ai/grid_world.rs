//! Interactive 2D GridWorld reinforcement learning laboratory with tactile maze sculpting,
//! exact Bellman value-iteration overlays, policy-gradient rollouts, and TD sonification.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::rl_sculpt::{
    apply_rl_brush_stroke, mean_bellman_residual, new_rl_brush_kit, render_rl_sculpt_toolbar,
    solve_bellman_with_policy, RlMazePreset, RlOverlayMode, RlSolverMode,
};
use crate::accessibility::AccessibleHoverText;
use crate::audio::{mappings::rl_td_surprise_frame, render_sonification_panel, SonificationFrame};
use crate::framework::{InputMode, InteractionContext, InteractiveTool, SpatialBrushKit};
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
        if self.mdp.is_terminal(&self.current_state) {
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

        self.current_state = outcome.next_state;
        self.total_reward += outcome.reward;
        self.steps += 1;
        self.last_td_error = outcome.td_error;
        self.latest_audio = rl_td_surprise_frame(
            outcome.state_value,
            outcome.td_error,
            outcome.reward,
            outcome.wind_pan,
        );
    }

    pub(super) fn train_batch(&mut self) {
        self.is_playing = false;
        for _ in 0..100 {
            for _ in 0..100 {
                if self.mdp.is_terminal(&self.current_state) {
                    break;
                }
                self.step_agent();
            }
            self.reset_episode();
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

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "grid_world_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.draw_transport_and_solver_controls(ui);
                ui.separator();
                ui.heading("Maze Sculpting");
                if render_rl_sculpt_toolbar(
                    ui,
                    "grid_world",
                    &mut self.mdp,
                    &mut self.brush_kit,
                    &mut self.maze_preset,
                ) {
                    if self.mdp.cell_role(self.current_state) == GridCellRole::SolidWall {
                        self.current_state = self.mdp.start;
                    }
                    self.bellman_dirty = true;
                    self.ensure_bellman_up_to_date();
                    self.revision = self.revision.wrapping_add(1);
                }
                ui.separator();
                self.draw_hyperparameter_controls(ui);
                ui.separator();
                render_sonification_panel(ui, "grid_world", &self.latest_audio, !self.is_playing);
            });
        });
    }

    fn draw_transport_and_solver_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("▶ Step")
                .accessible_hover_text("Advance the agent by one step")
                .clicked()
            {
                self.step_agent();
            }
            let (label, hover) = if self.is_playing {
                ("⏸ Pause", "Pause continuous RL playback")
            } else {
                ("▶ Play", "Start continuous RL playback")
            };
            if ui.button(label).accessible_hover_text(hover).clicked() {
                self.is_playing = !self.is_playing;
                if self.is_playing {
                    self.last_tick = crate::web_io::PlatformInstant::now();
                }
            }
        });
        if ui
            .button("▶ Train (100 Episodes)")
            .accessible_hover_text("Train the agent for 100 episodes instantly")
            .clicked()
        {
            self.train_batch();
        }
        if ui
            .button("↻ Reset Agent")
            .accessible_hover_text("Clear learned values and reset agent state")
            .clicked()
        {
            self.reset_agent();
        }
        self.draw_solver_and_overlay_selectors(ui);
    }

    fn draw_solver_and_overlay_selectors(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Solver:");
            egui::ComboBox::from_id_salt("grid_world_solver")
                .selected_text(self.solver_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [RlSolverMode::TabularQ, RlSolverMode::PolicyGradient] {
                        if ui
                            .selectable_value(&mut self.solver_mode, mode, mode.label())
                            .changed()
                        {
                            self.bellman_dirty = true;
                            self.ensure_bellman_up_to_date();
                            self.revision = self.revision.wrapping_add(1);
                        }
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Overlay:");
            egui::ComboBox::from_id_salt("grid_world_overlay")
                .selected_text(self.overlay_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [
                        RlOverlayMode::EmpiricalPolicy,
                        RlOverlayMode::BellmanVStar,
                        RlOverlayMode::PolicyValueVPi,
                        RlOverlayMode::BellmanResidual,
                    ] {
                        if ui
                            .selectable_value(&mut self.overlay_mode, mode, mode.label())
                            .changed()
                        {
                            self.bellman_dirty = true;
                            self.ensure_bellman_up_to_date();
                            self.revision = self.revision.wrapping_add(1);
                        }
                    }
                });
        });
    }

    fn draw_hyperparameter_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("RL Hyperparameters");
        let mut hp_changed = ui
            .add(egui::Slider::new(&mut self.learning_rate, 0.01..=1.0).text("Learning Rate (α)"))
            .changed();
        hp_changed |= ui
            .add(egui::Slider::new(&mut self.discount_factor, 0.0..=1.0).text("Discount (γ)"))
            .changed();
        hp_changed |= ui
            .add(egui::Slider::new(&mut self.exploration_rate, 0.0..=1.0).text("Exploration (ε)"))
            .changed();
        if hp_changed {
            self.sync_agent_hyperparameters();
        }
        ui.add(egui::Slider::new(&mut self.playback_speed, 1.0..=20.0).text("Speed (Hz)"));
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
