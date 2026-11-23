//! Episodic cumulative return and Bellman optimality residual plotter across sculpted MDPs.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::rl_sculpt::{RlMazePreset, RlSolverMode, build_rl_sculpt_payload};
use crate::async_sim::ui::{
    WorkerFaultAction, render_empty_plot_guidance, render_worker_fault_banner,
};
use crate::async_sim::{SimCommand, SimWorkerState, SimulationController};
use crate::audio::{SonificationFrame, render_sonification_panel};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::ai::reinforcement_learning::SculptedGridMdp;

#[path = "reward_plots/preset.rs"]
mod preset;
#[path = "reward_plots/runner.rs"]
mod runner;
#[cfg(test)]
#[path = "reward_plots/tests.rs"]
mod tests;

pub use runner::{RewardPlotsRunner, RlSnapshotData};

/// Interactive RL episodic return and Bellman convergence curve visualizer.
pub struct RewardPlotsTool {
    pub(crate) controller: SimulationController,
    pub(super) mdp: SculptedGridMdp,
    pub(super) solver_mode: RlSolverMode,
    pub(super) maze_preset: RlMazePreset,
    pub(super) rewards_per_episode: Vec<f64>,
    pub(super) bellman_residuals: Vec<f64>,
    pub(super) episodes_trained: usize,
    pub(super) target_episodes: usize,
    pub(super) training_steps: usize,
    pub(super) show_bellman_residual: bool,
    pub(super) latest_audio: SonificationFrame,
    pub(super) revision: u64,
}

impl Default for RewardPlotsTool {
    fn default() -> Self {
        let mdp = SculptedGridMdp::default_with_traps(5, 5, 0.9);
        let runner = RewardPlotsRunner::from_mdp(mdp.clone(), RlSolverMode::TabularQ);
        let controller = SimulationController::new(runner);
        Self {
            controller,
            mdp,
            solver_mode: RlSolverMode::TabularQ,
            maze_preset: RlMazePreset::BaselineTraps,
            rewards_per_episode: Vec::new(),
            bellman_residuals: Vec::new(),
            episodes_trained: 0,
            target_episodes: 0,
            training_steps: 100,
            show_bellman_residual: true,
            latest_audio: SonificationFrame::silent(),
            revision: 0,
        }
    }
}

impl RewardPlotsTool {
    pub(super) fn sync_sculpt_to_worker(&mut self) {
        let payload = build_rl_sculpt_payload(&self.mdp, self.solver_mode);
        let _ = self
            .controller
            .send_command(SimCommand::SyncSculptState(payload));
        self.revision = self.revision.wrapping_add(1);
    }

    /// Resets the agent in place or respawns a disconnected worker.
    pub fn reset(&mut self) {
        let mdp_clone = self.mdp.clone();
        let solver = self.solver_mode;
        self.controller
            .reset_or_respawn(move || RewardPlotsRunner::from_mdp(mdp_clone, solver));
        let payload = build_rl_sculpt_payload(&self.mdp, self.solver_mode);
        let _ = self
            .controller
            .send_command(SimCommand::SyncSculptState(payload));
        self.rewards_per_episode.clear();
        self.bellman_residuals.clear();
        self.episodes_trained = 0;
        self.target_episodes = 0;
        self.latest_audio = SonificationFrame::silent();
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "reward_plots_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.draw_training_section(ui, ctx);
                ui.separator();
                self.draw_topology_section(ui);
                ui.separator();
                render_sonification_panel(
                    ui,
                    "reward_plots",
                    &self.latest_audio,
                    !self.controller.state().is_running(),
                );
            });
        });
    }

    fn draw_training_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Training Controls");
        if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
            render_worker_fault_banner(
                ui,
                self.controller.state(),
                self.controller.diagnostic_report(),
                false,
            )
        {
            self.reset();
            ctx.request_repaint();
            ui.separator();
        }

        let state = self.controller.state();
        ui.label(format!("Total Episodes: {}", self.episodes_trained));
        ui.add(egui::Slider::new(&mut self.training_steps, 10..=1000).text("Episodes to Train"));

        if state.is_running() {
            let remaining = self.target_episodes.saturating_sub(self.episodes_trained);
            let progress = if self.target_episodes > 0 {
                (self.episodes_trained as f32 / self.target_episodes as f32).clamp(0.0, 1.0)
            } else {
                0.0
            };
            ui.add(
                egui::ProgressBar::new(progress).text(format!("Training: {remaining} remaining")),
            );
        }

        let has_unfinished =
            self.episodes_trained < self.target_episodes && self.target_episodes > 0;
        let play_label = if has_unfinished {
            "▶ Resume".to_string()
        } else {
            format!("▶ Train {}", self.training_steps)
        };

        if let Some(action) =
            PlaybackTransport::from_worker_state("reward_plots_transport", "Reward Plots", state)
                .play_label(&play_label)
                .step_label("⏭ Step Episode")
                .can_reset(true)
                .show(ui)
        {
            self.handle_transport_action(action, has_unfinished, ctx);
        }
    }

    fn draw_topology_section(&mut self, ui: &mut egui::Ui) {
        ui.heading("Solver & Maze Topology");
        ui.horizontal(|ui| {
            ui.label("Solver:");
            egui::ComboBox::from_id_salt("reward_plots_solver")
                .selected_text(self.solver_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [RlSolverMode::TabularQ, RlSolverMode::PolicyGradient] {
                        if ui
                            .selectable_value(&mut self.solver_mode, mode, mode.label())
                            .changed()
                        {
                            self.sync_sculpt_to_worker();
                        }
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Maze:");
            egui::ComboBox::from_id_salt("reward_plots_maze")
                .selected_text(self.maze_preset.label())
                .show_ui(ui, |ui| {
                    for preset in RlMazePreset::ALL {
                        if ui
                            .selectable_value(&mut self.maze_preset, preset, preset.label())
                            .clicked()
                        {
                            self.mdp =
                                preset.build_mdp(self.mdp.width, self.mdp.height, self.mdp.gamma);
                            self.sync_sculpt_to_worker();
                        }
                    }
                });
        });
        let mut changed = ui
            .add(egui::Slider::new(&mut self.mdp.wind_slip_prob, 0.0..=0.8).text("Wind Slip p"))
            .changed();
        changed |= ui
            .add(egui::Slider::new(&mut self.mdp.gamma, 0.0..=1.0).text("Discount (γ)"))
            .changed();
        if changed {
            self.sync_sculpt_to_worker();
        }
        ui.checkbox(
            &mut self.show_bellman_residual,
            "Show Mean Bellman Residual |V - V*|",
        );
    }

    fn handle_transport_action(
        &mut self,
        action: PlaybackAction,
        has_unfinished: bool,
        ctx: &egui::Context,
    ) {
        match action {
            PlaybackAction::Play => {
                if has_unfinished {
                    let _ = self.controller.send_command(SimCommand::Start);
                } else {
                    let _ = self
                        .controller
                        .send_command(SimCommand::TrainEpisodes(self.training_steps));
                }
            }
            PlaybackAction::Pause => {
                let _ = self.controller.send_command(SimCommand::Pause);
            }
            PlaybackAction::Step => {
                let _ = self.controller.send_command(SimCommand::Step(1));
            }
            PlaybackAction::StepBack | PlaybackAction::ScrubTo(_) => {}
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
        }
    }

    fn draw_plot(&self, parent_ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading(format!(
                "Episodic Return & Bellman Convergence ({})",
                self.solver_mode.label()
            ));

            if self.rewards_per_episode.is_empty() {
                render_empty_plot_guidance(
                    ui,
                    "No episodes trained yet — click '▶ Train ...' or '⏭ Step Episode' in the side panel to populate the reward and Bellman residual curves.",
                );
            }

            let reward_pts: PlotPoints = if self.rewards_per_episode.is_empty() {
                PlotPoints::from(vec![[0.0, 0.0]])
            } else {
                self.rewards_per_episode
                    .iter()
                    .enumerate()
                    .map(|(i, &reward)| [i as f64, reward])
                    .collect()
            };
            let residual_pts: PlotPoints = self
                .bellman_residuals
                .iter()
                .enumerate()
                .map(|(i, &res)| [i as f64, res])
                .collect();

            let show_residual = self.show_bellman_residual && !self.bellman_residuals.is_empty();
            Plot::new("reward_plot")
                .view_aspect(2.0)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Episode Reward", reward_pts));
                    if show_residual {
                        plot_ui.line(Line::new("Mean Bellman Residual |V - V*|", residual_pts));
                    }
                });
        });
    }
}

impl InteractiveTool for RewardPlotsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Reward Plots"
    }
    fn domain(&self) -> &'static str {
        "ai"
    }
    fn state_revision(&self) -> u64 {
        self.revision
    }
    fn save_state(&self) -> Option<String> {
        self.save_preset_state()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.try_load_preset_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        if let Some(snapshot) = self.controller.update() {
            if snapshot.sonification.is_audible() {
                self.latest_audio = snapshot.sonification;
            }
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<RlSnapshotData>())
            {
                self.rewards_per_episode = data.rewards_per_episode.clone();
                self.bellman_residuals = data.bellman_residuals.clone();
                self.episodes_trained = data.episodes_trained;
                self.target_episodes = data.target_episodes;
            }
        }

        if matches!(
            self.controller.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        self.draw_controls(ui, &ctx);
        self.draw_plot(ui);
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RewardPlotsTool",
        display_name: "Reward Plots",
        domain: "ai",
        tags: &[
            "reinforcement learning",
            "rl",
            "bellman",
            "episodic return",
            "discounted reward",
            "convergence",
            "policy",
            "sonification",
        ],
        build: || Box::new(RewardPlotsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RewardPlotsTool {
    fn theory_description(&self) -> String {
        "Episodic cumulative return $G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$ paired with mean Bellman optimality residual $|\\hat{V}(s) - V^*(s)|$ across sculpted windy MDPs.".into()
    }
    fn phonetic_description(&self) -> String {
        "Reinforcement learning episodic reward and Bellman convergence theory".into()
    }
    fn theory_citation(&self) -> String {
        "Reinforcement Learning: An Introduction (Sutton & Barto, 2018)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}
