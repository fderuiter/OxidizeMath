use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{
    SimCommand, SimWorkerState, SimulationController, SimulationRunner, StateSnapshot,
};
use crate::framework::{InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
use math_explorer::ai::reinforcement_learning::{algorithms::TabularQAgent, MarkovDecisionProcess};
use std::sync::Arc;

#[allow(missing_docs)]
pub struct RlSnapshotData {
    pub rewards_per_episode: Vec<f64>,
    pub episodes_trained: usize,
    pub target_episodes: usize,
}

#[allow(missing_docs)]
pub struct RewardPlotsRunner {
    env: GridWorldEnv,
    agent: TabularQAgent<GridState, Move>,
    rewards_per_episode: Vec<f64>,
    episodes_trained: usize,
    target_episodes: usize,
    is_training: bool,
}

impl Default for RewardPlotsRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardPlotsRunner {
    #[allow(missing_docs)]
    pub fn new() -> Self {
        let env = GridWorldEnv {
            width: 5,
            height: 5,
            start: GridState { x: 0, y: 0 },
            goal: GridState { x: 4, y: 4 },
            traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
            gamma: 0.9,
        };
        let agent = TabularQAgent::new(
            math_commons::primitives::UnitInterval::new(0.1).unwrap(),
            math_commons::primitives::UnitInterval::new(0.9).unwrap(),
            math_commons::primitives::UnitInterval::new(0.1).unwrap(),
            None,
        );
        Self {
            env,
            agent,
            rewards_per_episode: Vec::new(),
            episodes_trained: 0,
            target_episodes: 0,
            is_training: false,
        }
    }

    fn train_one_episode(&mut self) {
        let mut current_state = self.env.start;
        let mut episode_reward = 0.0;
        let mut steps = 0;

        while !self.env.is_terminal(&current_state) && steps < 100 {
            let actions = self.env.actions(&current_state);
            if actions.is_empty() {
                break;
            }

            if let Some(action) = self.agent.select_action(&current_state, &actions) {
                let next_state = self.env.step(&current_state, &action);
                let reward = self.env.reward(&current_state, &action, &next_state);
                let next_actions = self.env.actions(&next_state);

                self.agent
                    .update(&current_state, &action, reward, &next_state, &next_actions);

                episode_reward += reward;
                current_state = next_state;
            }
            steps += 1;
        }

        self.rewards_per_episode.push(episode_reward);
        self.episodes_trained += 1;
    }
}

impl SimulationRunner for RewardPlotsRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::TrainEpisodes(count) => {
                self.target_episodes = self.episodes_trained + count;
                self.is_training = true;
            }
            SimCommand::Start => {
                if self.episodes_trained < self.target_episodes {
                    self.is_training = true;
                }
            }
            SimCommand::Pause => {
                self.is_training = false;
            }
            SimCommand::Reset => {
                self.agent = TabularQAgent::new(
                    math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                    math_commons::primitives::UnitInterval::new(0.9).unwrap(),
                    math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                    None,
                );
                self.rewards_per_episode.clear();
                self.episodes_trained = 0;
                self.target_episodes = 0;
                self.is_training = false;
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.is_training && self.episodes_trained < self.target_episodes {
            self.train_one_episode();
            if self.episodes_trained >= self.target_episodes {
                self.is_training = false;
            }
        }
    }

    fn step_once(&mut self) {
        self.train_one_episode();
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let snapshot_data = Box::new(RlSnapshotData {
            rewards_per_episode: self.rewards_per_episode.clone(),
            episodes_trained: self.episodes_trained,
            target_episodes: self.target_episodes,
        });

        StateSnapshot {
            width: 0,
            height: 0,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![self.episodes_trained as f64],
            structured_data: Some(snapshot_data),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        20
    }

    fn is_complete(&self) -> bool {
        !self.is_training || self.episodes_trained >= self.target_episodes
    }

    fn is_finite_state(&self) -> bool {
        self.rewards_per_episode.iter().all(|r| r.is_finite())
    }
}

#[allow(missing_docs)]
pub struct RewardPlotsTool {
    pub(crate) controller: SimulationController,
    rewards_per_episode: Vec<f64>,
    episodes_trained: usize,
    target_episodes: usize,
    training_steps: usize,
}

impl Default for RewardPlotsTool {
    fn default() -> Self {
        let runner = RewardPlotsRunner::new();
        let controller = SimulationController::new(runner);
        Self {
            controller,
            rewards_per_episode: Vec::new(),
            episodes_trained: 0,
            target_episodes: 0,
            training_steps: 100,
        }
    }
}

impl RewardPlotsTool {
    /// Resets the agent in place or respawns a disconnected worker.
    pub fn reset(&mut self) {
        self.controller.reset_or_respawn(RewardPlotsRunner::new);
        self.rewards_per_episode.clear();
        self.episodes_trained = 0;
        self.target_episodes = 0;
    }

    fn draw_controls(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("reward_plots_controls").show(ctx, |ui| {
            ui.heading("Training Controls");
            ui.separator();

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
            ui.add(
                egui::Slider::new(&mut self.training_steps, 10..=1000).text("Episodes to Train"),
            );

            if state.is_running() {
                let remaining = self.target_episodes.saturating_sub(self.episodes_trained);
                let progress = if self.target_episodes > 0 {
                    (self.episodes_trained as f32 / self.target_episodes as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(progress)
                        .text(format!("Training: {remaining} remaining")),
                );
            }

            let has_unfinished =
                self.episodes_trained < self.target_episodes && self.target_episodes > 0;
            let play_label = if has_unfinished {
                "▶ Resume".to_string()
            } else {
                format!("▶ Train {}", self.training_steps)
            };

            if let Some(action) = PlaybackTransport::from_worker_state(
                "reward_plots_transport",
                "Reward Plots",
                state,
            )
            .play_label(&play_label)
            .step_label("⏭ Step Episode")
            .can_reset(true)
            .show(ui)
            {
                self.handle_transport_action(action, has_unfinished, ctx);
            }
        });
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
            PlaybackAction::Reset => {
                self.reset();
                ctx.request_repaint();
            }
        }
    }

    fn draw_plot(&self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cumulative Reward over Episodes");

            if self.rewards_per_episode.is_empty() {
                render_empty_plot_guidance(
                    ui,
                    "No episodes trained yet — click '▶ Train ...' or '⏭ Step Episode' in the side panel to populate the reward curve.",
                );
            }

            let points: PlotPoints = self
                .rewards_per_episode
                .iter()
                .enumerate()
                .map(|(i, &reward)| [i as f64, reward])
                .collect();

            Plot::new("reward_plot")
                .view_aspect(2.0)
                .legend(egui_plot::Legend::default())
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Episode Reward", points));
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

    fn show(&mut self, ctx: &egui::Context) {
        if let Some(snapshot) = self.controller.update() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<RlSnapshotData>())
            {
                self.rewards_per_episode = data.rewards_per_episode.clone();
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

        self.draw_controls(ctx);
        self.draw_plot(ctx);
    }
}

// [cite:modular_polynomials_review]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "RewardPlotsTool",
        display_name: "Reward Plots",
        domain: "ai",
        tags: &[],
        build: || Box::new(RewardPlotsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RewardPlotsTool {
    fn theory_description(&self) -> String {
        "Reinforcement learning episodic return: cumulative discounted reward $G_t = \\sum_{k=0}^{\\infty} \\gamma^k R_{t+k+1}$, illustrating policy convergence and variance reduction across training episodes.".into()
    }
    fn phonetic_description(&self) -> String {
        "Reinforcement learning episodic reward and return theory".into()
    }
    fn theory_citation(&self) -> String {
        "Reinforcement Learning: An Introduction (Sutton & Barto, 2018)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Reward Trajectory",
    "domain": "ai",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "episodes": 200,
    "smoothing_window": 10
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "reward_plots_starter",
        title: "Reward Trajectory",
        description: "Episode cumulative return and moving average reward learning dynamics across training episodes.",
        category: "Reinforcement Learning",
        icon: "🎯",
        domain: "ai",
        tool_name: Some("RewardPlotsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
