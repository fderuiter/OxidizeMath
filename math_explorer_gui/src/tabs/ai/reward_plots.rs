use crate::accessibility::AccessibleHoverText;
use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
use math_explorer::ai::reinforcement_learning::{algorithms::TabularQAgent, MarkovDecisionProcess};

pub struct RewardPlotsTool {
    env: GridWorldEnv,
    agent: TabularQAgent<GridState, Move>,
    rewards_per_episode: Vec<f64>,
    episodes_trained: usize,
    training_steps: usize,
}

impl Default for RewardPlotsTool {
    fn default() -> Self {
        let env = GridWorldEnv {
            width: 5,
            height: 5,
            start: GridState { x: 0, y: 0 },
            goal: GridState { x: 4, y: 4 },
            traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
            gamma: 0.9,
        };
        let agent = TabularQAgent::new(math_commons::primitives::UnitInterval::new(0.1).unwrap(), math_commons::primitives::UnitInterval::new(0.9).unwrap(), math_commons::primitives::UnitInterval::new(0.1).unwrap(), None);
        Self {
            env,
            agent,
            rewards_per_episode: Vec::new(),
            episodes_trained: 0,
            training_steps: 100, // default number of episodes to train at once
        }
    }
}

impl RewardPlotsTool {
    fn train_episodes(&mut self, num_episodes: usize) {
        for _ in 0..num_episodes {
            let mut current_state = self.env.start;
            let mut episode_reward = 0.0;
            let mut steps = 0;

            // Cap steps per episode to prevent infinite loops if agent gets stuck early on
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
}

impl InteractiveTool for RewardPlotsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Reward Plots"
    }

    fn export_data(&self) -> Option<crate::framework::ExportableData> {
        let pts: Vec<[f64; 2]> = self
            .rewards_per_episode
            .iter()
            .enumerate()
            .map(|(idx, &r)| [idx as f64 + 1.0, r])
            .collect();
        Some(crate::framework::ExportableData::single_series(
            "Reinforcement Learning Rewards",
            vec!["Episode".into(), "Total Reward".into()],
            "Episode Reward",
            pts,
        ))
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("reward_plots_controls").show(ctx, |ui| {
            ui.heading("Training Controls");
            ui.separator();

            ui.label(format!("Total Episodes: {}", self.episodes_trained));

            ui.add(
                egui::Slider::new(&mut self.training_steps, 10..=1000).text("Episodes to Train"),
            );

            if ui
                .button(format!("▶ Train {} Episodes", self.training_steps))
                .accessible_hover_text("Train the agent for the specified number of episodes")
                .clicked()
            {
                self.train_episodes(self.training_steps);
            }

            if ui
                .button("↻ Reset Agent")
                .accessible_hover_text(
                    "Clear the Q-table and reset the agent's knowledge and reward history",
                )
                .clicked()
            {
                self.agent = TabularQAgent::new(math_commons::primitives::UnitInterval::new(0.1).unwrap(), math_commons::primitives::UnitInterval::new(0.9).unwrap(), math_commons::primitives::UnitInterval::new(0.1).unwrap(), None);
                self.rewards_per_episode.clear();
                self.episodes_trained = 0;
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Cumulative Reward over Episodes");

            let points: PlotPoints = self
                .rewards_per_episode
                .iter()
                .enumerate()
                .map(|(i, &reward)| [i as f64, reward])
                .collect();

            let line = Line::new("Episode Reward", points);

            let plot = Plot::new("reward_plot")
                .view_aspect(2.0)
                .legend(egui_plot::Legend::default());

            plot.show(ui, |plot_ui| {
                plot_ui.line(line);
            });
        });
    }
}

// [cite:modular_polynomials_review]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "RewardPlotsTool",
        domain: "ai",
        tags: &[],
        build: || Box::new(RewardPlotsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for RewardPlotsTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
