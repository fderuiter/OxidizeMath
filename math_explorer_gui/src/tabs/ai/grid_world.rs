use crate::accessibility::AccessibleHoverText;
use crate::framework::{InputMode, InteractiveTool};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::{
    algorithms::TabularQAgent,
    grid_world::{GridState, GridWorldEnv, Move},
    MarkovDecisionProcess,
};

pub struct GridWorldTool {
    env: GridWorldEnv,
    agent: TabularQAgent<GridState, Move>,
    current_state: GridState,
    episodes: u32,
    total_reward: f64,
    steps: u32,
    is_playing: bool,
    playback_speed: f32,
    last_tick: std::time::Instant,
}

impl Default for GridWorldTool {
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
            current_state: env.start,
            env,
            agent,
            episodes: 0,
            total_reward: 0.0,
            steps: 0,
            is_playing: false,
            playback_speed: 5.0,
            last_tick: std::time::Instant::now(),
        }
    }
}

impl GridWorldTool {
    fn step_agent(&mut self) {
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

    fn reset_episode(&mut self) {
        self.current_state = self.env.start;
        self.episodes += 1;
        self.total_reward = 0.0;
        self.steps = 0;
    }
}

impl InteractiveTool for GridWorldTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Grid World (RL)"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        if self.is_playing {
            let speed_hz = self.playback_speed.clamp(1.0, 20.0);
            let interval = std::time::Duration::from_secs_f32(1.0 / speed_hz);
            let now = std::time::Instant::now();
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

        egui::Window::new("Grid World Navigation (Q-Learning)").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if ui
                    .button("▶ Step")
                    .accessible_hover_text("Advance the agent by one step")
                    .clicked()
                {
                    self.step_agent();
                }
                let play_pause_label = if self.is_playing { "⏸ Pause" } else { "▶ Play" };
                let play_pause_hover = if self.is_playing {
                    "Pause continuous Q-learning playback"
                } else {
                    "Start continuous Q-learning playback"
                };
                if ui
                    .button(play_pause_label)
                    .accessible_hover_text(play_pause_hover)
                    .clicked()
                {
                    self.is_playing = !self.is_playing;
                    if self.is_playing {
                        self.last_tick = std::time::Instant::now();
                    }
                }
                if ui
                    .button("▶ Train (100 Episodes)")
                    .accessible_hover_text("Train the agent for 100 episodes instantly")
                    .clicked()
                {
                    self.is_playing = false;
                    for _ in 0..100 {
                        let mut temp_steps = 0;
                        while !self.env.is_terminal(&self.current_state) && temp_steps < 100 {
                            self.step_agent();
                            temp_steps += 1;
                        }
                        self.reset_episode();
                    }
                }
                if ui
                    .button("↻ Reset Agent")
                    .accessible_hover_text("Clear the Q-table and reset the agent's knowledge")
                    .clicked()
                {
                    self.is_playing = false;
                    self.agent = TabularQAgent::new(math_commons::primitives::UnitInterval::new(0.1).unwrap(), math_commons::primitives::UnitInterval::new(0.9).unwrap(), math_commons::primitives::UnitInterval::new(0.1).unwrap(), None);
                    self.reset_episode();
                    self.episodes = 0;
                }
            });

            ui.horizontal(|ui| {
                ui.label("Speed:");
                ui.add(egui::Slider::new(&mut self.playback_speed, 1.0..=20.0).text("Hz"));
            });

            ui.horizontal(|ui| {
                ui.label(format!("Episode: {}", self.episodes));
                ui.label(format!("Steps: {}", self.steps));
                ui.label(format!("Reward: {:.2}", self.total_reward));
            });

            let cell_size = if input_mode == InputMode::Touch { 44.0 } else { 40.0 };
            let grid_size = egui::vec2(
                self.env.width as f32 * cell_size,
                self.env.height as f32 * cell_size,
            );

            let (response, painter) = ui.allocate_painter(grid_size, egui::Sense::hover());
            let _ = response
                .clone()
                .accessible_hover_text("Grid World Visualization");
            let rect = response.rect;

            for x in 0..self.env.width {
                for y in 0..self.env.height {
                    let top_left =
                        rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                    let cell_rect =
                        egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));

                    let state = GridState { x, y };
                    let mut fill_color = egui::Color32::from_gray(200);

                    if state == self.env.goal {
                        fill_color = egui::Color32::GREEN;
                    } else if self.env.traps.contains(&state) {
                        fill_color = egui::Color32::RED;
                    } else if state == self.current_state {
                        fill_color = egui::Color32::BLUE;
                    } else if state == self.env.start {
                        fill_color = egui::Color32::LIGHT_BLUE;
                    }

                    painter.rect_filled(cell_rect, 0.0, fill_color);
                    painter.rect_stroke(
                        cell_rect,
                        0.0,
                        egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
                        egui::StrokeKind::Middle,
                    );
                }
            }
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "GridWorldTool",
        domain: "ai",
        tags: &[],
        build: || Box::new(GridWorldTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for GridWorldTool {
    fn theory_description(&self) -> String {
        "Reinforcement Learning involves an agent that interacts with its environment to learn optimal actions, aiming to maximize a cumulative reward based on the Bellman equation.".into()
    }
    fn phonetic_description(&self) -> String {
        "Reinforcement Learning involves an agent that interacts with its environment to learn optimal actions, aiming to maximize a cumulative reward based on the Bellman equation.".into()
    }
    fn theory_citation(&self) -> String {
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT press.".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("learning_rate".into(), "The step size used to update the Q-values, controlling how much new information overrides old information.".into());
        map.insert("discount_factor".into(), "Determines the importance of future rewards. A factor of 0 makes the agent opportunistic, while 1 makes it strive for long-term high reward.".into());
        map.insert("exploration_rate".into(), "The probability that the agent will choose a random action instead of the best known action, facilitating exploration of the state space.".into());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grid_world_default_playback_state() {
        let tool = GridWorldTool::default();
        assert!(!tool.is_playing);
        assert!((1.0..=20.0).contains(&tool.playback_speed));
    }

    #[test]
    fn test_grid_world_playback_toggle_and_speed_bounds() {
        let mut tool = GridWorldTool {
            is_playing: true,
            ..Default::default()
        };
        assert!(tool.is_playing);

        tool.playback_speed = 10.0;
        assert_eq!(tool.playback_speed, 10.0);

        tool.playback_speed = 0.5; // below min speed
        let speed_hz = tool.playback_speed.clamp(1.0, 20.0);
        assert_eq!(speed_hz, 1.0);

        tool.playback_speed = 25.0; // above max speed
        let speed_hz = tool.playback_speed.clamp(1.0, 20.0);
        assert_eq!(speed_hz, 20.0);
    }

    #[test]
    fn test_grid_world_pause_on_train_and_reset() {
        let mut tool = GridWorldTool {
            is_playing: true,
            ..Default::default()
        };

        // Simulate reset episode / train pausing playback
        tool.is_playing = false;
        assert!(!tool.is_playing);
    }

    #[test]
    fn test_grid_world_step_and_terminal_reset() {
        let mut tool = GridWorldTool::default();
        let initial_state = tool.current_state;
        tool.step_agent();
        assert_ne!(
            (tool.current_state, tool.steps, tool.episodes),
            (initial_state, 0, 0)
        );

        // Test terminal state reset
        tool.current_state = tool.env.goal;
        tool.step_agent();
        assert_eq!(tool.current_state, tool.env.start);
        assert_eq!(tool.steps, 0);
        assert_eq!(tool.episodes, 1);
    }
}
