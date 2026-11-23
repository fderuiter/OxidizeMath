use crate::accessibility::AccessibleHoverText;
use crate::framework::{InputMode, InteractiveTool};
use eframe::egui;
use math_commons::primitives::UnitInterval;
use math_explorer::ai::reinforcement_learning::{
    algorithms::TabularQAgent,
    grid_world::{GridState, GridWorldEnv, Move},
    q_function::TabularQFunction,
    strategies::EpsilonGreedy,
    MarkovDecisionProcess,
};

pub struct GridWorldTool {
    env: GridWorldEnv,
    agent: TabularQAgent<GridState, Move>,
    learning_rate: f64,
    discount_factor: f64,
    exploration_rate: f64,
    current_state: GridState,
    episodes: u32,
    total_reward: f64,
    steps: u32,
    is_playing: bool,
    playback_speed: f32,
    last_tick: crate::web_io::PlatformInstant,
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

    fn sync_agent_hyperparameters(&mut self) {
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

    fn reset_agent(&mut self) {
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

    fn train_batch(&mut self) {
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

    fn cell_fill(&self, state: GridState, palette: &crate::app::SemanticPalette) -> egui::Color32 {
        if state == self.env.goal {
            palette.success
        } else if self.env.traps.contains(&state) {
            palette.error
        } else if state == self.current_state {
            palette.accent
        } else if state == self.env.start {
            palette.info
        } else {
            palette.surface_fill
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "grid_world_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Controls");
            ui.separator();
            ui.horizontal_wrapped(|ui| {
                if ui
                    .button("▶ Step")
                    .accessible_hover_text("Advance the agent by one step")
                    .clicked()
                {
                    self.step_agent();
                }
                let (label, hover) = if self.is_playing {
                    ("⏸ Pause", "Pause continuous Q-learning playback")
                } else {
                    ("▶ Play", "Start continuous Q-learning playback")
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
                .accessible_hover_text("Clear the Q-table and reset the agent's knowledge")
                .clicked()
            {
                self.reset_agent();
            }
            ui.separator();
            self.draw_hyperparameter_sliders(ui);
        });
    }

    fn draw_hyperparameter_sliders(&mut self, ui: &mut egui::Ui) {
        ui.heading("RL Hyperparameters");
        let mut hp_changed = ui
            .add(egui::Slider::new(&mut self.learning_rate, 0.01..=1.0).text("Learning Rate (α)"))
            .changed();
        hp_changed |= ui
            .add(
                egui::Slider::new(&mut self.discount_factor, 0.0..=1.0).text("Discount Factor (γ)"),
            )
            .changed();
        hp_changed |= ui
            .add(
                egui::Slider::new(&mut self.exploration_rate, 0.0..=1.0)
                    .text("Exploration Rate (ε)"),
            )
            .changed();
        if hp_changed {
            self.sync_agent_hyperparameters();
        }
        ui.add(egui::Slider::new(&mut self.playback_speed, 1.0..=20.0).text("Speed (Hz)"));
    }

    fn draw_central_canvas(&self, parent_ui: &mut egui::Ui, input_mode: InputMode) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Grid World Navigation (Q-Learning)");
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("Episode: {}", self.episodes));
                ui.label(format!("Steps: {}", self.steps));
                ui.label(format!("Reward: {:.2}", self.total_reward));
            });
            let cell_size = if input_mode == InputMode::Touch {
                44.0
            } else {
                40.0
            };
            let grid_size = egui::vec2(
                self.env.width as f32 * cell_size,
                self.env.height as f32 * cell_size,
            );
            let (response, painter) =
                ui.allocate_painter(grid_size, egui::Sense::focusable_noninteractive());
            let _ = response
                .clone()
                .accessible_hover_text("Grid World Visualization");
            let summary = format!(
                "Agent at ({}, {}), Episode {}, Reward {:.2}",
                self.current_state.x, self.current_state.y, self.episodes, self.total_reward
            );
            crate::accessibility::register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                "Grid World Visualization",
                Some(&summary),
            );
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            for x in 0..self.env.width {
                for y in 0..self.env.height {
                    let top_left =
                        response.rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                    let cell_rect =
                        egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));
                    painter.rect_filled(
                        cell_rect,
                        0.0,
                        self.cell_fill(GridState { x, y }, &palette),
                    );
                    painter.rect_stroke(
                        cell_rect,
                        0.0,
                        egui::Stroke::new(1.0_f32, palette.graph_edge),
                        egui::StrokeKind::Middle,
                    );
                }
            }
            crate::accessibility::paint_canvas_focus_ring(ui, &response);
        });
    }
}

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

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&GridWorldToolState {
            width: self.env.width,
            height: self.env.height,
            gamma: self.discount_factor,
            learning_rate: self.learning_rate,
            exploration_rate: self.exploration_rate,
            playback_speed: self.playback_speed,
            is_playing: self.is_playing,
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
        let start = GridState { x: 0, y: 0 };
        let goal = GridState {
            x: parsed.width - 1,
            y: parsed.height - 1,
        };
        let traps = vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }]
            .into_iter()
            .filter(|t| t.x < parsed.width && t.y < parsed.height && *t != start && *t != goal)
            .collect();
        self.learning_rate = lr_unit.value();
        self.discount_factor = gamma_unit.value();
        self.exploration_rate = eps_unit.value();
        self.env = GridWorldEnv {
            width: parsed.width,
            height: parsed.height,
            start,
            goal,
            traps,
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

impl scientific_metadata::theory::TheoryDescribable for GridWorldTool {
    fn theory_description(&self) -> String {
        format!(
            "Tabular Q-Learning updates state-action values via Bellman temporal difference $Q(s,a) \\leftarrow Q(s,a) + \\alpha [R + \\gamma \\max_{{a'}} Q(s',a') - Q(s,a)]$ with $\\alpha = {:.2}$, $\\gamma = {:.2}$, and $\\epsilon = {:.2}$.",
            self.learning_rate, self.discount_factor, self.exploration_rate
        )
    }
    fn phonetic_description(&self) -> String {
        "Reinforcement Learning involves an agent that interacts with its environment to learn optimal actions, aiming to maximize a cumulative reward based on the Bellman equation.".into()
    }
    fn theory_citation(&self) -> String {
        "Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction. MIT press."
            .into()
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
mod tests;

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
