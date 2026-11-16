use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{BackgroundWorker, SimCommand, SimulationRunner, StateSnapshot};
use crate::framework::{InputMode, InteractiveTool};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
use math_explorer::ai::reinforcement_learning::{algorithms::TabularQAgent, MarkovDecisionProcess};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, PartialEq)]
enum HeatmapView {
    MaxQ,
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone)]
#[allow(missing_docs)]
pub struct QTableSnapshot {
    pub episodes_trained: u32,
    pub q_values: HashMap<(GridState, Move), f64>,
}

#[allow(missing_docs)]
pub struct QTableRunner {
    pub env: GridWorldEnv,
    pub agent: TabularQAgent<GridState, Move>,
    pub episodes_trained: u32,
    pub target_episodes: u32,
    pub is_training: bool,
}

impl SimulationRunner for QTableRunner {
    fn process_command(&mut self, cmd: SimCommand) {
        match cmd {
            SimCommand::TrainEpisodes(count) => {
                self.target_episodes = self.episodes_trained + count as u32;
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
                self.episodes_trained = 0;
                self.target_episodes = 0;
                self.is_training = false;
            }
            _ => {}
        }
    }

    fn step(&mut self) {
        if self.is_training && self.episodes_trained < self.target_episodes {
            let batch = (self.target_episodes - self.episodes_trained).min(50);
            for _ in 0..batch {
                let mut current_state = self.env.start;
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

                        current_state = next_state;
                    }
                    steps += 1;
                }
                self.episodes_trained += 1;
            }
            if self.episodes_trained >= self.target_episodes {
                self.is_training = false;
            }
        }
    }

    fn get_snapshot(&self) -> StateSnapshot {
        let mut q_values = HashMap::new();
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                for action in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    let q = self.agent.get_q_value(&state, &action);
                    q_values.insert((state, action), q);
                }
            }
        }
        let snapshot = QTableSnapshot {
            episodes_trained: self.episodes_trained,
            q_values,
        };
        StateSnapshot {
            width: self.env.width as usize,
            height: self.env.height as usize,
            pixels: Arc::new(std::sync::RwLock::new(Vec::new())),
            custom_data: vec![self.episodes_trained as f64],
            structured_data: Some(Box::new(snapshot)),
        }
    }

    fn get_steps_per_frame(&self) -> usize {
        20
    }

    fn is_complete(&self) -> bool {
        !self.is_training || self.episodes_trained >= self.target_episodes
    }
}

pub struct QTableInspectorTool {
    env: GridWorldEnv,
    agent: TabularQAgent<GridState, Move>,
    cached_q_values: HashMap<(GridState, Move), f64>,
    episodes_trained: u32,
    view: HeatmapView,
    worker: BackgroundWorker<QTableRunner>,
}

impl Default for QTableInspectorTool {
    fn default() -> Self {
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
        let runner = QTableRunner {
            env: GridWorldEnv {
                width: 5,
                height: 5,
                start: GridState { x: 0, y: 0 },
                goal: GridState { x: 4, y: 4 },
                traps: vec![GridState { x: 2, y: 2 }, GridState { x: 3, y: 2 }],
                gamma: 0.9,
            },
            agent: TabularQAgent::new(
                math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                math_commons::primitives::UnitInterval::new(0.9).unwrap(),
                math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                None,
            ),
            episodes_trained: 0,
            target_episodes: 0,
            is_training: false,
        };
        let worker = BackgroundWorker::new(runner);
        Self {
            env,
            agent,
            cached_q_values: HashMap::new(),
            episodes_trained: 0,
            view: HeatmapView::MaxQ,
            worker,
        }
    }
}

impl QTableInspectorTool {
    #[cfg(test)]
    pub fn train(&mut self, episodes: u32) {
        for _ in 0..episodes {
            let mut current_state = self.env.start;
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

                    current_state = next_state;
                }
                steps += 1;
            }
        }
        self.episodes_trained += episodes;
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let state = GridState { x, y };
                for action in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    self.cached_q_values
                        .insert((state, action), self.agent.get_q_value(&state, &action));
                }
            }
        }
    }

    fn get_q(&self, state: &GridState, action: &Move) -> f64 {
        self.cached_q_values
            .get(&(*state, *action))
            .copied()
            .unwrap_or_else(|| self.agent.get_q_value(state, action))
    }

    fn get_q_value_for_view(&self, state: &GridState) -> f64 {
        match self.view {
            HeatmapView::MaxQ => {
                let mut max_q = f64::NEG_INFINITY;
                for action in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    let q = self.get_q(state, &action);
                    if q > max_q {
                        max_q = q;
                    }
                }
                if max_q == f64::NEG_INFINITY {
                    0.0
                } else {
                    max_q
                }
            }
            HeatmapView::Up => self.get_q(state, &Move::Up),
            HeatmapView::Down => self.get_q(state, &Move::Down),
            HeatmapView::Left => self.get_q(state, &Move::Left),
            HeatmapView::Right => self.get_q(state, &Move::Right),
        }
    }
}

impl InteractiveTool for QTableInspectorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Q-Table Inspector"
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        if let Some(snapshot) = self.worker.update() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<QTableSnapshot>())
            {
                self.cached_q_values = data.q_values.clone();
                self.episodes_trained = data.episodes_trained;
            }
        }
        if self.worker.is_running() {
            ctx.request_repaint();
        }

        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let touch_min = 44.0;
        let legend_size = if input_mode == InputMode::Touch {
            15.0_f32.max(touch_min)
        } else {
            15.0
        };
        let cell_size = if input_mode == InputMode::Touch {
            60.0_f32.max(touch_min)
        } else {
            60.0
        };

        egui::SidePanel::left("q_table_controls").show(ctx, |ui| {
            ui.heading("Training Controls");
            ui.separator();

            ui.label(format!("Episodes Trained: {}", self.episodes_trained));
            if self.worker.is_running() {
                ui.colored_label(
                    egui::Color32::from_rgb(60, 160, 240),
                    "⏳ Training in background...",
                );
            }

            if ui
                .button("▶ Train 100 Episodes")
                .accessible_hover_text("Train the agent for 100 episodes in the background")
                .clicked()
            {
                let _ = self.worker.send_command(SimCommand::TrainEpisodes(100));
            }
            if ui
                .button("▶ Train 1000 Episodes")
                .accessible_hover_text("Train the agent for 1000 episodes in the background")
                .clicked()
            {
                let _ = self.worker.send_command(SimCommand::TrainEpisodes(1000));
            }
            if ui
                .button("↻ Reset Agent")
                .accessible_hover_text("Clear the Q-table and reset the agent's knowledge")
                .clicked()
            {
                let _ = self.worker.send_command(SimCommand::Reset);
                self.agent = TabularQAgent::new(
                    math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                    math_commons::primitives::UnitInterval::new(0.9).unwrap(),
                    math_commons::primitives::UnitInterval::new(0.1).unwrap(),
                    None,
                );
                self.cached_q_values.clear();
                self.episodes_trained = 0;
            }

            ui.separator();
            ui.heading("Heatmap View");
            ui.radio_value(&mut self.view, HeatmapView::MaxQ, "Max Q-Value");
            ui.radio_value(&mut self.view, HeatmapView::Up, "Move Up Q-Value");
            ui.radio_value(&mut self.view, HeatmapView::Down, "Move Down Q-Value");
            ui.radio_value(&mut self.view, HeatmapView::Left, "Move Left Q-Value");
            ui.radio_value(&mut self.view, HeatmapView::Right, "Move Right Q-Value");

            ui.separator();
            ui.label("Legend:");
            ui.horizontal(|ui| {
                ui.label("Goal:");
                let (response, painter) =
                    ui.allocate_painter(egui::vec2(legend_size, legend_size), egui::Sense::hover());
                painter.rect_filled(response.rect, 0.0, egui::Color32::GREEN);
            });
            ui.horizontal(|ui| {
                ui.label("Trap:");
                let (response, painter) =
                    ui.allocate_painter(egui::vec2(legend_size, legend_size), egui::Sense::hover());
                painter.rect_filled(response.rect, 0.0, egui::Color32::RED);
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Q-Value Heatmap");

            let grid_size = egui::vec2(
                self.env.width as f32 * cell_size,
                self.env.height as f32 * cell_size,
            );

            let (response, painter) = ui.allocate_painter(grid_size, egui::Sense::hover());
            let _ = response
                .clone()
                .accessible_hover_text("Q-Table Heatmap Visualization");
            let rect = response.rect;

            // Determine min and max Q-values across the board for current view to normalize colors
            let mut min_q = 0.0;
            let mut max_q = 0.0;

            for x in 0..self.env.width {
                for y in 0..self.env.height {
                    let state = GridState { x, y };
                    let q = self.get_q_value_for_view(&state);
                    if q < min_q {
                        min_q = q;
                    }
                    if q > max_q {
                        max_q = q;
                    }
                }
            }

            // To avoid division by zero when min == max
            if (max_q - min_q).abs() < f64::EPSILON {
                max_q = min_q + 1.0;
            }

            for x in 0..self.env.width {
                for y in 0..self.env.height {
                    let top_left =
                        rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                    let cell_rect =
                        egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));

                    let state = GridState { x, y };
                    let q_value = self.get_q_value_for_view(&state);

                    let fill_color = if state == self.env.goal {
                        egui::Color32::GREEN
                    } else if self.env.traps.contains(&state) {
                        egui::Color32::RED
                    } else {
                        // Normalize Q value to 0..1 for coloring
                        let normalized = (q_value - min_q) / (max_q - min_q);

                        // Map 0 to low-value color (e.g. blue), 1 to high-value color (e.g. yellow)
                        let r = (normalized * 255.0) as u8;
                        let g = (normalized * 255.0) as u8;
                        let b = ((1.0 - normalized) * 255.0) as u8;

                        egui::Color32::from_rgb(r, g, b)
                    };

                    painter.rect_filled(cell_rect, 0.0, fill_color);
                    painter.rect_stroke(
                        cell_rect,
                        0.0,
                        egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
                        egui::StrokeKind::Middle,
                    );

                    // Display Q-value text
                    if state != self.env.goal && !self.env.traps.contains(&state) {
                        let text_color = egui::Color32::BLACK; // or white depending on background
                        painter.text(
                            cell_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            format!("{:.2}", q_value),
                            egui::FontId::proportional(14.0),
                            text_color,
                        );
                    }
                }
            }
        });
    }
}

// [cite:graph_parameters_rust]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "QTableInspectorTool",
        domain: "ai",
        tags: &[],
        build: || Box::new(QTableInspectorTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for QTableInspectorTool {
    fn theory_description(&self) -> String {
        "Q-learning temporal difference control: $Q(s, a) \\leftarrow Q(s, a) + \\alpha \\left[ r + \\gamma \\max_{a'} Q(s', a') - Q(s, a) \\right]$, converging to the optimal action-value function satisfying the Bellman optimality equation.".into()
    }
    fn phonetic_description(&self) -> String { "Q-learning and Bellman optimality equation theory".into() }
    fn theory_citation(&self) -> String { "Reinforcement Learning: An Introduction (Sutton & Barto, 2018)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

#[cfg(test)]
#[path = "q_table_inspector/tests.rs"]
mod tests;

