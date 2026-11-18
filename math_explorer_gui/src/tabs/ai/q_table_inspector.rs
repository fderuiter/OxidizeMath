use crate::accessibility::AccessibleHoverText;
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{InputMode, InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::algorithms::TabularQAgent;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
#[cfg(test)]
use math_explorer::ai::reinforcement_learning::MarkovDecisionProcess;
use std::collections::HashMap;

#[path = "q_table_inspector/runner.rs"]
mod runner;
pub use runner::*;

#[derive(Clone, Copy, PartialEq)]
enum HeatmapView {
    MaxQ,
    Up,
    Down,
    Left,
    Right,
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

    fn reset_agent(&mut self) {
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

    fn draw_controls(&mut self, ctx: &egui::Context, legend_size: f32) {
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

            if let Some(action) = PlaybackTransport::from_worker_state(
                "q_table_transport",
                "Q-Table Inspector",
                self.worker.state(),
            )
            .play_label("▶ Train 100")
            .step_label("⏭ Step Episode")
            .can_reset(true)
            .show(ui)
            {
                match action {
                    PlaybackAction::Play => {
                        let _ = self.worker.send_command(SimCommand::TrainEpisodes(100));
                    }
                    PlaybackAction::Pause => {
                        let _ = self.worker.send_command(SimCommand::Pause);
                    }
                    PlaybackAction::Step => {
                        let _ = self.worker.send_command(SimCommand::Step(1));
                    }
                    PlaybackAction::Reset => {
                        self.reset_agent();
                    }
                }
            }

            if ui
                .add_enabled(
                    self.worker.state().is_ready(),
                    egui::Button::new("▶ Train 1000 Episodes"),
                )
                .accessible_hover_text("Train the agent for 1000 episodes in the background")
                .clicked()
            {
                let _ = self.worker.send_command(SimCommand::TrainEpisodes(1000));
            }

            self.draw_view_and_legend(ui, legend_size);
        });
    }

    fn draw_view_and_legend(&mut self, ui: &mut egui::Ui, legend_size: f32) {
        ui.separator();
        ui.heading("Heatmap View");
        ui.radio_value(&mut self.view, HeatmapView::MaxQ, "Max Q-Value");
        ui.radio_value(&mut self.view, HeatmapView::Up, "Move Up Q-Value");
        ui.radio_value(&mut self.view, HeatmapView::Down, "Move Down Q-Value");
        ui.radio_value(&mut self.view, HeatmapView::Left, "Move Left Q-Value");
        ui.radio_value(&mut self.view, HeatmapView::Right, "Move Right Q-Value");

        ui.separator();
        ui.label("Legend:");
        for (label, color) in [
            ("Goal:", egui::Color32::GREEN),
            ("Trap:", egui::Color32::RED),
        ] {
            ui.horizontal(|ui| {
                ui.label(label);
                let (resp, painter) =
                    ui.allocate_painter(egui::vec2(legend_size, legend_size), egui::Sense::hover());
                painter.rect_filled(resp.rect, 0.0, color);
            });
        }
    }

    fn q_value_bounds(&self) -> (f64, f64) {
        let mut min_q = 0.0_f64;
        let mut max_q = 0.0_f64;
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let q = self.get_q_value_for_view(&GridState { x, y });
                min_q = min_q.min(q);
                max_q = max_q.max(q);
            }
        }
        if (max_q - min_q).abs() < f64::EPSILON {
            max_q = min_q + 1.0;
        }
        (min_q, max_q)
    }

    fn draw_heatmap(&self, ctx: &egui::Context, cell_size: f32) {
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
            let (min_q, max_q) = self.q_value_bounds();

            for x in 0..self.env.width {
                for y in 0..self.env.height {
                    let top_left = rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                    let cell_rect =
                        egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));
                    let state = GridState { x, y };
                    let q_value = self.get_q_value_for_view(&state);

                    let fill_color = if state == self.env.goal {
                        egui::Color32::GREEN
                    } else if self.env.traps.contains(&state) {
                        egui::Color32::RED
                    } else {
                        let norm = (q_value - min_q) / (max_q - min_q);
                        let c = (norm * 255.0) as u8;
                        egui::Color32::from_rgb(c, c, ((1.0 - norm) * 255.0) as u8)
                    };

                    painter.rect_filled(cell_rect, 0.0, fill_color);
                    painter.rect_stroke(
                        cell_rect,
                        0.0,
                        egui::Stroke::new(1.0_f32, egui::Color32::BLACK),
                        egui::StrokeKind::Middle,
                    );

                    if state != self.env.goal && !self.env.traps.contains(&state) {
                        painter.text(
                            cell_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            format!("{q_value:.2}"),
                            egui::FontId::proportional(14.0),
                            egui::Color32::BLACK,
                        );
                    }
                }
            }
        });
    }
}

impl InteractiveTool for QTableInspectorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Q-Table Inspector"
    }

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
        if matches!(
            self.worker.state(),
            SimWorkerState::Running | SimWorkerState::Stepping
        ) {
            ctx.request_repaint();
        }

        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let touch_min = 44.0_f32;
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

        self.draw_controls(ctx, legend_size);
        self.draw_heatmap(ctx, cell_size);
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
    fn phonetic_description(&self) -> String {
        "Q-learning and Bellman optimality equation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Reinforcement Learning: An Introduction (Sutton & Barto, 2018)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

#[cfg(test)]
#[path = "q_table_inspector/tests.rs"]
mod tests;
#[path = "q_table_inspector/preset.rs"]
mod preset;
