use crate::accessibility::AccessibleHoverText;
use crate::framework::{InputMode, InteractiveTool};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
use math_explorer::ai::reinforcement_learning::{algorithms::TabularQAgent, MarkovDecisionProcess};

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
    episodes_trained: u32,
    view: HeatmapView,
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
        let agent = TabularQAgent::new(math_commons::primitives::UnitInterval::new(0.1).unwrap(), math_commons::primitives::UnitInterval::new(0.9).unwrap(), math_commons::primitives::UnitInterval::new(0.1).unwrap(), None);
        Self {
            env,
            agent,
            episodes_trained: 0,
            view: HeatmapView::MaxQ,
        }
    }
}

impl QTableInspectorTool {
    fn train(&mut self, episodes: u32) {
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
    }

    fn get_q_value_for_view(&self, state: &GridState) -> f64 {
        match self.view {
            HeatmapView::MaxQ => {
                let mut max_q = f64::NEG_INFINITY;
                for action in [Move::Up, Move::Down, Move::Left, Move::Right] {
                    let q = self.agent.get_q_value(state, &action);
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
            HeatmapView::Up => self.agent.get_q_value(state, &Move::Up),
            HeatmapView::Down => self.agent.get_q_value(state, &Move::Down),
            HeatmapView::Left => self.agent.get_q_value(state, &Move::Left),
            HeatmapView::Right => self.agent.get_q_value(state, &Move::Right),
        }
    }
}

impl InteractiveTool for QTableInspectorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Q-Table Inspector"
    }

    fn export_data(&self) -> Option<crate::framework::ExportableData> {
        let mut row_headers = Vec::new();
        let col_headers = vec!["Up".into(), "Down".into(), "Left".into(), "Right".into()];
        let mut matrix = Vec::new();

        for y in 0..self.env.height {
            for x in 0..self.env.width {
                let state = GridState { x, y };
                row_headers.push(format!("State ({x}, {y})"));
                let row = vec![
                    self.agent.get_q_value(&state, &Move::Up),
                    self.agent.get_q_value(&state, &Move::Down),
                    self.agent.get_q_value(&state, &Move::Left),
                    self.agent.get_q_value(&state, &Move::Right),
                ];
                matrix.push(row);
            }
        }

        Some(crate::framework::ExportableData::grid(
            "Q-Table Matrix",
            row_headers,
            col_headers,
            matrix,
        ))
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        let input_mode = ctx.data(|d| {
            d.get_temp(egui::Id::new("INPUT_MODE"))
                .unwrap_or(InputMode::Mouse)
        });
        let touch_min = 44.0;
        let legend_size = if input_mode == InputMode::Touch { 15.0_f32.max(touch_min) } else { 15.0 };
        let cell_size = if input_mode == InputMode::Touch { 60.0_f32.max(touch_min) } else { 60.0 };

        egui::SidePanel::left("q_table_controls").show(ctx, |ui| {
            ui.heading("Training Controls");
            ui.separator();

            ui.label(format!("Episodes Trained: {}", self.episodes_trained));

            if ui
                .button("▶ Train 100 Episodes")
                .accessible_hover_text("Train the agent for 100 episodes instantly")
                .clicked()
            {
                self.train(100);
            }
            if ui
                .button("▶ Train 1000 Episodes")
                .accessible_hover_text("Train the agent for 1000 episodes instantly")
                .clicked()
            {
                self.train(1000);
            }
            if ui
                .button("↻ Reset Agent")
                .accessible_hover_text("Clear the Q-table and reset the agent's knowledge")
                .clicked()
            {
                self.agent = TabularQAgent::new(math_commons::primitives::UnitInterval::new(0.1).unwrap(), math_commons::primitives::UnitInterval::new(0.9).unwrap(), math_commons::primitives::UnitInterval::new(0.1).unwrap(), None);
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
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
