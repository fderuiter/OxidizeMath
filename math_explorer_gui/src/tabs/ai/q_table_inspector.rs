use crate::accessibility::AccessibleHoverText;
use crate::async_sim::ui::{
    render_empty_plot_guidance, render_worker_fault_banner, WorkerFaultAction,
};
use crate::async_sim::{BackgroundWorker, SimCommand, SimWorkerState};
use crate::framework::{InputMode, InteractiveTool, PlaybackAction, PlaybackTransport};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::grid_world::{GridState, GridWorldEnv, Move};
use std::collections::HashMap;

#[path = "q_table_inspector/runner.rs"]
mod runner;
pub use runner::*;

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub(super) enum HeatmapView {
    MaxQ,
    Up,
    Down,
    Left,
    Right,
}

pub struct QTableInspectorTool {
    env: GridWorldEnv,
    cached_q_values: HashMap<(GridState, Move), f64>,
    episodes_trained: u32,
    view: HeatmapView,
    pub(crate) worker: BackgroundWorker<QTableRunner>,
}

impl Default for QTableInspectorTool {
    fn default() -> Self {
        let env = default_grid_env();
        let worker = BackgroundWorker::new(QTableRunner::new());
        let mut tool = Self {
            env,
            cached_q_values: HashMap::new(),
            episodes_trained: 0,
            view: HeatmapView::MaxQ,
            worker,
        };
        tool.sync_from_latest();
        tool
    }
}

impl QTableInspectorTool {
    pub(super) fn sync_from_latest(&mut self) {
        if let Some(snapshot) = self.worker.latest_snapshot() {
            if let Some(data) = snapshot
                .structured_data
                .as_ref()
                .and_then(|d| d.downcast_ref::<QTableSnapshot>())
            {
                self.cached_q_values = data.q_values.clone();
                self.episodes_trained = data.episodes_trained;
            }
        }
    }

    /// Resets the background Q-learning agent in place or respawns a disconnected worker.
    pub fn reset(&mut self) {
        self.worker.reset_or_respawn(QTableRunner::new);
        self.cached_q_values.clear();
        self.episodes_trained = 0;
        self.sync_from_latest();
    }

    #[cfg(test)]
    pub fn train(&mut self, episodes: u32) {
        let target = self.episodes_trained + episodes;
        let _ = self
            .worker
            .send_command(SimCommand::TrainEpisodes(episodes as usize));
        for _ in 0..200 {
            if let Some(snap) = self.worker.update() {
                if let Some(data) = snap
                    .structured_data
                    .as_ref()
                    .and_then(|d| d.downcast_ref::<QTableSnapshot>())
                {
                    self.cached_q_values = data.q_values.clone();
                    self.episodes_trained = data.episodes_trained;
                    if self.episodes_trained >= target {
                        break;
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    fn get_q(&self, state: &GridState, action: &Move) -> f64 {
        self.cached_q_values
            .get(&(*state, *action))
            .copied()
            .unwrap_or(0.0)
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

    fn handle_transport_action(&mut self, action: PlaybackAction, ctx: &egui::Context) {
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
                self.reset();
                ctx.request_repaint();
            }
        }
    }

    fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui, ctx: &egui::Context, legend_size: f32) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "q_table_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Training Controls");
            ui.separator();

            if let Some(WorkerFaultAction::Reset | WorkerFaultAction::ResetParameters) =
                render_worker_fault_banner(
                    ui,
                    self.worker.state(),
                    self.worker.diagnostic_report(),
                    false,
                )
            {
                self.reset();
                ctx.request_repaint();
                ui.separator();
            }

            let state = self.worker.state();
            ui.label(format!("Episodes Trained: {}", self.episodes_trained));
            if state.is_running() {
                let info = crate::app::SemanticPalette::from_visuals(ui.visuals()).info;
                ui.colored_label(info, "⏳ Training in background...");
            }

            if let Some(action) = PlaybackTransport::from_worker_state(
                "q_table_transport",
                "Q-Table Inspector",
                state,
            )
            .play_label("▶ Train 100")
            .step_label("⏭ Step Episode")
            .can_reset(true)
            .show(ui)
            {
                self.handle_transport_action(action, ctx);
            }

            if ui
                .add_enabled(state.is_ready(), egui::Button::new("▶ Train 1000 Episodes"))
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

        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.separator();
        ui.label("Legend:");
        for (label, color) in [("Goal:", palette.success), ("Trap:", palette.error)] {
            ui.horizontal(|ui| {
                ui.label(label);
                let (resp, painter) =
                    ui.allocate_painter(egui::vec2(legend_size, legend_size), egui::Sense::hover());
                painter.rect_filled(resp.rect, 0.0, color);
            });
        }
    }

    fn compute_q_bounds(&self) -> (f64, f64) {
        let mut min_q = 0.0_f64;
        let mut max_q = 0.0_f64;
        for x in 0..self.env.width {
            for y in 0..self.env.height {
                let q = self.get_q_value_for_view(&GridState { x, y });
                if q.is_finite() {
                    min_q = min_q.min(q);
                    max_q = max_q.max(q);
                }
            }
        }
        if (max_q - min_q).abs() < f64::EPSILON {
            max_q = min_q + 1.0;
        }
        (min_q, max_q)
    }

    fn paint_heatmap_cell(
        &self,
        painter: &egui::Painter,
        cell_rect: egui::Rect,
        state: GridState,
        bounds: (f64, f64),
        palette: &crate::app::SemanticPalette,
    ) {
        let (min_q, max_q) = bounds;
        let q_value = self.get_q_value_for_view(&state);
        let fill_color = if state == self.env.goal {
            palette.success
        } else if self.env.traps.contains(&state) {
            palette.error
        } else {
            let norm = ((q_value - min_q) / (max_q - min_q)).clamp(0.0, 1.0);
            q_value_cell_color(norm, palette)
        };

        painter.rect_filled(cell_rect, 0.0, fill_color);
        painter.rect_stroke(
            cell_rect,
            0.0,
            egui::Stroke::new(1.0_f32, palette.graph_edge),
            egui::StrokeKind::Middle,
        );

        if state != self.env.goal && !self.env.traps.contains(&state) {
            let text_color = crate::app::contrasting_text_color(fill_color);
            painter.text(
                cell_rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("{q_value:.2}"),
                egui::FontId::proportional(14.0),
                text_color,
            );
        }
    }

    fn draw_heatmap(&self, parent_ui: &mut egui::Ui, cell_size: f32) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Q-Value Heatmap");
            if self.episodes_trained == 0 {
                render_empty_plot_guidance(
                    ui,
                    "Untrained baseline Q-table (0.00) — click '▶ Train 100' or '⏭ Step Episode' to update state-action values.",
                );
            }

            let grid_size = egui::vec2(
                self.env.width as f32 * cell_size,
                self.env.height as f32 * cell_size,
            );
            let (response, painter) =
                ui.allocate_painter(grid_size, egui::Sense::focusable_noninteractive());
            let _ = response
                .clone()
                .accessible_hover_text("Q-Table Heatmap Visualization");
            let view_name = match self.view {
                HeatmapView::MaxQ => "Max Q",
                HeatmapView::Up => "Up",
                HeatmapView::Down => "Down",
                HeatmapView::Left => "Left",
                HeatmapView::Right => "Right",
            };
            let summary = format!(
                "Episodes trained: {}, View: {}",
                self.episodes_trained, view_name
            );
            crate::accessibility::register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                "Q-Table Heatmap Visualization",
                Some(&summary),
            );
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let rect = response.rect;
            let bounds = self.compute_q_bounds();

            for x in 0..self.env.width {
                for y in 0..self.env.height {
                    let top_left =
                        rect.min + egui::vec2(x as f32 * cell_size, y as f32 * cell_size);
                    let cell_rect =
                        egui::Rect::from_min_size(top_left, egui::vec2(cell_size, cell_size));
                    self.paint_heatmap_cell(
                        &painter,
                        cell_rect,
                        GridState { x, y },
                        bounds,
                        &palette,
                    );
                }
            }
            crate::accessibility::paint_canvas_focus_ring(ui, &response);
        });
    }
}

fn lerp_q_channel(a: u8, b: u8, t: f64) -> u8 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * t.clamp(0.0, 1.0)).round() as u8
}

pub(crate) fn q_value_cell_color(
    normalized: f64,
    palette: &crate::app::SemanticPalette,
) -> egui::Color32 {
    let low = palette.axis_z;
    let high = palette.accent;
    egui::Color32::from_rgb(
        lerp_q_channel(low.r(), high.r(), normalized),
        lerp_q_channel(low.g(), high.g(), normalized),
        lerp_q_channel(low.b(), high.b(), normalized),
    )
}

impl InteractiveTool for QTableInspectorTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Q-Table Inspector"
    }
    fn export_capabilities(&self) -> crate::framework::ExportCapabilities {
        crate::framework::ExportCapabilities::none()
    }
    fn domain(&self) -> &'static str {
        "ai"
    }
    fn save_state(&self) -> Option<String> {
        self.save_preset_state()
    }
    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.try_load_preset_state(state)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
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

        self.draw_side_panel(ui, &ctx, legend_size);
        self.draw_heatmap(ui, cell_size);
    }
}

// [cite:graph_parameters_rust]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "QTableInspectorTool",
        display_name: "Q-Table Inspector",
        domain: "ai",
        tags: &["q-learning", "temporal difference", "bellman", "value table", "rl"],
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

#[path = "q_table_inspector/preset.rs"]
mod preset;
#[cfg(test)]
#[path = "q_table_inspector/tests.rs"]
mod tests;
