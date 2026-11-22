#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{GridBrushMode, GridWorldTool};
use crate::accessibility::{AccessibleHoverText, AccessibleTheoryHover};
use crate::framework::{
    cell_screen_rect, draw_brush_palette_controls, InputMode, RasterBrushCanvas,
};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::grid_world::GridState;

impl GridWorldTool {
    pub(super) fn cell_fill(
        &self,
        state: GridState,
        palette: &crate::app::SemanticPalette,
    ) -> egui::Color32 {
        if state == self.env.goal {
            palette.success
        } else if self.env.traps.contains(&state) {
            palette.error
        } else if self.env.walls.contains(&state) {
            palette.node_stroke
        } else if state == self.current_state {
            palette.accent
        } else if state == self.env.start {
            palette.info
        } else {
            palette.surface_fill
        }
    }

    fn draw_brush_controls(&mut self, ui: &mut egui::Ui) {
        ui.separator();
        ui.heading("Environment Sculpting");
        egui::ComboBox::from_id_salt("grid_world_brush_mode")
            .selected_text(self.brush_mode.label())
            .show_ui(ui, |ui| {
                for mode in GridBrushMode::ALL {
                    ui.selectable_value(&mut self.brush_mode, mode, mode.label());
                }
            });
        draw_brush_palette_controls(ui, "grid_world", &mut self.brush_state, 3);
        if ui
            .button("Clear Obstacles & Traps")
            .accessible_hover_text("Remove all walls and traps while keeping Start and Goal")
            .clicked()
        {
            self.clear_sculpted_obstacles();
        }
    }

    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
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
            ui.heading("RL Hyperparameters");
            let mut hp_changed = ui
                .add(egui::Slider::new(&mut self.learning_rate, 0.01..=1.0).text("Learning Rate (α)"))
                .changed();
            hp_changed |= ui
                .add(egui::Slider::new(&mut self.discount_factor, 0.0..=1.0).text("Discount Factor (γ)"))
                .changed();
            hp_changed |= ui
                .add(egui::Slider::new(&mut self.exploration_rate, 0.0..=1.0).text("Exploration Rate (ε)"))
                .changed();
            if hp_changed {
                self.sync_agent_hyperparameters();
            }
            ui.add(egui::Slider::new(&mut self.playback_speed, 1.0..=20.0).text("Speed (Hz)"));

            self.draw_brush_controls(ui);
        });
    }

    fn paint_grid_cells(
        &self,
        painter: &egui::Painter,
        rect: egui::Rect,
        palette: &crate::app::SemanticPalette,
    ) {
        let gw = self.env.width.max(1) as usize;
        let gh = self.env.height.max(1) as usize;
        for x in 0..gw {
            for y in 0..gh {
                let state = GridState {
                    x: x as i32,
                    y: y as i32,
                };
                let cell_rect = cell_screen_rect(rect, x, y, gw, gh, false);
                painter.rect_filled(cell_rect, 0.0, self.cell_fill(state, palette));
                if self.env.walls.contains(&state) {
                    painter.line_segment(
                        [cell_rect.left_top(), cell_rect.right_bottom()],
                        egui::Stroke::new(1.5_f32, palette.surface_fill),
                    );
                    painter.line_segment(
                        [cell_rect.right_top(), cell_rect.left_bottom()],
                        egui::Stroke::new(1.5_f32, palette.surface_fill),
                    );
                }
                painter.rect_stroke(
                    cell_rect,
                    0.0,
                    egui::Stroke::new(1.0_f32, palette.graph_edge),
                    egui::StrokeKind::Middle,
                );
            }
        }
    }

    pub(super) fn draw_central_canvas(&mut self, parent_ui: &mut egui::Ui, input_mode: InputMode) {
        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            ui.heading("Grid World Navigation (Q-Learning)");
            ui.separator();
            ui.horizontal(|ui| {
                ui.label(format!("Episode: {}", self.episodes));
                ui.label(format!("Steps: {}", self.steps));
                ui.label(format!("Reward: {:.2}", self.total_reward));
                ui.label(format!(
                    "Walls: {}, Traps: {}",
                    self.env.walls.len(),
                    self.env.traps.len()
                ));
            });
            let cell_size = if input_mode == InputMode::Touch {
                48.0
            } else {
                44.0
            };
            let gw = self.env.width.max(1) as usize;
            let gh = self.env.height.max(1) as usize;
            let max_size = egui::vec2(gw as f32 * cell_size, gh as f32 * cell_size);
            let summary = format!(
                "Agent at ({}, {}), Episode {}, Reward {:.2}, {} walls, {} traps",
                self.current_state.x,
                self.current_state.y,
                self.episodes,
                self.total_reward,
                self.env.walls.len(),
                self.env.traps.len()
            );
            let mut brush_state = std::mem::take(&mut self.brush_state);
            let (response, stroke) =
                RasterBrushCanvas::new("grid_world_canvas", gw, gh, "Grid World Visualization")
                    .role(egui::accesskit::Role::GraphicsDocument)
                    .value_summary(&summary)
                    .max_canvas_size(max_size)
                    .show_custom(ui, &mut brush_state, |painter, rect, palette| {
                        self.paint_grid_cells(painter, rect, palette);
                    });
            self.brush_state = brush_state;
            let _ = response.accessible_theory_hover(self);
            if let Some(stroke) = stroke {
                self.apply_brush_stroke(stroke);
            }
        });
    }
}
