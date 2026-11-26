//! Side-panel transport, solver/overlay selectors, RL maze sculpting, and stencil controls
//! for [`super::GridWorldTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{GridBrushMode, GridWorldTool};
use crate::accessibility::AccessibleHoverText;
use crate::audio::render_sonification_panel;
use crate::framework::{draw_brush_palette_controls, BrushMode};
use crate::tabs::ai::rl_sculpt::{render_rl_sculpt_toolbar, RlOverlayMode, RlSolverMode};
use eframe::egui;
use math_explorer::ai::reinforcement_learning::GridCellRole;

impl GridWorldTool {
    pub(super) fn draw_side_panel(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "grid_world_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                self.draw_transport_and_solver_controls(ui);
                ui.separator();
                self.draw_sculpting_controls(ui);
                ui.separator();
                self.draw_hyperparameter_controls(ui);
                ui.separator();
                render_sonification_panel(ui, "grid_world", &self.latest_audio, !self.is_playing);
            });
        });
    }

    fn draw_sculpting_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Maze Sculpting");
        let prev_kit_mode = self.brush_kit.mode;
        let prev_radius = self.brush_kit.radius;
        if render_rl_sculpt_toolbar(
            ui,
            "grid_world",
            &mut self.mdp,
            &mut self.brush_kit,
            &mut self.maze_preset,
        ) {
            if self.mdp.cell_role(self.current_state) == GridCellRole::SolidWall {
                self.current_state = self.mdp.start;
            }
            self.bellman_dirty = true;
            self.ensure_bellman_up_to_date();
            self.revision = self.revision.wrapping_add(1);
        }
        if self.brush_kit.mode != prev_kit_mode {
            self.brush_mode = match self.brush_kit.mode {
                BrushMode::Solid | BrushMode::ZeroFluxWall => GridBrushMode::Wall,
                BrushMode::RlTrap => GridBrushMode::Trap,
                BrushMode::RlGoal => GridBrushMode::Goal,
                BrushMode::Erase => GridBrushMode::Empty,
                _ => self.brush_mode,
            };
        }
        if (self.brush_kit.radius - prev_radius).abs() > f64::EPSILON {
            self.brush_state.radius = (self.brush_kit.radius - 0.5).round().max(0.0) as usize;
        }

        let prev_brush_mode = self.brush_mode;
        egui::ComboBox::from_id_salt("grid_world_brush_mode")
            .selected_text(self.brush_mode.label())
            .show_ui(ui, |ui| {
                for mode in GridBrushMode::ALL {
                    ui.selectable_value(&mut self.brush_mode, mode, mode.label());
                }
            });
        if self.brush_mode != prev_brush_mode {
            self.brush_kit.mode = match self.brush_mode {
                GridBrushMode::Wall => BrushMode::Solid,
                GridBrushMode::Trap => BrushMode::RlTrap,
                GridBrushMode::Goal => BrushMode::RlGoal,
                GridBrushMode::Empty | GridBrushMode::Start => BrushMode::Erase,
            };
        }
        draw_brush_palette_controls(ui, "grid_world", &mut self.brush_state, 3);
        if ui
            .button("Clear Obstacles & Traps")
            .accessible_hover_text("Remove all walls and traps while keeping Start and Goal")
            .clicked()
        {
            self.clear_sculpted_obstacles();
        }
    }

    fn draw_transport_and_solver_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("Controls");
        ui.horizontal_wrapped(|ui| {
            if ui
                .button("▶ Step")
                .accessible_hover_text("Advance the agent by one step")
                .clicked()
            {
                self.step_agent();
            }
            let (label, hover) = if self.is_playing {
                ("⏸ Pause", "Pause continuous RL playback")
            } else {
                ("▶ Play", "Start continuous RL playback")
            };
            if ui.button(label).accessible_hover_text(hover).clicked() {
                self.is_playing = !self.is_playing;
                if self.is_playing {
                    self.last_tick = crate::web_io::PlatformInstant::now();
                }
                self.revision = self.revision.wrapping_add(1);
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
            .accessible_hover_text("Clear learned values and reset agent state")
            .clicked()
        {
            self.reset_agent();
        }
        self.draw_solver_and_overlay_selectors(ui);
    }

    fn draw_solver_and_overlay_selectors(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Solver:");
            egui::ComboBox::from_id_salt("grid_world_solver")
                .selected_text(self.solver_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [RlSolverMode::TabularQ, RlSolverMode::PolicyGradient] {
                        if ui
                            .selectable_value(&mut self.solver_mode, mode, mode.label())
                            .changed()
                        {
                            self.bellman_dirty = true;
                            self.ensure_bellman_up_to_date();
                            self.revision = self.revision.wrapping_add(1);
                        }
                    }
                });
        });
        ui.horizontal(|ui| {
            ui.label("Overlay:");
            egui::ComboBox::from_id_salt("grid_world_overlay")
                .selected_text(self.overlay_mode.label())
                .show_ui(ui, |ui| {
                    for mode in [
                        RlOverlayMode::EmpiricalPolicy,
                        RlOverlayMode::BellmanVStar,
                        RlOverlayMode::PolicyValueVPi,
                        RlOverlayMode::BellmanResidual,
                    ] {
                        if ui
                            .selectable_value(&mut self.overlay_mode, mode, mode.label())
                            .changed()
                        {
                            self.bellman_dirty = true;
                            self.ensure_bellman_up_to_date();
                            self.revision = self.revision.wrapping_add(1);
                        }
                    }
                });
        });
    }

    fn draw_hyperparameter_controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("RL Hyperparameters");
        let mut hp_changed = ui
            .add(egui::Slider::new(&mut self.learning_rate, 0.01..=1.0).text("Learning Rate (α)"))
            .changed();
        hp_changed |= ui
            .add(egui::Slider::new(&mut self.discount_factor, 0.0..=1.0).text("Discount (γ)"))
            .changed();
        hp_changed |= ui
            .add(egui::Slider::new(&mut self.exploration_rate, 0.0..=1.0).text("Exploration (ε)"))
            .changed();
        if hp_changed {
            self.sync_agent_hyperparameters();
        }
        if ui
            .add(egui::Slider::new(&mut self.playback_speed, 1.0..=20.0).text("Speed (Hz)"))
            .changed()
        {
            self.revision = self.revision.wrapping_add(1);
        }
    }
}
