#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Side-panel controls for [`LossLandscapeTool`] (arena transport, empirical dataset/loss, physics, sculpting).

use super::arena::{RacerKind, LR_MAX, LR_MIN};
use super::data::{LossMetric, TargetDataMode, MAX_TABLE_ROWS};
use super::surfaces::{
    LandscapeSurface, MAX_SCULPT_BUMPS, SCULPT_AMP_MAX, SCULPT_AMP_MIN, SCULPT_SIGMA_MAX,
    SCULPT_SIGMA_MIN,
};
use super::{
    CanvasInteractionMode, LandscapeViewMode, LossLandscapeTool, PredictorMode, LOSS_MIN_SPAN,
    LOSS_PARAM_MAX_BOUND, LOSS_PARAM_MIN_BOUND,
};
use crate::framework::{
    bounded_drag_value, enforce_ordered_drag_range, InteractiveTool, PlaybackAction,
    PlaybackTransport,
};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;
use math_explorer::pure_math::analysis::momentum_dynamics::{
    MAX_DAMPING, MAX_DT, MAX_MASS, MAX_TEMPERATURE, MIN_DAMPING, MIN_DT, MIN_MASS, MIN_TEMPERATURE,
};

impl LossLandscapeTool {
    pub(super) fn draw_ordered_bound_pair(
        ui: &mut egui::Ui,
        label: &str,
        min_v: &mut f64,
        max_v: &mut f64,
    ) -> bool {
        ui.label(label);
        let (mut min_changed, mut max_changed) = (false, false);
        ui.horizontal(|ui| {
            min_changed = ui
                .add(
                    bounded_drag_value(min_v, LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND, 0.1)
                        .prefix("Min: "),
                )
                .changed();
            max_changed = ui
                .add(
                    bounded_drag_value(max_v, LOSS_PARAM_MIN_BOUND..=LOSS_PARAM_MAX_BOUND, 0.1)
                        .prefix("Max: "),
                )
                .changed();
        });
        enforce_ordered_drag_range(
            min_v,
            max_v,
            LOSS_PARAM_MIN_BOUND,
            LOSS_PARAM_MAX_BOUND,
            LOSS_MIN_SPAN,
            min_changed,
            max_changed,
        );
        min_changed || max_changed
    }

    pub(super) fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        self.poll_obj_export(parent_ui.ctx());
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "loss_landscape_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Loss Landscape Arena");
            Self::draw_ordered_bound_pair(ui, "Weight (w) Range", &mut self.w_min, &mut self.w_max);
            Self::draw_ordered_bound_pair(ui, "Bias (b) Range", &mut self.b_min, &mut self.b_max);
            ui.separator();
            egui::ScrollArea::vertical()
                .id_salt("loss_arena_scroll")
                .show(ui, |ui| {
                    self.draw_transport_and_mode_ui(ui);
                    if self.surface == LandscapeSurface::LinearMse {
                        ui.separator();
                        self.draw_empirical_regression_ui(ui);
                    }
                    ui.separator();
                    self.draw_physics_and_sculpt_ui(ui);
                    ui.separator();
                    self.draw_racers_and_camera_ui(ui);
                });
        });
    }

    fn draw_transport_and_mode_ui(&mut self, ui: &mut egui::Ui) {
        if let Some(act) =
            PlaybackTransport::in_thread("loss_arena_transport", self.name(), self.is_playing)
                .show(ui)
        {
            match act {
                PlaybackAction::Play => self.is_playing = true,
                PlaybackAction::Pause => self.is_playing = false,
                PlaybackAction::Step => self.step_simulation(),
                PlaybackAction::Reset => {
                    self.is_playing = false;
                    self.reset_arena();
                }
            }
        }
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.view_mode, LandscapeViewMode::View3D, "3D Surface");
            ui.selectable_value(
                &mut self.view_mode,
                LandscapeViewMode::Contour2D,
                "2D Contours",
            );
        });
        let prev_surface = self.surface;
        egui::ComboBox::from_label("Surface")
            .selected_text(self.surface.label())
            .show_ui(ui, |ui| {
                for s in LandscapeSurface::ALL {
                    ui.selectable_value(&mut self.surface, s, s.label());
                }
            });
        if self.surface != prev_surface {
            self.apply_surface_preset(self.surface);
        }
        ui.horizontal_wrapped(|ui| {
            ui.selectable_value(
                &mut self.interaction_mode,
                CanvasInteractionMode::Orbit,
                "Orbit",
            );
            ui.selectable_value(
                &mut self.interaction_mode,
                CanvasInteractionMode::Slingshot,
                "Slingshot",
            );
            ui.selectable_value(
                &mut self.interaction_mode,
                CanvasInteractionMode::SculptBrush,
                "Sculpt",
            );
        });
    }

    fn draw_empirical_regression_ui(&mut self, ui: &mut egui::Ui) {
        let prev_mode = self.predictor_mode;
        ui.label("Predictor Formula:");
        ui.horizontal(|ui| {
            ui.radio_value(
                &mut self.predictor_mode,
                PredictorMode::Linear,
                "Linear (w*x+b)",
            );
            ui.radio_value(
                &mut self.predictor_mode,
                PredictorMode::Custom,
                "Custom f(x,w,b)",
            );
        });
        let mut reset_needed = self.predictor_mode != prev_mode;
        if self.predictor_mode == PredictorMode::Custom {
            ui.add_space(4.0);
            reset_needed |= self.math_input.ui(ui);
        }
        reset_needed |= self.draw_target_data_section(ui);
        reset_needed |= self.draw_loss_metric_section(ui);
        if reset_needed {
            self.reset_arena();
        }
    }

    fn draw_target_data_section(&mut self, ui: &mut egui::Ui) -> bool {
        ui.label("Target Dataset:");
        let mut target_changed = false;
        egui::ComboBox::from_id_salt("loss_target_mode")
            .selected_text(self.target_mode.label())
            .show_ui(ui, |ui| {
                for mode in [
                    TargetDataMode::PresetLinear,
                    TargetDataMode::Formula,
                    TargetDataMode::Table,
                ] {
                    if ui
                        .selectable_value(&mut self.target_mode, mode, mode.label())
                        .changed()
                    {
                        target_changed = true;
                    }
                }
            });
        target_changed |= self.draw_target_mode_inputs(ui);
        if target_changed {
            self.refresh_target_samples();
        }
        target_changed
    }

    fn draw_target_mode_inputs(&mut self, ui: &mut egui::Ui) -> bool {
        let mut changed = false;
        match self.target_mode {
            TargetDataMode::PresetLinear => {
                ui.small("5 points sampled along y = 2x + 1");
            }
            TargetDataMode::Formula => {
                ui.add_space(4.0);
                changed |= self.target_formula_bar.ui(ui);
                let [mut x0, mut x1] = self.target_x_bounds;
                if Self::draw_ordered_bound_pair(ui, "Sample x Range", &mut x0, &mut x1) {
                    self.target_x_bounds = [x0, x1];
                    changed = true;
                }
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.target_sample_count, 2..=MAX_TABLE_ROWS)
                            .text("Sample Count"),
                    )
                    .changed();
            }
            TargetDataMode::Table => {
                ui.small("Enter (x, y) pairs (one per line):");
                changed |= ui
                    .add(
                        egui::TextEdit::multiline(&mut self.target_table_text)
                            .desired_rows(4)
                            .code_editor(),
                    )
                    .changed();
                if let Some(ref err) = self.table_parse_error {
                    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
                    ui.colored_label(palette.error, format!("⚠️ {err}"));
                }
            }
        }
        changed
    }

    fn draw_loss_metric_section(&mut self, ui: &mut egui::Ui) -> bool {
        ui.label("Loss Objective:");
        let prev_metric = self.loss_metric;
        egui::ComboBox::from_id_salt("loss_metric_mode")
            .selected_text(self.loss_metric.label())
            .show_ui(ui, |ui| {
                for metric in [LossMetric::Mse, LossMetric::Mae, LossMetric::Huber] {
                    ui.selectable_value(&mut self.loss_metric, metric, metric.label());
                }
            });
        let mut changed = self.loss_metric != prev_metric;
        if self.loss_metric == LossMetric::Huber {
            let prev_delta = self.huber_delta;
            ui.add(
                AnimatedSlider::new(&mut self.huber_delta, 0.05..=20.0)
                    .text("Huber delta")
                    .id_salt("loss_huber_delta"),
            );
            changed |= (self.huber_delta - prev_delta).abs() > 1e-12;
        }
        changed
    }

    fn draw_physics_and_sculpt_ui(&mut self, ui: &mut egui::Ui) {
        let max_res = crate::presets::MAX_INTERACTIVE_GRID_SIZE;
        self.resolution = self.resolution.clamp(10, max_res);
        ui.add(egui::Slider::new(&mut self.resolution, 10..=max_res).text("Grid"));
        ui.add(egui::Slider::new(&mut self.arena_config.mass, MIN_MASS..=MAX_MASS).text("Mass m"));
        ui.add(
            egui::Slider::new(&mut self.arena_config.damping, MIN_DAMPING..=MAX_DAMPING)
                .text("Damp γ"),
        );
        ui.add(
            egui::Slider::new(
                &mut self.arena_config.temperature,
                MIN_TEMPERATURE..=MAX_TEMPERATURE,
            )
            .text("Temp kBT"),
        );
        ui.add(
            egui::Slider::new(&mut self.arena_config.dt, MIN_DT..=MAX_DT)
                .logarithmic(true)
                .text("Step dt"),
        );
        ui.add(
            egui::Slider::new(&mut self.arena_config.learning_rate, LR_MIN..=LR_MAX)
                .logarithmic(true)
                .text("LR η"),
        );
        ui.label(format!(
            "Sculpt Bumps: {}/{}",
            self.sculpt_bumps.len(),
            MAX_SCULPT_BUMPS
        ));
        ui.add(
            egui::Slider::new(&mut self.sculpt_amplitude, SCULPT_AMP_MIN..=SCULPT_AMP_MAX)
                .text("Amp A"),
        );
        ui.add(
            egui::Slider::new(&mut self.sculpt_sigma, SCULPT_SIGMA_MIN..=SCULPT_SIGMA_MAX)
                .text("Width σ"),
        );
        if !self.sculpt_bumps.is_empty() && ui.button("Clear Sculpted Bumps").clicked() {
            self.sculpt_bumps.clear();
            self.reset_arena();
        }
    }

    fn draw_racers_and_camera_ui(&mut self, ui: &mut egui::Ui) {
        let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
        ui.label(format!("Arena Step: {}", self.arena.step_count));
        for (idx, kind) in RacerKind::ALL.iter().copied().enumerate() {
            if !kind.is_available_in_build() {
                continue;
            }
            let tr = &self.arena.racers[idx];
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.arena_config.enabled_racers[idx], "");
                ui.colored_label(
                    kind.color(&palette),
                    format!(
                        "{}: L={:.3} H={:.3}",
                        kind.label(),
                        tr.potential_energy,
                        tr.potential_energy + tr.kinetic_energy
                    ),
                );
            });
        }
        egui::CollapsingHeader::new("Camera 3D")
            .default_open(false)
            .show(ui, |ui| self.camera.ui(ui));
        let is_saving = self.cache.obj_export_rx.is_some();
        if ui
            .add_enabled(!is_saving, egui::Button::new("Export to OBJ"))
            .clicked()
        {
            self.trigger_obj_export();
        }
        if let Some((ref message, is_error)) = self.cache.obj_export_status {
            let color = if is_error {
                palette.error
            } else {
                palette.success
            };
            ui.colored_label(color, message);
        }
    }
}
