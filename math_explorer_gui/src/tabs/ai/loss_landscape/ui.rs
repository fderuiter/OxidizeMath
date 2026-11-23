#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::data::{LossMetric, TargetDataMode, MAX_TABLE_ROWS};
use super::{
    LossLandscapeTool, PredictorMode, LOSS_MIN_SPAN, LOSS_PARAM_MAX_BOUND, LOSS_PARAM_MIN_BOUND,
};
use crate::framework::{bounded_drag_value, enforce_ordered_drag_range};
use crate::reflective_ui::AnimatedSlider;
use eframe::egui;

impl LossLandscapeTool {
    fn draw_ordered_bound_pair(
        ui: &mut egui::Ui,
        label: &str,
        min_v: &mut f64,
        max_v: &mut f64,
    ) -> bool {
        ui.label(label);
        let mut min_changed = false;
        let mut max_changed = false;
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
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "loss_landscape_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Parameters");
            ui.separator();
            self.draw_predictor_section(ui);
            ui.separator();
            self.draw_target_data_section(ui);
            ui.separator();
            self.draw_loss_metric_section(ui);
            ui.separator();
            Self::draw_ordered_bound_pair(
                ui,
                "Weight (Slope) Range",
                &mut self.w_min,
                &mut self.w_max,
            );
            Self::draw_ordered_bound_pair(
                ui,
                "Bias (Intercept) Range",
                &mut self.b_min,
                &mut self.b_max,
            );
            ui.separator();
            ui.add(
                egui::Slider::new(&mut self.resolution, 10..=50)
                    .text("Resolution (Grid Size) - points"),
            );
        });
    }

    fn draw_predictor_section(&mut self, ui: &mut egui::Ui) {
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
        if self.predictor_mode == PredictorMode::Custom {
            ui.add_space(4.0);
            self.math_input.ui(ui);
        }
    }

    fn draw_target_data_section(&mut self, ui: &mut egui::Ui) {
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

    fn draw_loss_metric_section(&mut self, ui: &mut egui::Ui) {
        ui.label("Loss Objective:");
        egui::ComboBox::from_id_salt("loss_metric_mode")
            .selected_text(self.loss_metric.label())
            .show_ui(ui, |ui| {
                for metric in [LossMetric::Mse, LossMetric::Mae, LossMetric::Huber] {
                    ui.selectable_value(&mut self.loss_metric, metric, metric.label());
                }
            });
        if self.loss_metric == LossMetric::Huber {
            ui.add(
                AnimatedSlider::new(&mut self.huber_delta, 0.05..=20.0)
                    .text("Huber delta")
                    .id_salt("loss_huber_delta"),
            );
        }
    }
}
