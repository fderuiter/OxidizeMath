//! Side-panel controls and comparative scoring table for `CalibrationWorkbenchTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::app::SemanticPalette;
use eframe::egui;
use math_explorer::pure_math::statistics::calibration::BinningStrategy;

use super::state::{
    CalibrationWorkbenchConfig, ForecastScenario, RecalibratorKind, WorkbenchEvaluation,
    WorkbenchView,
};

/// Renders the left control panel and returns `true` when any parameter changed.
pub fn draw_controls_panel(
    parent_ui: &mut egui::Ui,
    cfg: &mut CalibrationWorkbenchConfig,
    eval: &WorkbenchEvaluation,
) -> bool {
    let mut changed = false;
    let panel_id = crate::framework::scoped_panel_id(parent_ui, "calibration_workbench_controls");
    crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.heading("Calibration Workbench");
            ui.separator();

            changed |= draw_selectors(ui, cfg);
            ui.separator();
            changed |= draw_parameter_sliders(ui, cfg);
            ui.separator();
            draw_metrics_table(ui, eval);
        });
    });
    changed
}

fn draw_selectors(ui: &mut egui::Ui, cfg: &mut CalibrationWorkbenchConfig) -> bool {
    let mut changed = false;
    ui.label("Diagnostic View");
    egui::ComboBox::from_id_salt("cal_wb_view")
        .selected_text(cfg.view.label())
        .show_ui(ui, |ui| {
            for mode in WorkbenchView::ALL {
                if ui
                    .selectable_value(&mut cfg.view, mode, mode.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });

    ui.add_space(4.0);
    ui.label("Forecast Benchmark");
    egui::ComboBox::from_id_salt("cal_wb_scenario")
        .selected_text(cfg.scenario.label())
        .show_ui(ui, |ui| {
            for sc in ForecastScenario::ALL {
                if ui
                    .selectable_value(&mut cfg.scenario, sc, sc.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });

    ui.add_space(4.0);
    ui.label("Post-Hoc Recalibrator");
    egui::ComboBox::from_id_salt("cal_wb_recal")
        .selected_text(cfg.recalibrator.label())
        .show_ui(ui, |ui| {
            for rec in RecalibratorKind::ALL {
                if ui
                    .selectable_value(&mut cfg.recalibrator, rec, rec.label())
                    .changed()
                {
                    changed = true;
                }
            }
        });

    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Binning:");
        if ui
            .selectable_value(&mut cfg.binning, BinningStrategy::Uniform, "Uniform")
            .changed()
        {
            changed = true;
        }
        if ui
            .selectable_value(&mut cfg.binning, BinningStrategy::Quantile, "Quantile")
            .changed()
        {
            changed = true;
        }
    });
    changed
}

fn draw_parameter_sliders(ui: &mut egui::Ui, cfg: &mut CalibrationWorkbenchConfig) -> bool {
    let mut changed = false;
    changed |= ui
        .add(egui::Slider::new(&mut cfg.logit_scale, 0.2..=5.0).text("Logit Scale"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut cfg.logit_bias, -2.5..=2.5).text("Logit Bias"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut cfg.crps_spread, 0.2..=4.0).text("CRPS Spread σ"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut cfg.num_bins, 2..=30).text("Bins"))
        .changed();
    changed |= ui
        .add(egui::Slider::new(&mut cfg.sample_count, 60..=800).text("Cohort N"))
        .changed();

    ui.add_space(4.0);
    if ui.button("↺ Reset Defaults").clicked() {
        *cfg = CalibrationWorkbenchConfig::default();
        changed = true;
    }
    changed
}

fn draw_metrics_table(ui: &mut egui::Ui, eval: &WorkbenchEvaluation) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    ui.strong("Held-Out Proper Scoring & Calibration");
    ui.colored_label(palette.info, &eval.fit_summary);
    ui.add_space(4.0);

    let r = &eval.raw_metrics;
    let c = &eval.cal_metrics;
    egui::Grid::new("cal_wb_metrics_grid")
        .striped(true)
        .spacing([10.0, 4.0])
        .show(ui, |ui| {
            ui.strong("Metric");
            ui.strong("Raw");
            ui.strong("Calib");
            ui.end_row();

            metric_row(ui, "Brier Score", r.brier, c.brier, true, &palette);
            metric_row(ui, "Log-Loss", r.log_loss, c.log_loss, true, &palette);
            metric_row(ui, "ECE", r.ece, c.ece, true, &palette);
            metric_row(ui, "MCE", r.mce, c.mce, true, &palette);
            metric_row(
                ui,
                "Murphy REL",
                r.murphy.reliability,
                c.murphy.reliability,
                true,
                &palette,
            );
            metric_row(
                ui,
                "Murphy RES",
                r.murphy.resolution,
                c.murphy.resolution,
                false,
                &palette,
            );
            metric_row(ui, "ROC-AUC", r.roc_auc, c.roc_auc, false, &palette);
            metric_row(
                ui,
                "PR-AUC (Step)",
                r.pr_auc_step,
                c.pr_auc_step,
                false,
                &palette,
            );
            metric_row(
                ui,
                "PR-AUC (Trap)",
                r.pr_auc_trap,
                c.pr_auc_trap,
                false,
                &palette,
            );
            metric_row(
                ui,
                "Mean CRPS",
                eval.mean_raw_crps,
                eval.mean_cal_crps,
                true,
                &palette,
            );
        });
}

fn metric_row(
    ui: &mut egui::Ui,
    name: &str,
    raw_val: f64,
    cal_val: f64,
    lower_is_better: bool,
    palette: &SemanticPalette,
) {
    ui.label(name);
    ui.monospace(format!("{raw_val:.4}"));
    let improved = if lower_is_better {
        cal_val + 1e-6 < raw_val
    } else {
        cal_val > raw_val + 1e-6
    };
    let color = if improved {
        palette.success
    } else {
        ui.visuals().text_color()
    };
    ui.colored_label(color, format!("{cal_val:.4}"));
    ui.end_row();
}

// [cite:clinical_trials_statistics]
