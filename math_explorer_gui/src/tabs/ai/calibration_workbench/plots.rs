//! `egui_plot` renderers for `CalibrationWorkbenchTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::app::SemanticPalette;
use eframe::egui;
use eframe::egui::accesskit::Role;
use egui_plot::{Bar, BarChart, Legend, Line, LineStyle, MarkerShape, Plot, Points};

use super::state::{CalibrationWorkbenchConfig, WorkbenchEvaluation, WorkbenchView};

/// Renders the active diagnostic plot view inside the central panel.
pub fn draw_workbench_plot(
    ui: &mut egui::Ui,
    cfg: &CalibrationWorkbenchConfig,
    eval: &WorkbenchEvaluation,
) {
    let palette = SemanticPalette::from_visuals(ui.visuals());
    egui::CentralPanel::default().show_inside(ui, |ui| {
        ui.horizontal_wrapped(|ui| {
            ui.strong(cfg.view.label());
            ui.separator();
            ui.label(format!(
                "{} | {} ({})",
                cfg.scenario.label(),
                cfg.recalibrator.label(),
                eval.fit_summary
            ));
        });
        ui.add_space(4.0);

        match cfg.view {
            WorkbenchView::Reliability => render_reliability_plot(ui, eval, &palette),
            WorkbenchView::MurphyBars => render_murphy_bars_plot(ui, eval, &palette),
            WorkbenchView::RocCurve => render_roc_plot(ui, eval, &palette),
            WorkbenchView::PrCurve => render_pr_plot(ui, eval, &palette),
            WorkbenchView::CrpsForecast => render_crps_plot(ui, eval, &palette),
        }
    });
}

fn non_empty_bin_points(
    bins: &[math_explorer::pure_math::statistics::calibration::ReliabilityBin],
) -> Vec<[f64; 2]> {
    bins.iter()
        .filter(|b| b.count > 0)
        .map(|b| [b.mean_confidence, b.empirical_accuracy])
        .collect()
}

fn render_reliability_plot(
    ui: &mut egui::Ui,
    eval: &WorkbenchEvaluation,
    palette: &SemanticPalette,
) {
    let raw_pts = non_empty_bin_points(&eval.raw_bins);
    let cal_pts = non_empty_bin_points(&eval.cal_bins);

    let resp = Plot::new("calibration_workbench_reliability")
        .legend(Legend::default())
        .x_axis_label("Mean Predicted Confidence")
        .y_axis_label("Empirical Event Frequency")
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Perfect Calibration (y = x)", vec![[0.0, 0.0], [1.0, 1.0]])
                    .color(palette.graph_edge)
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(
                    format!("Raw (ECE {:.3})", eval.raw_metrics.ece),
                    raw_pts.clone(),
                )
                .color(palette.series_color(1))
                .width(2.0_f32),
            );
            plot_ui.points(
                Points::new("Raw Bins", raw_pts)
                    .color(palette.series_color(1))
                    .radius(4.5_f32)
                    .shape(MarkerShape::Circle),
            );
            plot_ui.line(
                Line::new(
                    format!("Recalibrated (ECE {:.3})", eval.cal_metrics.ece),
                    cal_pts.clone(),
                )
                .color(palette.series_color(0))
                .width(2.5_f32),
            );
            plot_ui.points(
                Points::new("Recalibrated Bins", cal_pts)
                    .color(palette.series_color(0))
                    .radius(5.0_f32)
                    .shape(MarkerShape::Diamond),
            );
        });

    let summary = format!(
        "Raw ECE {:.4}, Recalibrated ECE {:.4}",
        eval.raw_metrics.ece, eval.cal_metrics.ece
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "Calibration Reliability Diagram",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
}

fn render_murphy_bars_plot(
    ui: &mut egui::Ui,
    eval: &WorkbenchEvaluation,
    palette: &SemanticPalette,
) {
    let raw_m = eval.raw_metrics.murphy;
    let cal_m = eval.cal_metrics.murphy;
    let raw_vals = [
        raw_m.reliability,
        raw_m.resolution,
        raw_m.uncertainty,
        raw_m.brier_score,
    ];
    let cal_vals = [
        cal_m.reliability,
        cal_m.resolution,
        cal_m.uncertainty,
        cal_m.brier_score,
    ];
    let labels = [
        "0: Reliability",
        "1: Resolution",
        "2: Uncertainty",
        "3: Brier (REL-RES+UNC)",
    ];

    let raw_bars: Vec<Bar> = raw_vals
        .iter()
        .enumerate()
        .map(|(i, &v)| Bar::new(i as f64 - 0.16, v).width(0.28).name(labels[i]))
        .collect();
    let cal_bars: Vec<Bar> = cal_vals
        .iter()
        .enumerate()
        .map(|(i, &v)| Bar::new(i as f64 + 0.16, v).width(0.28).name(labels[i]))
        .collect();

    let resp = Plot::new("calibration_workbench_murphy")
        .legend(Legend::default())
        .x_axis_label("Component (0=REL, 1=RES, 2=UNC, 3=Brier)")
        .y_axis_label("Quadratic Score Contribution")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui
                .bar_chart(BarChart::new("Raw Forecast", raw_bars).color(palette.series_color(1)));
            plot_ui.bar_chart(
                BarChart::new("Recalibrated Forecast", cal_bars).color(palette.series_color(0)),
            );
        });

    let summary = format!(
        "Murphy partition: Raw REL {:.4} RES {:.4} UNC {:.4}, Recalibrated REL {:.4} RES {:.4}",
        raw_m.reliability, raw_m.resolution, raw_m.uncertainty, cal_m.reliability, cal_m.resolution
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "Murphy Brier Score Decomposition Bar Chart",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
}

fn render_roc_plot(ui: &mut egui::Ui, eval: &WorkbenchEvaluation, palette: &SemanticPalette) {
    let raw_pts: Vec<[f64; 2]> = eval.raw_roc.iter().map(|p| [p.fpr, p.tpr]).collect();
    let cal_pts: Vec<[f64; 2]> = eval.cal_roc.iter().map(|p| [p.fpr, p.tpr]).collect();

    let resp = Plot::new("calibration_workbench_roc")
        .legend(Legend::default())
        .x_axis_label("False Positive Rate (1 - Specificity)")
        .y_axis_label("True Positive Rate (Sensitivity)")
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new("Chance Classifier", vec![[0.0, 0.0], [1.0, 1.0]])
                    .color(palette.graph_edge)
                    .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(
                    format!("Raw ROC (AUC {:.3})", eval.raw_metrics.roc_auc),
                    raw_pts,
                )
                .color(palette.series_color(1))
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    format!("Recalibrated ROC (AUC {:.3})", eval.cal_metrics.roc_auc),
                    cal_pts,
                )
                .color(palette.series_color(0))
                .width(2.5_f32),
            );
        });

    let summary = format!(
        "ROC-AUC Raw {:.4}, Recalibrated {:.4}",
        eval.raw_metrics.roc_auc, eval.cal_metrics.roc_auc
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "Receiver Operating Characteristic Curve",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
}

fn render_pr_plot(ui: &mut egui::Ui, eval: &WorkbenchEvaluation, palette: &SemanticPalette) {
    let raw_pts: Vec<[f64; 2]> = eval
        .raw_pr
        .iter()
        .map(|p| [p.recall, p.precision])
        .collect();
    let cal_pts: Vec<[f64; 2]> = eval
        .cal_pr
        .iter()
        .map(|p| [p.recall, p.precision])
        .collect();
    let prev = eval.prevalence;

    let resp = Plot::new("calibration_workbench_pr")
        .legend(Legend::default())
        .x_axis_label("Recall (Sensitivity)")
        .y_axis_label("Precision (Positive Predictive Value)")
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(
                    format!("Prevalence Baseline ({prev:.2})"),
                    vec![[0.0, prev], [1.0, prev]],
                )
                .color(palette.graph_edge)
                .style(LineStyle::dashed_dense()),
            );
            plot_ui.line(
                Line::new(
                    format!("Raw PR (AP {:.3})", eval.raw_metrics.pr_auc_step),
                    raw_pts,
                )
                .color(palette.series_color(1))
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    format!("Recalibrated PR (AP {:.3})", eval.cal_metrics.pr_auc_step),
                    cal_pts,
                )
                .color(palette.series_color(0))
                .width(2.5_f32),
            );
        });

    let summary = format!(
        "Step PR-AUC Raw {:.4}, Recalibrated {:.4}",
        eval.raw_metrics.pr_auc_step, eval.cal_metrics.pr_auc_step
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "Precision-Recall Threshold Sweep Curve",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
}

fn render_crps_plot(ui: &mut egui::Ui, eval: &WorkbenchEvaluation, palette: &SemanticPalette) {
    let resp = Plot::new("calibration_workbench_crps")
        .legend(Legend::default())
        .x_axis_label("Observation Offset (y - μ)")
        .y_axis_label("Continuous Ranked Probability Score (CRPS)")
        .include_y(0.0)
        .show(ui, |plot_ui| {
            plot_ui.line(
                Line::new(
                    format!("Raw Gaussian CRPS (mean {:.3})", eval.mean_raw_crps),
                    eval.crps_raw_curve.clone(),
                )
                .color(palette.series_color(1))
                .width(2.0_f32),
            );
            plot_ui.line(
                Line::new(
                    format!("Calibrated Gaussian CRPS (mean {:.3})", eval.mean_cal_crps),
                    eval.crps_cal_curve.clone(),
                )
                .color(palette.series_color(0))
                .width(2.5_f32),
            );
            plot_ui.line(
                Line::new(
                    "Sorted Empirical Ensemble CRPS (M = 31)",
                    eval.crps_ens_curve.clone(),
                )
                .color(palette.series_color(2))
                .style(LineStyle::dashed_dense())
                .width(2.0_f32),
            );
        });

    let summary = format!(
        "Mean Gaussian CRPS Raw {:.4}, Calibrated {:.4}",
        eval.mean_raw_crps, eval.mean_cal_crps
    );
    register_canvas_accessibility(
        ui,
        &resp.response,
        Role::GraphicsDocument,
        "Gaussian and Empirical Ensemble CRPS Curve",
        Some(&summary),
    );
    paint_canvas_focus_ring(ui, &resp.response);
}

// [cite:clinical_trials_statistics]
