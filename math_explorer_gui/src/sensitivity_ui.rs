#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! UI rendering and live-region status announcements for [`super::SensitivityEngine`].

use super::{SensitivityEngine, SensitivityMetric};
use eframe::egui;

/// Renders the collapsible Sensitivity & Perturbation analysis panel in the UI sidebar.
pub fn render_sensitivity_panel_ui(ui: &mut egui::Ui, engine: &SensitivityEngine, id_source: &str) {
    render_labeled_sensitivity_panel_ui(ui, engine, id_source, "Baseline Metric");
}

/// Renders the collapsible Sensitivity & Perturbation analysis panel with an explicit observable label.
pub fn render_labeled_sensitivity_panel_ui(
    ui: &mut egui::Ui,
    engine: &SensitivityEngine,
    id_source: &str,
    metric_label: &str,
) {
    let baseline_heading = if metric_label.is_empty() || metric_label == "Baseline Metric" {
        format!("Baseline Metric: {:.4}", engine.baseline_output)
    } else if metric_label.starts_with("Baseline ") {
        format!("{}: {:.4}", metric_label, engine.baseline_output)
    } else {
        format!("Baseline {}: {:.4}", metric_label, engine.baseline_output)
    };

    if !engine.is_converged() && !engine.gesture_deferred {
        ui.ctx().request_repaint();
    }
    sync_sensitivity_budget_announcement(engine);

    ui.add_space(6.0);
    egui::CollapsingHeader::new("Sensitivity & Perturbation Analysis")
        .id_salt(format!("{}_sensitivity_panel", id_source))
        .default_open(true)
        .show(ui, |ui| {
            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            if let Some(notice) = engine.budget_notice() {
                ui.colored_label(palette.warning, format!("ℹ️ {notice}"));
                ui.add_space(2.0);
            }
            ui.label(baseline_heading);
            ui.label(
                egui::RichText::new(
                    "Normalized elasticity |(dY/dP) * (P/Y)| via central finite diff (±1% delta)",
                )
                .weak()
                .small(),
            );
            ui.add_space(4.0);

            if engine.metrics.is_empty() {
                ui.label("No active parameters to analyze.");
                return;
            }

            for metric in &engine.metrics {
                render_sensitivity_metric_card(ui, metric, &palette);
                ui.add_space(2.0);
            }
        });
}

fn sync_sensitivity_budget_announcement(engine: &SensitivityEngine) {
    let mut last = engine.last_announced_budget.borrow_mut();
    match engine.budget_notice() {
        Some(msg) => {
            if last.as_deref() != Some(msg) {
                crate::accessibility::announce_status(msg);
                *last = Some(msg.to_string());
            }
        }
        None => {
            *last = None;
        }
    }
}

fn render_sensitivity_metric_card(
    ui: &mut egui::Ui,
    metric: &SensitivityMetric,
    palette: &crate::app::SemanticPalette,
) {
    ui.group(|ui| {
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(&metric.param_name).strong());
            if metric.is_high_sensitivity {
                ui.colored_label(palette.error, "⚠️ High Sensitivity (> 2.0 elasticity)");
            }
        });

        ui.horizontal(|ui| {
            ui.label(format!("dY/dP: {:.4}", metric.partial_derivative));
            ui.add_space(10.0);
            ui.label(format!("Elasticity: {:.2}", metric.elasticity));
        });

        let progress = (metric.elasticity / 3.0).clamp(0.0, 1.0) as f32;
        let color = crate::app::SemanticPalette::progress_bar_fill(ui.visuals(), metric.elasticity);

        let progress_bar = egui::ProgressBar::new(progress)
            .fill(color)
            .text(format!("{:.2} elasticity", metric.elasticity));
        ui.add(progress_bar);
    });
}
