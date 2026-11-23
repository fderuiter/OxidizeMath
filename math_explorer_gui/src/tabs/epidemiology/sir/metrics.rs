#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints};
use std::sync::Arc;

use super::SirTool;
use crate::reflective_ui::render_copyable_metric;

/// Computes interpolated `(peak_day, peak_infected)` from simulated `Infected` points.
pub(crate) fn compute_epidemic_peak(i_points: &[PlotPoint]) -> Option<(f64, f64)> {
    let (idx, max_pt) = i_points
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, p)| p.x.is_finite() && p.y.is_finite())
        .max_by(|(_, a), (_, b)| a.y.total_cmp(&b.y))?;

    if idx > 0 && idx + 1 < i_points.len() {
        let (p0, p1, p2) = (i_points[idx - 1], max_pt, i_points[idx + 1]);
        if p0.x.is_finite()
            && p0.y.is_finite()
            && p2.x.is_finite()
            && p2.y.is_finite()
            && p0.x < p1.x
            && p1.x < p2.x
            && (p1.y > p0.y || p1.y > p2.y)
        {
            let u0 = p0.x - p1.x;
            let u2 = p2.x - p1.x;
            let d0 = p0.y - p1.y;
            let d2 = p2.y - p1.y;
            let denom = u0 * u2 * (u2 - u0);
            if denom.is_finite() && denom.abs() > 1e-18 {
                let a = (u0 * d2 - u2 * d0) / denom;
                let b = (u2 * u2 * d0 - u0 * u0 * d2) / denom;
                if a.is_finite() && b.is_finite() && a < -1e-15 {
                    let u_v = (-b / (2.0 * a)).clamp(u0, u2);
                    let day = p1.x + u_v;
                    let peak = (p1.y + b * u_v + a * u_v * u_v).max(p1.y);
                    if day.is_finite() && peak.is_finite() {
                        return Some((day, peak));
                    }
                }
            }
        }
    }
    Some((max_pt.x, max_pt.y))
}

/// Renders copyable epidemic summary metrics (`R₀`, `Peak Infected`, and `Peak Day`).
pub(super) fn render_epidemic_metrics(ui: &mut egui::Ui, tool: &SirTool) {
    let r0 = if tool.gamma > 0.0 {
        tool.beta / tool.gamma
    } else {
        f64::INFINITY
    };
    render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{r0:.2}"));
    if let Some((peak_day, peak_infected)) = tool.epidemic_peak() {
        render_copyable_metric(ui, "Peak Infected", &format!("{peak_infected:.1}"));
        render_copyable_metric(ui, "Peak Day", &format!("Day {peak_day:.2}"));
    }
}

/// Renders the SIR simulation plot and CSV dataset import header.
pub(super) fn render_simulation_plot(ui: &mut egui::Ui, tool: &SirTool) {
    let plot_id = ui.make_persistent_id("sir_plot");
    ui.horizontal(|ui| {
        ui.heading("Simulation");
        if ui.button("Import Dataset (CSV)").clicked() {
            egui_plot::export::trigger_import_csv_for_plot(ui.ctx(), plot_id);
        }
    });

    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let plot = Plot::new("sir_plot")
        .id(plot_id)
        .view_aspect(2.0)
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        plot_ui.line(
            Line::new(
                "Susceptible",
                PlotPoints::from_shared(Arc::clone(&tool.display_s)),
            )
            .color(palette.axis_z),
        );
        plot_ui.line(
            Line::new(
                "Infected",
                PlotPoints::from_shared(Arc::clone(&tool.display_i)),
            )
            .color(palette.axis_x),
        );
        plot_ui.line(
            Line::new(
                "Recovered",
                PlotPoints::from_shared(Arc::clone(&tool.display_r)),
            )
            .color(palette.axis_y),
        );
        tool.snapshot_manager.render_on_plot(plot_ui);
    });
}
