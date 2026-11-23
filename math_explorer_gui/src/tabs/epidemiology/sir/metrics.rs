#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints};
use std::sync::Arc;

use super::SirTool;
use crate::reflective_ui::render_copyable_metric;

/// Computes interpolated `(peak_day, peak_infected)` from simulated `Infected` points.
pub(crate) fn compute_epidemic_peak(i_points: &[PlotPoint]) -> Option<(f64, f64)> {
    math_explorer::poi::peak_extremum(i_points).map(|[day, peak]| (day, peak))
}

/// Renders copyable epidemic summary metrics (`R₀`, `Peak Infected`, and `Peak Day`).
pub(super) fn render_epidemic_metrics(ui: &mut egui::Ui, tool: &SirTool) {
    let r0 = if tool.gamma > 0.0 {
        tool.beta / tool.gamma
    } else {
        f64::INFINITY
    };
    render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{r0:.2}"));
    if let Some((peak_day, peak_infected)) = compute_epidemic_peak(&tool.i_points) {
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
    let plot_height = (ui.available_width() * 0.5).clamp(220.0, 340.0);
    let plot = Plot::new("sir_plot")
        .id(plot_id)
        .height(plot_height)
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
