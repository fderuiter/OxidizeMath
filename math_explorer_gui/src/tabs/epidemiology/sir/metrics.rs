#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints};
use math_explorer::epidemiology::compartmental::SIRModel;
use std::sync::Arc;

use super::{SirTool, SIR_STEP_DT};
use crate::app::SemanticPalette;
use crate::framework::{
    branch_a_ghost_line, branch_b_solid_line, fork_marker_callout, fork_marker_vline,
    render_divergence_strip, InteractiveTool, TraceSeries,
};
use crate::reflective_ui::render_copyable_metric;

impl SirTool {
    pub(super) fn evaluate_sensitivity(&mut self, ui: &mut egui::Ui) {
        let params = [
            ("N", self.n),
            ("I0", self.i0),
            ("beta", self.beta),
            ("gamma", self.gamma),
        ];
        let duration_bits = crate::sensitivity::canonical_f64_bits(self.duration);

        self.sensitivity_engine
            .compute_with_context(&params, &duration_bits, |p_map| {
                let n = p_map.get("N").copied().unwrap_or(1000.0).max(1.0);
                let i0 = p_map.get("I0").copied().unwrap_or(10.0).clamp(0.0, n);
                let beta = p_map.get("beta").copied().unwrap_or(0.5).max(0.0);
                let gamma = p_map.get("gamma").copied().unwrap_or(0.1).max(0.0001);

                if let Ok(mut model) = SIRModel::new(n, i0, beta, gamma) {
                    let dt = 0.2;
                    let mut max_i: f64 = 0.0;
                    for _ in 0..250 {
                        max_i = max_i.max(model.state().i);
                        model.step(dt);
                    }
                    max_i
                } else {
                    0.0
                }
            });

        crate::sensitivity::render_sensitivity_panel_ui(ui, &self.sensitivity_engine, "sir_tool");
    }

    pub(super) fn render_snapshot_bar(&mut self, ui: &mut egui::Ui) {
        let param_summary = self.param_summary();
        let current_len = self.snapshot_manager.len();
        let s_points = &self.s_points;
        let i_points = &self.i_points;
        let r_points = &self.r_points;
        if self.snapshot_manager.render_ui(ui, move || {
            let label = format!("Baseline {}", current_len + 1);
            let to_pairs = |pts: &[PlotPoint]| pts.iter().map(|p| [p.x, p.y]).collect::<Vec<_>>();
            let series = vec![
                TraceSeries::new("Susceptible", to_pairs(s_points)),
                TraceSeries::new("Infected", to_pairs(i_points)),
                TraceSeries::new("Recovered", to_pairs(r_points)),
            ];
            Some((label, param_summary, series))
        }) {
            self.revision = self.revision.wrapping_add(1).max(1);
        }
    }
}

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
    ui.horizontal_wrapped(|ui| {
        render_copyable_metric(ui, "Basic Reproduction Number (R₀)", &format!("{r0:.2}"));
        if let Some((peak_day, peak_infected)) = compute_epidemic_peak(&tool.i_points) {
            render_copyable_metric(ui, "Peak Infected", &format!("{peak_infected:.1}"));
            render_copyable_metric(ui, "Peak Day", &format!("Day {peak_day:.2}"));
        }
    });
}

/// Renders the SIR simulation plot.
pub(super) fn render_simulation_plot(ui: &mut egui::Ui, tool: &SirTool) {
    let plot_id = ui.make_persistent_id("sir_plot");
    ui.heading("Simulation");

    if let Some(ref snap) = tool.cf.branch_engine.active_snapshot {
        render_divergence_strip(ui, tool.name(), snap);
    }

    let palette = SemanticPalette::from_visuals(ui.visuals());
    let plot = Plot::new("sir_plot")
        .id(plot_id)
        .view_aspect(2.0)
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        render_sir_plot_curves(tool, plot_ui, &palette);
    });
}

fn render_sir_plot_curves<'a>(
    tool: &'a SirTool,
    plot_ui: &mut egui_plot::PlotUi<'a>,
    palette: &SemanticPalette,
) {
    let shared = |arc: &Arc<Vec<PlotPoint>>| PlotPoints::from_shared(Arc::clone(arc));
    if let Some(ref desc) = tool.cf.fork {
        plot_ui.line(branch_a_ghost_line(
            "Susceptible (Branch A)",
            shared(&tool.cf.display_s_a),
            palette.axis_z,
        ));
        plot_ui.line(branch_a_ghost_line(
            "Infected (Branch A)",
            shared(&tool.cf.display_i_a),
            palette.axis_x,
        ));
        plot_ui.line(branch_a_ghost_line(
            "Recovered (Branch A)",
            shared(&tool.cf.display_r_a),
            palette.axis_y,
        ));
        plot_ui.line(branch_b_solid_line(
            "Susceptible (Branch B)",
            shared(&tool.display_s),
            palette.axis_z,
        ));
        plot_ui.line(branch_b_solid_line(
            "Infected (Branch B)",
            shared(&tool.display_i),
            palette.axis_x,
        ));
        plot_ui.line(branch_b_solid_line(
            "Recovered (Branch B)",
            shared(&tool.display_r),
            palette.axis_y,
        ));
        let fork_t = desc.fork_step as f64 * SIR_STEP_DT;
        plot_ui.vline(fork_marker_vline(palette, fork_t));
        plot_ui.callout(fork_marker_callout(
            [fork_t, tool.cf.fork_infected_y],
            fork_t,
            palette,
        ));
    } else {
        plot_ui.line(Line::new("Susceptible", shared(&tool.display_s)).color(palette.axis_z));
        plot_ui.line(Line::new("Infected", shared(&tool.display_i)).color(palette.axis_x));
        plot_ui.line(Line::new("Recovered", shared(&tool.display_r)).color(palette.axis_y));
    }
    tool.snapshot_manager.render_on_plot(plot_ui);
}
