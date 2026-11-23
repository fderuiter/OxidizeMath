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
    if let Some((peak_day, peak_infected)) = compute_epidemic_peak(&tool.i_points) {
        render_copyable_metric(ui, "Peak Infected", &format!("{peak_infected:.1}"));
        render_copyable_metric(ui, "Peak Day", &format!("Day {peak_day:.2}"));
    }
}

/// Renders the live empirical network overlay toggle and heterogeneous R₀ badge inline.
fn render_network_overlay_controls(
    ui: &mut egui::Ui,
    tool: &mut SirTool,
    net_trace: Option<&crate::app::EpidemicTraceSignal>,
) {
    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let resp = ui
        .checkbox(
            &mut tool.show_network_overlay,
            "Overlay Network Epidemic Trace",
        )
        .on_hover_text(
            "Plot live empirical S(t), I(t), and R(t) trajectories streamed from Network Propagation.",
        );
    if resp.changed() {
        let status = if tool.show_network_overlay {
            "SIR Model: Network epidemic overlay enabled"
        } else {
            "SIR Model: Network epidemic overlay hidden"
        };
        crate::accessibility::announce_live_status(ui.ctx(), status);
    }
    if let Some(trace) = net_trace {
        ui.label(
            egui::RichText::new(format!(
                "Network rev #{} (N={}, R₀_het={:.2})",
                trace.revision, trace.population, trace.heterogeneous_r0
            ))
            .small()
            .color(palette.info),
        );
    } else {
        ui.label(
            egui::RichText::new("No active Network Propagation stream")
                .small()
                .color(palette.muted_text),
        );
    }
}

fn render_empirical_overlay_lines(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    palette: &crate::app::SemanticPalette,
    n_pop: f64,
    trace: &crate::app::EpidemicTraceSignal,
) {
    if trace.population == 0 {
        return;
    }
    let scale = n_pop / (trace.population as f64);
    let style = egui_plot::LineStyle::dashed_dense();
    let s_emp: Vec<[f64; 2]> = trace
        .susceptible
        .iter()
        .map(|p| [p[0], p[1] * scale])
        .collect();
    let i_emp: Vec<[f64; 2]> = trace
        .infected
        .iter()
        .map(|p| [p[0], p[1] * scale])
        .collect();
    let r_emp: Vec<[f64; 2]> = trace
        .recovered
        .iter()
        .map(|p| [p[0], p[1] * scale])
        .collect();
    plot_ui.line(
        Line::new("Network S (empirical)", s_emp)
            .color(palette.axis_z)
            .style(style),
    );
    plot_ui.line(
        Line::new("Network I (empirical)", i_emp)
            .color(palette.axis_x)
            .style(style),
    );
    plot_ui.line(
        Line::new("Network R (empirical)", r_emp)
            .color(palette.axis_y)
            .style(style),
    );
}

/// Renders the SIR simulation plot, CSV dataset import header, and inline network overlay toggle.
pub(super) fn render_simulation_plot(
    ui: &mut egui::Ui,
    tool: &mut SirTool,
    net_trace: Option<&crate::app::EpidemicTraceSignal>,
) {
    let plot_id = ui.make_persistent_id("sir_plot");
    ui.horizontal_wrapped(|ui| {
        ui.heading("Simulation");
        if ui.button("Import Dataset (CSV)").clicked() {
            egui_plot::export::trigger_import_csv_for_plot(ui.ctx(), plot_id);
        }
        ui.separator();
        render_network_overlay_controls(ui, tool, net_trace);
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
        if tool.show_network_overlay {
            if let Some(trace) = net_trace {
                render_empirical_overlay_lines(plot_ui, &palette, tool.n, trace);
            }
        }
        tool.snapshot_manager.render_on_plot(plot_ui);
    });
}

/// Evaluates and renders local parameter sensitivities for [`SirTool`].
pub(super) fn evaluate_sensitivity(ui: &mut egui::Ui, tool: &mut SirTool) {
    use math_explorer::epidemiology::compartmental::SIRModel;

    let params = [
        ("N", tool.n),
        ("I0", tool.i0),
        ("beta", tool.beta),
        ("gamma", tool.gamma),
    ];
    let duration_bits = crate::sensitivity::canonical_f64_bits(tool.duration);

    tool.sensitivity_engine
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

    crate::sensitivity::render_sensitivity_panel_ui(ui, &tool.sensitivity_engine, "sir_tool");
}
