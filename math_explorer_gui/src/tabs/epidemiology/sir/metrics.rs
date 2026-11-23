#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Line, LineStyle, Plot, PlotPoint, PlotPoints, VLine};
use std::sync::Arc;

use super::{SirTool, MAX_SIR_DURATION, MIN_SIR_DURATION};
use crate::reflective_ui::render_copyable_metric;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

/// Computes interpolated `(peak_day, peak_infected)` from simulated `Infected` points.
pub(crate) fn compute_epidemic_peak(i_points: &[PlotPoint]) -> Option<(f64, f64)> {
    math_explorer::poi::peak_extremum(i_points).map(|[day, peak]| (day, peak))
}

/// Renders copyable epidemic summary metrics (`R₀`, `Peak Infected`, `Peak Day`, and branch peak reduction).
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
        if let Some(ref twin) = tool.branch.shadow {
            if let Some((base_day, base_peak)) = compute_epidemic_peak(&twin.i_points) {
                let delta_peak = peak_infected - base_peak;
                let pct = if base_peak.abs() > 1.0e-9 {
                    (delta_peak / base_peak) * 100.0
                } else {
                    0.0
                };
                render_copyable_metric(
                    ui,
                    "Shadow Baseline Peak Infected",
                    &format!("{base_peak:.1} (Day {base_day:.2})"),
                );
                render_copyable_metric(
                    ui,
                    "Counterfactual Peak ΔI",
                    &format!("{delta_peak:+.1} ({pct:+.1}%)"),
                );
            }
        }
    }
}

fn render_sir_shadow_overlay(
    plot_ui: &mut egui_plot::PlotUi<'_>,
    twin: &super::branch::SirShadowTwin,
    palette: crate::app::SemanticPalette,
) {
    for (label, series, color) in [
        ("Susceptible (Shadow)", &twin.display_s, palette.axis_z),
        ("Infected (Shadow)", &twin.display_i, palette.axis_x),
        ("Recovered (Shadow)", &twin.display_r, palette.axis_y),
    ] {
        plot_ui.line(
            Line::new(label, PlotPoints::from_shared(Arc::clone(series)))
                .color(color)
                .style(LineStyle::Dashed { length: 7.0_f32 }),
        );
    }
    plot_ui.vline(
        VLine::new(format!("Fork @ Day {:.1}", twin.fork_time), twin.fork_time)
            .color(palette.accent)
            .style(LineStyle::Dashed { length: 4.0_f32 }),
    );
}

/// Renders the SIR simulation plot with optional shadow-twin dashed polylines and fork marker.
pub(super) fn render_simulation_plot(ui: &mut egui::Ui, tool: &SirTool) {
    let plot_id = ui.make_persistent_id("sir_plot");
    ui.heading("Simulation");

    let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
    let plot_height = (ui.available_width() * 0.5).clamp(220.0, 340.0);
    let plot = Plot::new("sir_plot")
        .id(plot_id)
        .height(plot_height)
        .legend(egui_plot::Legend::default());

    plot.show(ui, |plot_ui| {
        if let Some(ref twin) = tool.branch.shadow {
            render_sir_shadow_overlay(plot_ui, twin, palette);
        }
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

impl TheoryDescribable for SirTool {
    fn theory_description(&self) -> String {
        let r0 = self.beta / self.gamma;
        format!(
            "Kermack-McKendrick SIR compartmental epidemic model: $\\frac{{dS}}{{dt}} = -\\frac{{\\beta S I}}{{N}}$, $\\frac{{dI}}{{dt}} = \\frac{{\\beta S I}}{{N}} - \\gamma I$, $\\frac{{dR}}{{dt}} = \\gamma I$. Transmission rate $\\beta = {:.3}$, recovery rate $\\gamma = {:.3}$, basic reproduction number $R_0 = {:.2}$.",
            self.beta, self.gamma, r0
        )
    }

    fn phonetic_description(&self) -> String {
        "Kermack-McKendrick SIR compartmental epidemic model theory".into()
    }

    fn theory_citation(&self) -> String {
        "A Contribution to the Mathematical Theory of Epidemics (Kermack & McKendrick, 1927)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        HashMap::new()
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        [
            ("N", 100.0, 100_000.0, 100.0),
            ("I0", 1.0, 50_000.0, 1.0),
            ("beta", 0.0, 5.0, 0.02),
            ("gamma", 0.0, 1.0, 0.01),
            ("duration", MIN_SIR_DURATION, MAX_SIR_DURATION, 1.0),
        ]
        .into_iter()
        .map(|(k, min, max, step)| (k.to_string(), ParameterConstraint { min, max, step }))
        .collect()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": { "schema_version": "1.0", "title": "Kermack-McKendrick SIR Model", "domain": "epidemiology", "timestamp": "2026-11-16T00:00:00Z" },
  "config": { "n": 10000.0, "i0": 25.0, "beta": 0.3, "gamma": 0.1, "duration": 120.0 }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "sir_starter",
        title: "Kermack-McKendrick SIR Model",
        description: "dS/dt = -βSI/N, dI/dt = βSI/N - γI: Compartmental epidemic outbreak dynamics and herd immunity.",
        category: "Compartmental Models",
        icon: "🦠",
        domain: "epidemiology",
        tool_name: Some("SirTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
