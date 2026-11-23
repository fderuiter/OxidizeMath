#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use eframe::egui;
use egui_plot::{Line, Plot, PlotPoint, PlotPoints};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::sync::Arc;

use super::{SirTool, MAX_SIR_DURATION, MIN_SIR_DURATION};
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

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "SirTool",
        display_name: "SIR Model",
        domain: "epidemiology",
        tags: &[],
        build: || Box::new(SirTool::default()),
    }
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
        let mut map = HashMap::with_capacity(5);
        map.insert(
            "N".to_string(),
            ParameterConstraint {
                min: 100.0,
                max: 100_000.0,
                step: 100.0,
            },
        );
        map.insert(
            "I0".to_string(),
            ParameterConstraint {
                min: 1.0,
                max: 50_000.0,
                step: 1.0,
            },
        );
        map.insert(
            "beta".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 5.0,
                step: 0.02,
            },
        );
        map.insert(
            "gamma".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        );
        map.insert(
            "duration".to_string(),
            ParameterConstraint {
                min: MIN_SIR_DURATION,
                max: MAX_SIR_DURATION,
                step: 1.0,
            },
        );
        map
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
