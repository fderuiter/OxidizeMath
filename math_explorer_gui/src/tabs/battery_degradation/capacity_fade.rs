#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{CodeArtifact, ExportCapabilities, InteractiveTool};
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints};
use math_commons::ast::{format_latex_scalar, format_python_float, format_rust_f64};
use math_explorer::applied::battery_degradation::{
    Capacity, Cycles, DepthOfDischarge, PowerLawModel,
};

pub struct CapacityFadeTool {
    pub(crate) dod: f64,
    pub(crate) temperature: f64,
    pub(crate) cycles_to_simulate: f64,
}

impl Default for CapacityFadeTool {
    fn default() -> Self {
        Self {
            dod: 80.0,
            temperature: 25.0,
            cycles_to_simulate: 2000.0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityFadeState {
    dod: f64,
    temperature: f64,
    cycles_to_simulate: f64,
}

impl InteractiveTool for CapacityFadeTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Capacity Fade"
    }
    fn domain(&self) -> &'static str {
        "battery_degradation"
    }

    fn has_finite_parameters(&self) -> bool {
        self.dod.is_finite()
            && self.temperature.is_finite()
            && self.cycles_to_simulate.is_finite()
            && (0.0..=100.0).contains(&self.dod)
            && (-20.0..=60.0).contains(&self.temperature)
            && (100.0..=10_000.0).contains(&self.cycles_to_simulate)
    }

    fn export_capabilities(&self) -> ExportCapabilities {
        ExportCapabilities::plot_with_code()
    }

    fn export_code_artifact(&self) -> Option<CodeArtifact> {
        self.build_code_artifact()
    }

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&CapacityFadeState {
            dod: self.dod,
            temperature: self.temperature,
            cycles_to_simulate: self.cycles_to_simulate,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: CapacityFadeState =
            serde_json::from_str(state).map_err(|e| format!("Invalid capacity fade state: {e}"))?;
        if !s.dod.is_finite()
            || !(0.0..=100.0).contains(&s.dod)
            || !s.temperature.is_finite()
            || !(-20.0..=60.0).contains(&s.temperature)
            || !s.cycles_to_simulate.is_finite()
            || !(100.0..=10_000.0).contains(&s.cycles_to_simulate)
        {
            return Err("Capacity fade parameters out of bounds".into());
        }
        self.dod = s.dod;
        self.temperature = s.temperature;
        self.cycles_to_simulate = s.cycles_to_simulate;
        Ok(())
    }

    fn export_datasets(&self) -> Vec<(String, Vec<[f64; 2]>)> {
        vec![("Capacity".to_string(), self.capacity_curve_points())]
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.dod = self.dod.clamp(0.0, 100.0);
        self.cycles_to_simulate = self.cycles_to_simulate.max(100.0);
        self.render_controls_panel(ui);
        self.render_projection_panel(ui);
    }
}

impl CapacityFadeTool {
    pub(crate) fn capacity_curve_points(&self) -> Vec<[f64; 2]> {
        let model = PowerLawModel::standard();
        let dod_val = DepthOfDischarge::new(self.dod);
        (0..=100)
            .map(|i| {
                let cycle = (self.cycles_to_simulate / 100.0) * (i as f64);
                if let (Ok(c), Ok(d)) = (Cycles::new(cycle), dod_val.clone()) {
                    [cycle, model.capacity(c, d).as_f64()]
                } else {
                    [cycle, 0.0]
                }
            })
            .collect()
    }

    pub(crate) fn projected_life_cycles(dod_percent: f64) -> (f64, f64) {
        let model = PowerLawModel::standard();
        if let Ok(d) = DepthOfDischarge::new(dod_percent) {
            let n80 = model
                .cycles_to_capacity(Capacity::new_clamped(0.8), d)
                .as_f64();
            let n70 = model.n70(d).as_f64();
            (n80, n70)
        } else {
            (0.0, 0.0)
        }
    }

    fn render_controls_panel(&mut self, ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(ui, "battery_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Parameters");
            ui.add_space(5.0);
            ui.add(
                egui::Slider::new(&mut self.dod, 0.0..=100.0)
                    .text("Depth of Discharge (DoD) - %"),
            );
            ui.small("Percentage of battery capacity used per cycle.");
            ui.add_space(10.0);
            ui.add_enabled_ui(false, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.temperature, -20.0..=60.0)
                        .text("Temperature (°C)"),
                );
            })
            .response
            .on_disabled_hover_text(
                "Temperature dependency is not yet implemented in the core model.",
            );
            ui.add_space(10.0);
            ui.add(
                egui::Slider::new(&mut self.cycles_to_simulate, 100.0..=10000.0)
                    .text("Simulation Range - Cycles"),
            );
        });
    }

    fn render_projection_panel(&self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Capacity Fade Projection");
            ui.label("Estimation based on Power Law Model for Li-ion batteries.");

            let points = self.capacity_curve_points();

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            Plot::new("capacity_fade_plot")
                .view_aspect(2.0)
                .x_axis_label("Cycles")
                .y_axis_label("State of Health (Capacity)")
                .include_y(0.0)
                .include_y(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new("Capacity", PlotPoints::new(points)));
                    plot_ui.hline(
                        HLine::new("End of Life (70%)", 0.7)
                            .color(palette.error)
                            .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                    );
                    plot_ui.hline(
                        HLine::new("First Life (80%)", 0.8)
                            .color(palette.warning)
                            .style(egui_plot::LineStyle::Dashed { length: 10.0 }),
                    );
                });

            ui.add_space(10.0);
            let (n80, n70) = Self::projected_life_cycles(self.dod);
            render_copyable_metric(ui, "First Life (80%)", &format!("{n80:.0} cycles"));
            render_copyable_metric(
                ui,
                "Projected Cycle Life (to 70%)",
                &format!("{n70:.0} cycles"),
            );
        });
    }

    fn build_code_artifact(&self) -> Option<CodeArtifact> {
        if !self.has_finite_parameters() || self.dod <= 0.0 {
            return None;
        }
        let model = PowerLawModel::standard();
        let n70 = DepthOfDischarge::new(self.dod).ok().map(|d| model.n70(d).as_f64())?;
        let dod_tex = format_latex_scalar(self.dod).ok()?;
        let n70_tex = format_latex_scalar(n70).ok()?;
        let cyc_tex = format_latex_scalar(self.cycles_to_simulate).ok()?;

        let symbolic_latex = "\\begin{aligned}\nN_{70}(\\text{DoD}) &= \\alpha \\cdot \\text{DoD}^{\\beta} \\\\\nC(N) &= 0.7^{N / N_{70}(\\text{DoD})}\n\\end{aligned}".to_string();
        let calibrated_latex = format!(
            "\\begin{{aligned}}\nN_{{70}}({dod_tex}\\%) &= 101900 \\cdot {dod_tex}^{{-1.2639}} \\approx {n70_tex} \\\\\nC(N) &= 0.7^{{N / {n70_tex}}}, \\quad N \\in [0, {cyc_tex}]\n\\end{{aligned}}"
        );
        let rust_source = self.build_capacity_rust()?;
        let python_source = self.build_capacity_python()?;

        Some(CodeArtifact {
            symbolic_latex,
            calibrated_latex,
            rust_source,
            python_source,
        })
    }

    fn build_capacity_rust(&self) -> Option<String> {
        let dod_rs = format_rust_f64(self.dod).ok()?;
        let cyc_rs = format_rust_f64(self.cycles_to_simulate).ok()?;
        Some(format!(
            "// Standalone Rust 2024 Li-ion Power-Law capacity fade model generated by OxidizeMath\n\
             pub const ALPHA: f64 = 101900.0_f64;\n\
             pub const BETA: f64 = -1.2639_f64;\n\n\
             pub fn n70_cycles(dod_pct: f64) -> f64 {{\n\
             \x20   (ALPHA * dod_pct.clamp(0.0, 100.0).powf(BETA)).max(0.0)\n\
             }}\n\n\
             pub fn capacity_retention(cycles: f64, dod_pct: f64) -> f64 {{\n\
             \x20   let n70 = n70_cycles(dod_pct);\n\
             \x20   if n70 <= 0.0 {{\n\
             \x20       return 0.0;\n\
             \x20   }}\n\
             \x20   0.7_f64.powf(cycles.max(0.0) / n70).clamp(0.0, 1.0)\n\
             }}\n\n\
             fn main() {{\n\
             \x20   let dod: f64 = {dod_rs};\n\
             \x20   let max_cycles: f64 = {cyc_rs};\n\
             \x20   for i in 0..=100 {{\n\
             \x20       let c = (max_cycles / 100.0) * (i as f64);\n\
             \x20       println!(\"{{c:.1}},{{:.9}}\", capacity_retention(c, dod));\n\
             \x20   }}\n\
             }}\n"
        ))
    }

    fn build_capacity_python(&self) -> Option<String> {
        let dod_py = format_python_float(self.dod).ok()?;
        let cyc_py = format_python_float(self.cycles_to_simulate).ok()?;
        Some(format!(
            "\"\"\"Standalone Python Li-ion Power-Law capacity fade model generated by OxidizeMath.\"\"\"\n\
             import numpy as np\n\n\
             ALPHA = 101900.0\n\
             BETA = -1.2639\n\n\
             def n70_cycles(dod_pct: float) -> float:\n\
             \x20   return float(max(0.0, ALPHA * (np.clip(dod_pct, 0.0, 100.0) ** BETA)))\n\n\
             def capacity_retention(cycles: float | np.ndarray, dod_pct: float):\n\
             \x20   n70 = n70_cycles(dod_pct)\n\
             \x20   if n70 <= 0.0:\n\
             \x20       return np.zeros_like(cycles, dtype=np.float64)\n\
             \x20   return np.clip(0.7 ** (np.maximum(cycles, 0.0) / n70), 0.0, 1.0)\n\n\
             if __name__ == \"__main__\":\n\
             \x20   cycles = np.linspace(0.0, {cyc_py}, 101)\n\
             \x20   soh = capacity_retention(cycles, {dod_py})\n\
             \x20   print(f\"N70={{n70_cycles({dod_py}):.1f}} cycles, final SoH={{float(soh[-1]):.9f}}\")\n"
        ))
    }
}

// [cite:mmwave_radiotherapy_setup]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "CapacityFadeTool",
        display_name: "Capacity Fade",
        domain: "battery_degradation",
        tags: &[],
        build: || Box::new(CapacityFadeTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for CapacityFadeTool {
    fn theory_description(&self) -> String {
        format!(
            "Lithium-ion capacity fade power-law degradation: remaining capacity $C(N) = C_0 \\left(1 - 0.3 (N / N_{{70}})^z\\right)$ at {:.1}% depth of discharge and {:.1}°C operating temperature.",
            self.dod, self.temperature
        )
    }
    fn phonetic_description(&self) -> String { "Battery capacity fade power-law degradation theory".into() }
    fn theory_citation(&self) -> String { "Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Capacity Fade",
    "domain": "battery_degradation",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "dod": 80.0,
    "temperature": 25.0,
    "cycles_to_simulate": 2500.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "capacity_fade_starter",
        title: "Capacity Fade",
        description: "Power-law lithium-ion state-of-health capacity retention curve across charge-discharge cycles.",
        category: "Electrochemistry",
        icon: "🔋",
        domain: "battery_degradation",
        tool_name: Some("CapacityFadeTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::framework::InteractiveTool;

    #[test]
    fn test_first_life_80_threshold_error_and_power_law_fit() {
        let mut tool = CapacityFadeTool::default();
        let (n80, n70) = CapacityFadeTool::projected_life_cycles(tool.dod);
        assert!(n80 > 0.0 && n80 < n70);
        tool.cycles_to_simulate = (n70 * 1.2).max(2000.0);

        let datasets = tool.export_datasets();
        let cap_pts: Vec<egui_plot::PlotPoint> = datasets[0]
            .1
            .iter()
            .map(|p| egui_plot::PlotPoint::new(p[0], p[1]))
            .collect();
        let crossings = egui_plot::poi::interpolate_threshold_crossings(&cap_pts, 0.8);
        assert_eq!(crossings.len(), 1);
        assert!(
            (crossings[0] - n80).abs() < 0.5,
            "80% threshold cycle error = {} cycles",
            (crossings[0] - n80).abs()
        );

        let n70_curve: Vec<[f64; 2]> = (10..=100)
            .step_by(5)
            .map(|dod| {
                let (_, life70) = CapacityFadeTool::projected_life_cycles(dod as f64);
                [dod as f64, life70]
            })
            .collect();
        let fit = egui_plot::fit_model(
            &n70_curve,
            egui_plot::FitType::PowerLaw,
            "PowerLawModel::n70",
            false,
        )
        .expect("power law fit");
        assert!(fit.r_squared >= 0.99);
    }
}

