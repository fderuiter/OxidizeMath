use crate::framework::InteractiveTool;
use crate::reflective_ui::render_copyable_metric;
use eframe::egui;
use egui_plot::{HLine, Line, Plot, PlotPoints};
use math_explorer::applied::battery_degradation::{
    Capacity, Cycles, DepthOfDischarge, PowerLawModel,
};

pub struct CapacityFadeTool {
    dod: f64,
    temperature: f64,
    cycles_to_simulate: f64,
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
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Capacity Fade"
    }
    fn domain(&self) -> &'static str {
        "battery_degradation"
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
                egui::Slider::new(&mut self.dod, 0.0..=100.0).text("Depth of Discharge (DoD) - %"),
            );
            ui.small("Percentage of battery capacity used per cycle.");
            ui.add_space(10.0);
            ui.add_enabled_ui(false, |ui| {
                ui.add(
                    egui::Slider::new(&mut self.temperature, -20.0..=60.0).text("Temperature (°C)"),
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
    fn phonetic_description(&self) -> String {
        "Battery capacity fade power-law degradation theory".into()
    }
    fn theory_citation(&self) -> String {
        "Capacity Fade Model for Li-ion Batteries (Smith et al., 2011)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
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
        let mut crossings = [0.0_f64; 4];
        let count =
            math_explorer::poi::threshold_crossings_into(&datasets[0].1, 0.8, &mut crossings);
        assert_eq!(count, 1);
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
        let fit = math_explorer::fit::fit_model(
            &n70_curve,
            math_explorer::fit::FitModel::PowerLaw,
            false,
        )
        .expect("power law fit");
        assert!(fit.r_squared >= 0.99);
    }
}
