use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub struct Co2ProjectionsTool {
    reduction_scenario: f64,
}

impl Default for Co2ProjectionsTool {
    fn default() -> Self {
        Self {
            reduction_scenario: 0.0,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Co2State {
    reduction_scenario: f64,
}

impl InteractiveTool for Co2ProjectionsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "CO2 Projections"
    }
    fn domain(&self) -> &'static str {
        "climate"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&Co2State {
            reduction_scenario: self.reduction_scenario,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: Co2State = serde_json::from_str(state)
            .map_err(|e| format!("Invalid Co2ProjectionsTool state: {e}"))?;
        self.reduction_scenario = s.reduction_scenario;
        Ok(())
    }

    fn show(&mut self, ctx: &egui::Context) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            self.show_ui(ui);
        });
    }
    fn show_ui(&mut self, ui: &mut egui::Ui) {
        ui.add(
            egui::Slider::new(&mut self.reduction_scenario, 0.0..=1.0).text("Global CO2 Concentration Projections")
                .text("Emission Reduction Scenario"),
        );

        let (historical, projected) =
            math_explorer::climate::dataset::get_co2_projections(self.reduction_scenario);

        let plot = Plot::new("co2_projections_plot")
            .view_aspect(2.0)
            .x_axis_formatter(|x, _range| format!("{:.0}", x.value))
            .y_axis_formatter(|y, _range| format!("{:.1} ppm", y.value))
            .legend(egui_plot::Legend::default());

        plot.show(ui, |plot_ui| {
            let hist_points = PlotPoints::new(historical);
            let hist_line = Line::new("Historical CO2", hist_points).width(2.0_f32);
            plot_ui.line(hist_line);

            let proj_points = PlotPoints::new(projected);
            let proj_line = Line::new("Projected CO2", proj_points)
                .width(2.0_f32)
                .color(egui::Color32::RED);
            plot_ui.line(proj_line);
        });
    }
}

// [cite:isosurface_extraction]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "Co2ProjectionsTool",
        domain: "climate",
        tags: &[],
        build: || Box::new(Co2ProjectionsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for Co2ProjectionsTool {
    fn theory_description(&self) -> String {
        format!(
            "Atmospheric carbon cycle mass balance model projecting CO2 concentration trajectories under an annual emissions reduction rate of {:.1}%: $C(t) = C_0 + \\int_0^t (E(t') - S(t')) dt'$.",
            self.reduction_scenario
        )
    }
    fn phonetic_description(&self) -> String { "Atmospheric carbon dioxide projection and emissions scenario theory".into() }
    fn theory_citation(&self) -> String { "IPCC Sixth Assessment Report: The Physical Science Basis (IPCC, 2021)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CO2 Emissions Trajectory",
    "domain": "climate",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "reduction_scenario": 0.5
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "co2_projections_starter",
        title: "CO2 Emissions Trajectory",
        description: "Atmospheric carbon concentration forecasting across shared socioeconomic pathways (SSPs).",
        category: "Atmospheric Physics",
        icon: "🏭",
        domain: "climate",
        tool_name: Some("Co2ProjectionsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
