use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};

pub struct Co2ProjectionsTool {
    reduction_scenario: f64,
    synthetic_walk: bool,
}

impl Default for Co2ProjectionsTool {
    fn default() -> Self {
        Self {
            reduction_scenario: 0.0,
            synthetic_walk: false,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Co2State {
    reduction_scenario: f64,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    synthetic_walk: bool,
}

impl InteractiveTool for Co2ProjectionsTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "CO2 Projections"
    }
    fn domain(&self) -> &'static str {
        "climate"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&Co2State {
            reduction_scenario: self.reduction_scenario,
            synthetic_walk: self.synthetic_walk,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: Co2State =
            serde_json::from_str(state).map_err(|e| format!("Invalid CO2 state: {e}"))?;
        if !s.reduction_scenario.is_finite() || !(0.0..=1.0).contains(&s.reduction_scenario) {
            return Err("reduction_scenario must be finite within [0.0, 1.0]".into());
        }
        self.reduction_scenario = s.reduction_scenario;
        self.synthetic_walk = s.synthetic_walk;
        Ok(())
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        egui::CentralPanel::default().show_inside(ui, |ui| {
            ui.heading("Global CO2 Concentration Projections");
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut self.reduction_scenario, 0.0..=1.0)
                        .text("Emission Reduction Scenario"),
                );
                ui.checkbox(&mut self.synthetic_walk, "Synthetic Random Walk");
            });

            let (historical, projected) = if self.synthetic_walk {
                let mut rng = oxidize_core::rng::OxidizeRng::default();
                math_explorer::climate::dataset::get_co2_projections_with_rng(
                    &mut rng,
                    self.reduction_scenario,
                )
            } else {
                math_explorer::climate::dataset::get_co2_projections(self.reduction_scenario)
            };

            let plot = Plot::new("co2_projections_plot")
                .view_aspect(2.0)
                .x_axis_formatter(|x, _range| format!("{:.0}", x.value))
                .y_axis_formatter(|y, _range| format!("{:.1} ppm", y.value))
                .legend(egui_plot::Legend::default());

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            plot.show(ui, |plot_ui| {
                let hist_points = PlotPoints::new(historical);
                let hist_line = Line::new("Historical CO2", hist_points)
                    .width(2.0_f32)
                    .color(palette.series_color(0));
                plot_ui.line(hist_line);

                let proj_points = PlotPoints::new(projected);
                let proj_line = Line::new("Projected CO2", proj_points)
                    .width(2.0_f32)
                    .color(palette.series_color(3));
                plot_ui.line(proj_line);
            });
        });
    }
}

// [cite:isosurface_extraction]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "Co2ProjectionsTool",
        display_name: "CO2 Projections",
        domain: "climate",
        tags: &["carbon", "emissions", "radiative forcing", "greenhouse", "atmosphere"],
        build: || Box::new(Co2ProjectionsTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for Co2ProjectionsTool {
    fn theory_description(&self) -> String {
        format!(
            "Atmospheric carbon cycle mass balance model projecting CO2 concentration trajectories under an annual emissions reduction rate of {:.1}%: $C(t) = C_0 + \\int_0^t (E(t') - S(t')) dt'$.",
            self.reduction_scenario * 100.0
        )
    }
    fn phonetic_description(&self) -> String {
        "Atmospheric carbon dioxide projection and emissions scenario theory".into()
    }
    fn theory_citation(&self) -> String {
        "IPCC Sixth Assessment Report: The Physical Science Basis (IPCC, 2021)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "CO2 Emissions Trajectory",
    "domain": "climate",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "reduction_scenario": 0.35
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "co2_projections_starter",
        title: "CO2 Emissions Trajectory",
        description: "Historical and projected atmospheric CO2 concentration curves under emission reduction scenarios.",
        category: "Atmospheric Physics",
        icon: "🏭",
        domain: "climate",
        tool_name: Some("Co2ProjectionsTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scientific_metadata::theory::TheoryDescribable;

    #[test]
    fn test_co2_percentage_formatting_and_single_heading_slider_label() {
        let mut tool = Co2ProjectionsTool {
            reduction_scenario: 0.35,
            synthetic_walk: false,
        };
        assert!(tool.theory_description().contains("35.0%"));

        let ctx = egui::Context::default();
        ctx.enable_accesskit();
        let out = ctx.run(
            egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::Pos2::ZERO,
                    egui::vec2(1000.0, 700.0),
                )),
                ..Default::default()
            },
            |ctx| tool.show(ctx),
        );
        let mut stack: Vec<&egui::epaint::Shape> = out.shapes.iter().map(|c| &c.shape).collect();
        let mut headings = 0;
        let mut slider_labels = 0;
        while let Some(shape) = stack.pop() {
            match shape {
                egui::epaint::Shape::Text(t) => match t.galley.text() {
                    "Global CO2 Concentration Projections" => headings += 1,
                    "Emission Reduction Scenario" => slider_labels += 1,
                    _ => {}
                },
                egui::epaint::Shape::Vec(v) => stack.extend(v.iter()),
                _ => {}
            }
        }
        assert_eq!(headings, 1);
        assert_eq!(slider_labels, 1);
    }
}
