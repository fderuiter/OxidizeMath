use crate::framework::InteractiveTool;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::physics::medical::dose::algorithm::calculate_terma;

pub struct BeamProfilingTool {
    incident_fluence: f64,
    mu: f64,
}

impl Default for BeamProfilingTool {
    fn default() -> Self {
        Self {
            incident_fluence: 100.0,
            mu: 0.1,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct BeamProfilingState {
    incident_fluence: f64,
    mu: f64,
}

impl InteractiveTool for BeamProfilingTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Beam Profiling"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&BeamProfilingState {
            incident_fluence: self.incident_fluence,
            mu: self.mu,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(loaded) = serde_json::from_str::<BeamProfilingState>(state) {
            self.incident_fluence = loaded.incident_fluence;
            self.mu = loaded.mu;
        }
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("beam_profiling_controls").show(ctx, |ui| {
            ui.heading("Beam Parameters");
            ui.separator();

            ui.add(egui::Slider::new(&mut self.incident_fluence, 0.0..=200.0).text("Incident Fluence (Psi_0)"));
            ui.add(egui::Slider::new(&mut self.mu, 0.0..=1.0).text("Attenuation Coeff. (mu)"));

            ui.separator();
            ui.label("Formula: D(d) = mu * Psi_0 * exp(-mu * d)");

            ui.add_space(10.0);
            ui.label("TERMA (Total Energy Released per Mass) represents the primary energy fluence released into the medium at a point, before accounting for secondary electron transport (scatter).");
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let points: PlotPoints = (0..300)
                .map(|i| {
                    let depth = i as f64 * 0.1;
                    let dose =
                        calculate_terma(self.incident_fluence, self.mu, depth).unwrap_or(0.0);
                    [depth, dose]
                })
                .collect();

            let line = Line::new("Depth Dose", points);
            Plot::new("depth_dose_curve")
                .view_aspect(2.0)
                .x_axis_label("Depth (cm)")
                .y_axis_label("Dose (Arbitrary Units)")
                .show(ui, |plot_ui| plot_ui.line(line));
        });
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "BeamProfilingTool",
        domain: "medical",
        tags: &[],
        build: || Box::new(BeamProfilingTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BeamProfilingTool {
    fn theory_description(&self) -> String { "Theoretical context not available.".into() }
    fn phonetic_description(&self) -> String { "Theoretical context not available.".into() }
    fn theory_citation(&self) -> String { "Uncited".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}
