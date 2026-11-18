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
#[serde(deny_unknown_fields)]
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

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: BeamProfilingState =
            serde_json::from_str(state).map_err(|e| format!("Invalid BeamProfiling state: {e}"))?;
        if !loaded.incident_fluence.is_finite() || !(0.0..=200.0).contains(&loaded.incident_fluence)
        {
            return Err(format!(
                "incident_fluence out of [0.0, 200.0]: {}",
                loaded.incident_fluence
            ));
        }
        if !loaded.mu.is_finite() || !(0.0..=1.0).contains(&loaded.mu) {
            return Err(format!("mu out of [0.0, 1.0]: {}", loaded.mu));
        }
        self.incident_fluence = loaded.incident_fluence;
        self.mu = loaded.mu;
        Ok(())
    }

    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("beam_profiling_controls").show(ctx, |ui| {
            ui.heading("Beam Parameters");
            ui.separator();

            ui.add(egui::Slider::new(&mut self.incident_fluence, 0.0..=200.0).text("Incident Fluence (Psi_0)"));
            ui.add(egui::Slider::new(&mut self.mu, 0.0..=1.0).text("Attenuation Coeff. (mu)"));

            ui.separator();
            crate::reflective_ui::render_formula_with_export(
                ui,
                r"D(d) = \mu \Psi_0 e^{-\mu d}",
                Some("D(d) = mu * Psi_0 * exp(-mu * d)"),
            );

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
    fn theory_description(&self) -> String {
        "TERMA depth dose: $D(d) = \\mu \\Psi_0 e^{-\\mu d}$ represents the primary energy fluence released into the medium at depth $d$.".into()
    }
    fn phonetic_description(&self) -> String { "TERMA beam profiling theory".into() }
    fn theory_citation(&self) -> String { "The Physics of Radiation Therapy (Khan, 2014)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Radiotherapy TERMA Depth Profile",
    "domain": "medical",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "incident_fluence": 100.0,
    "mu": 0.1
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "beam_profiling_starter",
        title: "Radiotherapy TERMA Depth Profile",
        description: "D(d) = μΨ₀e^(-μd): Primary photon energy fluence attenuation depth-dose profile.",
        category: "Medical Physics",
        icon: "🔦",
        domain: "medical",
        tool_name: Some("BeamProfilingTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
