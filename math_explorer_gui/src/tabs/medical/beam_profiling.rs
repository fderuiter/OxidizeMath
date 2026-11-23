#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::framework::{CodeArtifact, ExportCapabilities, InteractiveTool};
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_commons::ast::{format_latex_scalar, format_python_float, format_rust_f64};
use math_explorer::physics::medical::dose::algorithm::calculate_terma;

pub struct BeamProfilingTool {
    pub(crate) incident_fluence: f64,
    pub(crate) mu: f64,
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
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Beam Profiling"
    }

    fn domain(&self) -> &'static str {
        "medical"
    }

    fn has_finite_parameters(&self) -> bool {
        self.incident_fluence.is_finite()
            && self.mu.is_finite()
            && (0.0..=200.0).contains(&self.incident_fluence)
            && (0.0..=1.0).contains(&self.mu)
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

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "beam_profiling_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
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

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
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

impl BeamProfilingTool {
    fn build_code_artifact(&self) -> Option<CodeArtifact> {
        if !self.has_finite_parameters() {
            return None;
        }
        let psi_tex = format_latex_scalar(self.incident_fluence).ok()?;
        let mu_tex = format_latex_scalar(self.mu).ok()?;
        let symbolic_latex =
            "\\begin{aligned}\nD(d) &= \\mu \\cdot \\Psi_0 \\cdot e^{-\\mu d}\n\\end{aligned}"
                .to_string();
        let calibrated_latex = format!(
            "\\begin{{aligned}}\nD(d) &= {mu_tex} \\cdot {psi_tex} \\cdot e^{{-{mu_tex} \\cdot d}}, \\quad d \\in [0, 29.9]\\,\\text{{cm}}\n\\end{{aligned}}"
        );

        let psi_rs = format_rust_f64(self.incident_fluence).ok()?;
        let mu_rs = format_rust_f64(self.mu).ok()?;
        let rust_source = format!(
            "// Standalone Rust 2024 TERMA beam depth-dose evaluator generated by OxidizeMath\n\
             pub const INCIDENT_FLUENCE: f64 = {psi_rs};\n\
             pub const MU: f64 = {mu_rs};\n\n\
             pub fn terma_dose(depth_cm: f64) -> f64 {{\n\
             \x20   MU * INCIDENT_FLUENCE * (-MU * depth_cm).exp()\n\
             }}\n\n\
             fn main() {{\n\
             \x20   for i in 0..300 {{\n\
             \x20       let d = (i as f64) * 0.1;\n\
             \x20       println!(\"{{d:.2}},{{:.9}}\", terma_dose(d));\n\
             \x20   }}\n\
             }}\n"
        );

        let psi_py = format_python_float(self.incident_fluence).ok()?;
        let mu_py = format_python_float(self.mu).ok()?;
        let python_source = format!(
            "\"\"\"Standalone Python TERMA depth-dose evaluator generated by OxidizeMath.\"\"\"\n\
             import numpy as np\n\n\
             INCIDENT_FLUENCE = {psi_py}\n\
             MU = {mu_py}\n\n\
             def terma_dose(depth_cm: float | np.ndarray):\n\
             \x20   return MU * INCIDENT_FLUENCE * np.exp(-MU * depth_cm)\n\n\
             if __name__ == \"__main__\":\n\
             \x20   depths = np.arange(300, dtype=np.float64) * 0.1\n\
             \x20   doses = terma_dose(depths)\n\
             \x20   print(f\"Surface TERMA D(0) = {{float(doses[0]):.9f}}, D(10cm) = {{float(terma_dose(10.0)):.9f}}\")\n"
        );

        Some(CodeArtifact {
            symbolic_latex,
            calibrated_latex,
            rust_source,
            python_source,
        })
    }
}

// [cite:dwarf_galaxy_empirical_dependencies]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "BeamProfilingTool",
        display_name: "Beam Profiling",
        domain: "medical",
        tags: &[],
        build: || Box::new(BeamProfilingTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for BeamProfilingTool {
    fn theory_description(&self) -> String {
        "TERMA depth dose: $D(d) = \\mu \\Psi_0 e^{-\\mu d}$ represents the primary energy fluence released into the medium at depth $d$.".into()
    }
    fn phonetic_description(&self) -> String {
        "TERMA beam profiling theory".into()
    }
    fn theory_citation(&self) -> String {
        "The Physics of Radiation Therapy (Khan, 2014)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        std::collections::HashMap::new()
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Radiotherapy TERMA Depth Profile",
    "domain": "medical",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "incident_fluence": 150.0,
    "mu": 0.06
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
