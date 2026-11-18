use crate::accessibility::AccessibleTheoryHover;
use eframe::egui;
use egui_plot::{Bar, BarChart, Plot};
use scientific_metadata::theory::TheoryDescribable;
use math_explorer::physics::quantum::clebsch_gordan;
use std::collections::HashMap;

use crate::framework::InteractiveTool;

pub struct ClebschGordanTool {
    j1: f64,
    m1: f64,
    j2: f64,
    m2: f64,
    j: f64,
    m: f64,
    result: f64,
}

impl Default for ClebschGordanTool {
    fn default() -> Self {
        // Initialize with a known valid state (e.g. j1=1/2, j2=1/2 -> j=1, m=0)
        let mut tool = Self {
            j1: 0.5,
            m1: 0.5,
            j2: 0.5,
            m2: -0.5,
            j: 1.0,
            m: 0.0,
            result: 0.0,
        };
        tool.calculate();
        tool
    }
}

impl ClebschGordanTool {
    fn calculate(&mut self) {
        self.result = clebsch_gordan(self.j1, self.m1, self.j2, self.m2, self.j, self.m);
    }
}

impl TheoryDescribable for ClebschGordanTool {
    fn theory_description(&self) -> String {
        format!(
            "Clebsch-Gordan coefficient for j1={}, m1={}, j2={}, m2={}, J={}, M={} is {:.6}",
            self.j1, self.m1, self.j2, self.m2, self.j, self.m, self.result
        )
    }

    fn phonetic_description(&self) -> String {
        format!(
            "Clebsch-Gordan coefficient, Bra j1 equals {}, m1 equals {}, j2 equals {}, m2 equals {}, Ket J equals {}, M equals {}. The result is {:.6}",
            self.j1, self.m1, self.j2, self.m2, self.j, self.m, self.result
        )
    }

    fn theory_citation(&self) -> String {
        "Wigner, E. P. (1959). Group Theory and its Application to the Quantum Mechanics of Atomic Spectra. Academic Press; Sakurai, J. J., & Napolitano, J. (2020). Modern Quantum Mechanics (3rd ed.). Cambridge University Press.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Clebsch-Gordan coefficient".to_string(),
        );
        map
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ClebschGordanState {
    j1: f64,
    m1: f64,
    j2: f64,
    m2: f64,
    j: f64,
    m: f64,
}

impl InteractiveTool for ClebschGordanTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    
    fn name(&self) -> &'static str {
        "Clebsch-Gordan"
    }

    fn domain(&self) -> &'static str {
        "quantum"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&ClebschGordanState {
            j1: self.j1,
            m1: self.m1,
            j2: self.j2,
            m2: self.m2,
            j: self.j,
            m: self.m,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let s: ClebschGordanState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid ClebschGordanTool state: {e}"))?;
        if ![s.j1, s.m1, s.j2, s.m2, s.j, s.m].iter().all(|v| v.is_finite()) {
            return Err("Non-finite quantum numbers in ClebschGordanTool".to_string());
        }
        self.j1 = s.j1;
        self.m1 = s.m1;
        self.j2 = s.j2;
        self.m2 = s.m2;
        self.j = s.j;
        self.m = s.m;
        self.calculate();
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("clebsch_controls").show(ctx, |ui| {
            ui.heading("Parameters");
            ui.separator();

            let mut changed = false;

            ui.group(|ui| {
                ui.label("Angular Momentum 1");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.j1, 0.0..=10.0)
                            .step_by(0.5)
                            .text("j1"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.m1, -10.0..=10.0)
                            .step_by(0.5)
                            .text("m1"),
                    )
                    .changed();
            });

            ui.group(|ui| {
                ui.label("Angular Momentum 2");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.j2, 0.0..=10.0)
                            .step_by(0.5)
                            .text("j2"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.m2, -10.0..=10.0)
                            .step_by(0.5)
                            .text("m2"),
                    )
                    .changed();
            });

            ui.group(|ui| {
                ui.label("Total Angular Momentum");
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.j, 0.0..=20.0)
                            .step_by(1.0)
                            .text("J"),
                    )
                    .changed();
                changed |= ui
                    .add(
                        egui::Slider::new(&mut self.m, -20.0..=20.0)
                            .step_by(1.0)
                            .text("M"),
                    )
                    .changed();
            });

            if changed {
                self.calculate();
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Clebsch-Gordan Coefficient Calculator");
            ui.separator();

            // Formula display
            ui.label(
                egui::RichText::new(format!(
                    "⟨ j1, m1; j2, m2 | J, M ⟩ = ⟨ {:.1}, {:.1}; {:.1}, {:.1} | {:.1}, {:.1} ⟩",
                    self.j1, self.m1, self.j2, self.m2, self.j, self.m
                ))
                .size(20.0),
            );

            ui.add_space(20.0);

            ui.horizontal(|ui| {
                ui.label(
                    egui::RichText::new("Result (Amplitude): ")
                        .size(24.0)
                        .strong(),
                );
                ui.label(
                    egui::RichText::new(format!("{:.6}", self.result))
                        .size(24.0)
                        .color(egui::Color32::LIGHT_GREEN),
                );
            });

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("Probability (|C|²): ").size(20.0));
                ui.label(egui::RichText::new(format!("{:.6}", self.result.powi(2))).size(20.0));
            });

            ui.add_space(20.0);

            // Use Plot for visualization as requested
            let bar = Bar::new(0.0, self.result).width(0.5).name("Amplitude");
            let chart =
                BarChart::new("Coefficient Value", vec![bar]).color(egui::Color32::LIGHT_BLUE);

            let response = Plot::new("clebsch_plot")
                .view_aspect(2.0)
                .include_y(-1.0)
                .include_y(1.0)
                .show(ui, |plot_ui| {
                    plot_ui.bar_chart(chart);
                })
                .response;
            response.accessible_theory_hover(self);

            // Warnings / Info
            if (self.m1 + self.m2 - self.m).abs() > 1e-9 {
                ui.colored_label(
                    egui::Color32::RED,
                    "Warning: M must equal m1 + m2 for non-zero result.",
                );
            }

            let j_min = (self.j1 - self.j2).abs();
            let j_max = self.j1 + self.j2;
            if self.j < j_min || self.j > j_max {
                ui.colored_label(
                    egui::Color32::RED,
                    format!(
                        "Warning: J must be in range [{:.1}, {:.1}] (Triangle Inequality).",
                        j_min, j_max
                    ),
                );
            }
        });
    }
}

// [cite:quantum_mechanics]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "ClebschGordanTool",
        display_name: "Clebsch-Gordan",
        domain: "quantum",
        tags: &[],
        build: || Box::new(ClebschGordanTool::default()),
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Clebsch-Gordan Coefficients",
    "domain": "quantum",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "j1": 1.0,
    "m1": 1.0,
    "j2": 1.0,
    "m2": -1.0,
    "j": 2.0,
    "m": 0.0
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "clebsch_gordan_starter",
        title: "Clebsch-Gordan Coefficients",
        description: "⟨j₁m₁ j₂m₂ | J M⟩: Angular momentum addition and SU(2) group tensor product decomposition.",
        category: "Quantum Mechanics",
        icon: "⚛️",
        domain: "quantum",
        tool_name: Some("ClebschGordanTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
