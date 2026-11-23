use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use math_explorer::physics::fluid_dynamics::regimes::{
    FlatPlateClassifier, FlowClassifier, FlowRegime, PipeFlowClassifier,
};

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum Geometry {
    Pipe,
    FlatPlate,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TurbulenceState {
    density: f64,
    dynamic_viscosity: f64,
    velocity: f64,
    length: f64,
    geometry: Geometry,
}

pub struct TurbulenceTool {
    // Physics Parameters
    density: f64,           // kg/m^3
    dynamic_viscosity: f64, // Pa.s
    velocity: f64,          // m/s
    length: f64,            // m (Diameter or Length)
    geometry: Geometry,

    // Visualization State
    time: f64,
}

impl Default for TurbulenceTool {
    fn default() -> Self {
        Self {
            density: 1000.0,          // Water
            dynamic_viscosity: 0.001, // Water at 20C
            velocity: 1.0,
            length: 1.0,
            geometry: Geometry::Pipe,
            time: 0.0,
        }
    }
}

use crate::framework::InteractiveTool;

impl InteractiveTool for TurbulenceTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }
    fn name(&self) -> &'static str {
        "Turbulence / Reynolds Analysis"
    }

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&TurbulenceState {
            density: self.density,
            dynamic_viscosity: self.dynamic_viscosity,
            velocity: self.velocity,
            length: self.length,
            geometry: self.geometry,
        })
        .ok()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: TurbulenceState =
            serde_json::from_str(state).map_err(|e| format!("Invalid turbulence state: {e}"))?;
        if ![
            loaded.density,
            loaded.dynamic_viscosity,
            loaded.velocity,
            loaded.length,
        ]
        .iter()
        .all(|v| v.is_finite())
            || !(0.0..=20_000.0).contains(&loaded.density)
            || loaded.density <= 0.0
            || !(1e-6..=10.0).contains(&loaded.dynamic_viscosity)
            || !(0.0..=1000.0).contains(&loaded.velocity)
            || !(0.0..=100.0).contains(&loaded.length)
            || loaded.length <= 0.0
        {
            return Err("Turbulence fluid or geometry parameters out of bounds".into());
        }
        self.density = loaded.density;
        self.dynamic_viscosity = loaded.dynamic_viscosity;
        self.velocity = loaded.velocity;
        self.length = loaded.length;
        self.geometry = loaded.geometry;
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show_inside(&mut self, ui: &mut egui::Ui) {
        let ctx = &ui.ctx().clone();
        let dt = ctx.input(|i| i.stable_dt).min(0.1) as f64;
        self.update(dt);

        let mut reynolds_number = 0.0;

        // Calculate Reynolds Number
        // Re = (rho * v * L) / mu
        if self.dynamic_viscosity > 0.0 {
            reynolds_number = (self.density * self.velocity * self.length) / self.dynamic_viscosity;
        }

        // Classify Regime
        let regime = match self.geometry {
            Geometry::Pipe => {
                let classifier = PipeFlowClassifier;
                classifier.classify(reynolds_number)
            }
            Geometry::FlatPlate => {
                let classifier = FlatPlateClassifier;
                classifier.classify(reynolds_number)
            }
        };

        let panel_id = crate::framework::scoped_panel_id(ui, "turbulence_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(ui, |ui| {
            ui.heading("Turbulence Analysis");
            ui.separator();

            ui.label("Geometry");
            ui.horizontal(|ui| {
                ui.radio_value(&mut self.geometry, Geometry::Pipe, "Pipe Flow");
                ui.radio_value(&mut self.geometry, Geometry::FlatPlate, "Flat Plate");
            });

            ui.separator();
            ui.heading("Fluid Properties");
            ui.label("Density (kg/m³)");
            ui.add(
                egui::DragValue::new(&mut self.density)
                    .speed(10.0)
                    .range(0.0..=20000.0),
            );

            ui.label("Dynamic Viscosity (Pa·s)");
            ui.add(
                egui::DragValue::new(&mut self.dynamic_viscosity)
                    .speed(0.0001)
                    .range(1e-6..=10.0),
            );

            ui.separator();
            ui.heading("Flow Parameters");
            ui.label("Velocity (m/s)");
            ui.add(
                egui::DragValue::new(&mut self.velocity)
                    .speed(0.1)
                    .range(0.0..=1000.0),
            );

            ui.label(match self.geometry {
                Geometry::Pipe => "Diameter (m)",
                Geometry::FlatPlate => "Length (m)",
            });
            ui.add(
                egui::DragValue::new(&mut self.length)
                    .speed(0.01)
                    .range(0.0..=100.0),
            );

            ui.separator();
            ui.heading("Results");
            ui.label(format!("Reynolds Number: {:.2}", reynolds_number));

            let palette = crate::app::SemanticPalette::from_visuals(ui.visuals());
            let (regime_text, color) = match regime {
                FlowRegime::Laminar => ("Laminar", palette.success),
                FlowRegime::Transitional => ("Transitional", palette.warning),
                FlowRegime::Turbulent => ("Turbulent", palette.error),
            };
            ui.colored_label(color, format!("Regime: {}", regime_text));

            ui.separator();
            ui.label("Presets:");
            if ui.button("💧 Water in Pipe").clicked() {
                self.density = 1000.0;
                self.dynamic_viscosity = 0.001;
                self.geometry = Geometry::Pipe;
                self.length = 0.1;
                self.velocity = 0.5;
            }
            if ui.button("💨 Air over Wing").clicked() {
                self.density = 1.225;
                self.dynamic_viscosity = 1.81e-5;
                self.geometry = Geometry::FlatPlate;
                self.length = 1.0;
                self.velocity = 50.0;
            }
            if ui.button("🍯 Honey").clicked() {
                self.density = 1420.0;
                self.dynamic_viscosity = 10.0;
                self.velocity = 0.1;
            }
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            Plot::new("turbulence_plot")
                .data_aspect(1.0)
                .view_aspect(0.5)
                .show(ui, |plot_ui| {
                    // Visualize flow lines
                    let n_lines = 10;
                    let length = 10.0;

                    for i in 0..n_lines {
                        let y_base = (i as f64 - n_lines as f64 / 2.0) * 1.0;

                        let points: PlotPoints = (0..100)
                            .map(|j| {
                                let x = (j as f64 / 100.0) * length;

                                let noise = match regime {
                                    FlowRegime::Laminar => 0.0,
                                    FlowRegime::Transitional => {
                                        if x > length * 0.3 {
                                            0.2 * (3.0 * x + self.time * 5.0 + (i as f64)).sin()
                                        } else {
                                            0.0
                                        }
                                    }
                                    FlowRegime::Turbulent => {
                                        // Simple pseudo-randomness based on position and time
                                        let phase1 = x * 10.0 + self.time * 10.0 + (i as f64);
                                        let phase2 = x * 23.0 - self.time * 15.0 + (i as f64 * 2.0);
                                        let amp = (x / length).powf(0.5); // Turbulence increases downstream
                                        amp * (0.2 * phase1.sin()
                                            + 0.1 * phase2.sin()
                                            + 0.1 * (phase1 * 3.0).cos())
                                    }
                                };

                                [x, y_base + noise]
                            })
                            .collect();

                        // Line::new requires name as first argument in egui_plot 0.34
                        let line_color = crate::app::SemanticPalette::from_visuals(
                            &plot_ui.ctx().style().visuals,
                        )
                        .series_color(0);
                        plot_ui.line(Line::new("Streamline", points).color(line_color));
                    }
                });
        });

        // Request repaint for animation
        ctx.request_repaint();
    }
}

impl TurbulenceTool {
    pub fn update(&mut self, dt: f64) {
        self.time += dt;
    }
}

// [cite:generative_turbulence]

inventory::submit! {
    crate::framework::ToolMetadata {
        id: "TurbulenceTool",
        display_name: "Turbulence / Reynolds Analysis",
        domain: "fluid_dynamics",
        tags: &["reynolds", "kolmogorov", "laminar", "energy cascade", "eddy"],
        build: || Box::new(TurbulenceTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for TurbulenceTool {
    fn theory_description(&self) -> String {
        let re = (self.density * self.velocity * self.length) / self.dynamic_viscosity;
        let geom_str = match self.geometry {
            Geometry::Pipe => "pipe flow with critical threshold $Re_c \\approx 2300$",
            Geometry::FlatPlate => "boundary layer over a flat plate with transition threshold $Re_c \\approx 5 \\times 10^5$",
        };
        format!(
            "Fluid flow regime analysis for {} governed by Reynolds number $Re = \\frac{{\\rho v L}}{{\\mu}} = {:.1}$.",
            geom_str, re
        )
    }
    fn phonetic_description(&self) -> String {
        "Reynolds number and turbulence flow regime theory".into()
    }
    fn theory_citation(&self) -> String {
        "Turbulent Flows (Pope, 2000); Boundary-Layer Theory (Schlichting & Gersten, 2016)".into()
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Laminar".into(),
            "Viscous forces dominate, smooth streamline flow".into(),
        );
        map.insert(
            "Transitional".into(),
            "Onset of turbulent spots and velocity fluctuations".into(),
        );
        map.insert(
            "Turbulent".into(),
            "Inertial forces dominate, chaotic eddy vortex cascades".into(),
        );
        map
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Reynolds Flow Regime Analysis",
    "domain": "fluid_dynamics",
    "timestamp": "2026-11-16T00:00:00Z"
  },
  "config": {
    "density": 1.225,
    "dynamic_viscosity": 0.0000181,
    "velocity": 25.0,
    "length": 1.0,
    "geometry": "FlatPlate"
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turbulence_starter",
        title: "Reynolds Flow Regime Analysis",
        description: "Re = ρvL/μ: Laminar, transitional, and turbulent flow classification in pipe and flat-plate geometries.",
        category: "Turbulent Flow",
        icon: "🌪️",
        domain: "fluid_dynamics",
        tool_name: Some("TurbulenceTool"),
        preset_generator: || STARTER_PRESET_JSON,
    }
}
