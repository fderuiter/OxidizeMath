//! Unit tests verifying the `HarmonicOscillatorTool` example in `docs/architecture/interactive_tool_walkthrough.md`.

use crate::accessibility::{paint_canvas_focus_ring, register_canvas_accessibility};
use crate::framework::{scoped_panel_id, InteractiveTool, PaneSidePanel, ToolMetadata};
use crate::starter_gallery::StarterPresetMetadata;
use eframe::egui;
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct HarmonicOscillatorState {
    mass: f32,
    spring_k: f32,
    damping: f32,
}

struct HarmonicOscillatorTool {
    mass: f32,
    spring_k: f32,
    damping: f32,
    position: f32,
    velocity: f32,
}

impl Default for HarmonicOscillatorTool {
    fn default() -> Self {
        Self {
            mass: 1.0,
            spring_k: 10.0,
            damping: 0.1,
            position: 1.0,
            velocity: 0.0,
        }
    }
}

impl HarmonicOscillatorTool {
    fn advance_step(&mut self, dt: f32) {
        assert!(self.mass > 0.0 && dt.is_finite());
        let acceleration =
            (-self.spring_k * self.position - self.damping * self.velocity) / self.mass;
        self.velocity += acceleration * dt;
        self.position += self.velocity * dt;
    }

    fn serialize_state_json(&self) -> Option<String> {
        serde_json::to_string(&HarmonicOscillatorState {
            mass: self.mass,
            spring_k: self.spring_k,
            damping: self.damping,
        })
        .ok()
    }

    fn restore_state_json(&mut self, state: &str) -> Result<(), String> {
        let loaded: HarmonicOscillatorState =
            serde_json::from_str(state).map_err(|e| format!("Invalid oscillator state: {e}"))?;
        if !loaded.mass.is_finite() || !(0.1..=10.0).contains(&loaded.mass) {
            return Err("mass must be finite in [0.1, 10.0]".to_string());
        }
        if !loaded.spring_k.is_finite() || !(0.1..=50.0).contains(&loaded.spring_k) {
            return Err("spring_k must be finite in [0.1, 50.0]".to_string());
        }
        if !loaded.damping.is_finite() || !(0.0..=2.0).contains(&loaded.damping) {
            return Err("damping must be finite in [0.0, 2.0]".to_string());
        }
        self.mass = loaded.mass;
        self.spring_k = loaded.spring_k;
        self.damping = loaded.damping;
        self.position = 1.0;
        self.velocity = 0.0;
        Ok(())
    }
}

impl TheoryDescribable for HarmonicOscillatorTool {
    fn theory_description(&self) -> String {
        format!(
            "Second-order linear damped harmonic oscillator with mass m = {:.2} kg, spring stiffness k = {:.2} N/m, and viscous damping c = {:.2} N·s/m.",
            self.mass, self.spring_k, self.damping
        )
    }

    fn phonetic_description(&self) -> String {
        "Damped harmonic oscillator governed by Hooke's restoring force and linear viscous damping"
            .to_string()
    }

    fn theory_citation(&self) -> String {
        "Landau, L. D., & Lifshitz, E. M. (1976). Mechanics (3rd ed.). Butterworth-Heinemann."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("mass".to_string(), "Inertial mass m (kg)".to_string());
        map.insert(
            "spring_k".to_string(),
            "Hookean spring stiffness k (N/m)".to_string(),
        );
        map.insert(
            "damping".to_string(),
            "Linear viscous damping coefficient c (N·s/m)".to_string(),
        );
        map
    }

    fn theory_parameters(&self) -> HashMap<String, ParameterConstraint> {
        let mut params = HashMap::new();
        params.insert(
            "mass".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 10.0,
                step: 0.1,
            },
        );
        params.insert(
            "spring_k".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 50.0,
                step: 0.5,
            },
        );
        params.insert(
            "damping".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 2.0,
                step: 0.05,
            },
        );
        params
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(r"m \ddot{x}(t) + c \dot{x}(t) + k x(t) = 0".to_string())
    }
}

impl InteractiveTool for HarmonicOscillatorTool {
    fn name(&self) -> &'static str {
        "Harmonic Oscillator"
    }

    fn domain(&self) -> &'static str {
        "physics"
    }

    fn theory(&self) -> &dyn TheoryDescribable {
        self
    }

    fn save_state(&self) -> Option<String> {
        self.serialize_state_json()
    }

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        self.restore_state_json(state)
    }

    fn show_inside(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = scoped_panel_id(parent_ui, "harmonic_oscillator_controls");
        PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Oscillator Parameters");
            ui.add(egui::Slider::new(&mut self.mass, 0.1..=10.0).text("Mass (kg)"));
            ui.add(egui::Slider::new(&mut self.spring_k, 0.1..=50.0).text("Spring Constant k"));
            ui.add(egui::Slider::new(&mut self.damping, 0.0..=2.0).text("Damping c"));

            ui.horizontal(|ui| {
                if ui.button("Step").clicked() {
                    self.advance_step(0.05);
                }
                if ui.button("Reset Position").clicked() {
                    self.position = 1.0;
                    self.velocity = 0.0;
                }
            });
        });

        egui::CentralPanel::default().show_inside(parent_ui, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::focusable_noninteractive());
            let summary = format!("x = {:.2} m, v = {:.2} m/s", self.position, self.velocity);
            register_canvas_accessibility(
                ui,
                &response,
                egui::accesskit::Role::GraphicsDocument,
                "Damped Harmonic Oscillator Canvas",
                Some(&summary),
            );
            paint_canvas_focus_ring(ui, &response);

            let center = response.rect.center();
            let bob_pos = egui::Pos2::new(center.x + self.position * 100.0, center.y);
            painter.line_segment(
                [center, bob_pos],
                egui::Stroke::new(2.0_f32, egui::Color32::GRAY),
            );
            painter.circle_filled(bob_pos, 15.0, egui::Color32::from_rgb(50, 150, 250));
        });
    }

    fn step_simulation(&mut self) {
        self.advance_step(0.05);
    }
}

const STARTER_PRESET_JSON: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Underdamped Harmonic Oscillator",
    "domain": "physics",
    "timestamp": "2026-11-18T00:00:00Z"
  },
  "config": {
    "mass": 2.5,
    "spring_k": 15.0,
    "damping": 0.25
  }
}"#;

#[test]
fn test_walkthrough_harmonic_oscillator_tool_contract_and_rendering() {
    let tool_meta = ToolMetadata {
        id: "HarmonicOscillatorTool",
        display_name: "Harmonic Oscillator",
        domain: "physics",
        tags: &["mechanics", "differential_equations", "oscillator"],
        build: || Box::new(HarmonicOscillatorTool::default()),
    };
    let starter_meta = StarterPresetMetadata {
        id: "harmonic_oscillator_starter",
        title: "Underdamped Harmonic Oscillator",
        description: "m·x'' + c·x' + k·x = 0: Damped Hookean spring-mass oscillation.",
        category: "Classical Mechanics",
        icon: "🌀",
        domain: "physics",
        tool_name: Some("HarmonicOscillatorTool"),
        preset_generator: || STARTER_PRESET_JSON,
    };

    let mut tool = (tool_meta.build)();
    assert_eq!(tool.name(), tool_meta.display_name);
    assert_eq!(tool.domain(), tool_meta.domain);
    assert!(tool.theory().theory_description().contains("1.00 kg"));
    assert!(!tool.theory().phonetic_description().is_empty());
    assert!(tool.theory().theory_citation().contains("Landau"));
    assert_eq!(tool.theory().available_descriptions().len(), 3);
    assert_eq!(tool.theory().theory_parameters().len(), 3);
    assert!(tool.theory().theory_latex_equation().is_some());

    let ctx = egui::Context::default();
    let _ = ctx.run(egui::RawInput::default(), |ctx| {
        tool.show(ctx);
    });

    tool.step_simulation();
    tool.import_preset_json((starter_meta.preset_generator)())
        .expect("starter preset JSON must import cleanly");
    let exported = tool
        .export_preset_json()
        .expect("preset export must succeed");
    assert!(exported.contains("2.5") && exported.contains("\"physics\""));
    assert!(tool
        .try_load_state(r#"{"mass":0.0,"spring_k":10.0,"damping":0.1}"#)
        .is_err());
    assert!(tool
        .try_load_state(r#"{"mass":1.0,"spring_k":10.0,"damping":0.1,"unknown":true}"#)
        .is_err());
}
