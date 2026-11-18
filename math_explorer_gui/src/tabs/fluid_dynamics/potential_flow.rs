use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Arrows, Plot};
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex,
};
use std::f64::consts::PI;

pub struct PotentialFlowTool {
    field: PotentialFlowField,
    // UI State
    grid_size: usize,
    view_range: f64,

    // Add Element State
    selected_element_type: ElementType,
    new_element_strength: f64,
    new_element_x: f64,
    new_element_y: f64,
    new_element_angle: f64,
}

#[derive(PartialEq, Clone, Copy, serde::Serialize, serde::Deserialize)]
enum ElementType {
    Uniform,
    Source,
    Sink,
    Vortex,
    Doublet,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct PotentialFlowState {
    grid_size: usize,
    view_range: f64,
    selected_element_type: ElementType,
    new_element_strength: f64,
    new_element_x: f64,
    new_element_y: f64,
    new_element_angle: f64,
}

impl Default for PotentialFlowTool {
    fn default() -> Self {
        let mut field = PotentialFlowField::new();
        // Default: Flow past a cylinder (Uniform + Doublet)
        field.add(Box::new(UniformFlow::new(10.0, 0.0)));
        // Radius R=2, U=10 => kappa = 2*pi*U*R^2 = 2*pi*10*4 = 80*pi approx 251.3
        let kappa = 2.0 * PI * 10.0 * 2.0 * 2.0;
        field.add(Box::new(Doublet::new(kappa, 0.0, 0.0)));

        Self {
            field,
            grid_size: 20,
            view_range: 10.0,
            selected_element_type: ElementType::Source,
            new_element_strength: 10.0,
            new_element_x: 0.0,
            new_element_y: 0.0,
            new_element_angle: 0.0,
        }
    }
}

use crate::framework::InteractiveTool;
use crate::presets::InteractivePreset;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug, PartialEq)]
pub struct PotentialFlowPresetConfig {
    pub preset_type: String,
    pub grid_size: usize,
    pub view_range: f64,
}

impl InteractivePreset for PotentialFlowTool {
    type Config = PotentialFlowPresetConfig;

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn title(&self) -> String {
        "Potential Flow - Cylinder Vector Field".to_string()
    }

    fn export_config(&self) -> Self::Config {
        PotentialFlowPresetConfig {
            preset_type: "CylinderFlow".to_string(),
            grid_size: self.grid_size,
            view_range: self.view_range,
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.grid_size = config.grid_size;
        self.view_range = config.view_range;
        if config.preset_type == "SourceSink" {
            let mut field = PotentialFlowField::new();
            field.add(Box::new(Source::new(10.0, -3.0, 0.0)));
            field.add(Box::new(Source::new(-10.0, 3.0, 0.0)));
            self.field = field;
        } else {
            let mut field = PotentialFlowField::new();
            field.add(Box::new(UniformFlow::new(10.0, 0.0)));
            let kappa = 2.0 * PI * 10.0 * 2.0 * 2.0;
            field.add(Box::new(Doublet::new(kappa, 0.0, 0.0)));
            self.field = field;
        }
        Ok(())
    }
}

impl InteractiveTool for PotentialFlowTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Potential Flow"
    }

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn apply_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        self.from_json_str(json_str)
    }

    fn export_preset_json(&self) -> Result<String, String> {
        self.to_json_string()
    }

    fn import_preset_json(&mut self, json_str: &str) -> Result<(), String> {
        self.from_json_str(json_str)
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&PotentialFlowState {
            grid_size: self.grid_size,
            view_range: self.view_range,
            selected_element_type: self.selected_element_type,
            new_element_strength: self.new_element_strength,
            new_element_x: self.new_element_x,
            new_element_y: self.new_element_y,
            new_element_angle: self.new_element_angle,
        })
        .ok()
    }

    fn load_state(&mut self, state: &str) {
        if let Ok(loaded) = serde_json::from_str::<PotentialFlowState>(state) {
            self.grid_size = loaded.grid_size;
            self.view_range = loaded.view_range;
            self.selected_element_type = loaded.selected_element_type;
            self.new_element_strength = loaded.new_element_strength;
            self.new_element_x = loaded.new_element_x;
            self.new_element_y = loaded.new_element_y;
            self.new_element_angle = loaded.new_element_angle;
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("potential_flow_controls").show(ctx, |ui| {
            ui.heading("Potential Flow Controls");
            if ui.button("🎨 Starter Gallery").clicked() {
                let cur = ctx.data(|d| d.get_temp::<bool>(egui::Id::new("SHOW_STARTER_GALLERY")).unwrap_or(false));
                ctx.data_mut(|d| d.insert_temp(egui::Id::new("SHOW_STARTER_GALLERY"), !cur));
            }
            ui.separator();

            ui.collapsing("Add Element", |ui| {
                ui.radio_value(
                    &mut self.selected_element_type,
                    ElementType::Uniform,
                    "Uniform Flow",
                );
                ui.radio_value(
                    &mut self.selected_element_type,
                    ElementType::Source,
                    "Source",
                );
                ui.radio_value(&mut self.selected_element_type, ElementType::Sink, "Sink");
                ui.radio_value(
                    &mut self.selected_element_type,
                    ElementType::Vortex,
                    "Vortex",
                );
                ui.radio_value(
                    &mut self.selected_element_type,
                    ElementType::Doublet,
                    "Doublet",
                );

                ui.separator();

                ui.label("Strength / Velocity");
                ui.add(egui::DragValue::new(&mut self.new_element_strength).speed(0.1));

                match self.selected_element_type {
                    ElementType::Uniform => {
                        ui.add(egui::Slider::new(
                            &mut self.new_element_angle,
                            -180.0..=180.0,
                        ).text("Angle (degrees)"));
                    }
                    _ => {
                        ui.label("Position X");
                        ui.add(egui::DragValue::new(&mut self.new_element_x).speed(0.1));
                        ui.label("Position Y");
                        ui.add(egui::DragValue::new(&mut self.new_element_y).speed(0.1));
                    }
                }

                if ui
                    .button("➕ Add Element")
                    .accessible_hover_text("Add the selected flow element to the field")
                    .clicked()
                {
                    self.add_element();
                }
            });

            ui.separator();

            if ui
                .button("🔄 Clear All")
                .accessible_hover_text("Remove all potential flow elements and reset the field")
                .clicked()
            {
                self.clear();
            }

            ui.separator();
            ui.add(egui::Slider::new(&mut self.grid_size, 10..=math_commons::registry::MAX_GRID_SIZE).text("Visualization - Grid Density"));
            ui.add(egui::Slider::new(&mut self.view_range, 5.0..=50.0).text("View Range"));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            Plot::new("potential_flow_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .show(ui, |plot_ui| {
                    // Generate vector field arrows
                    let step = (2.0 * self.view_range) / (self.grid_size as f64);
                    let min = -self.view_range;
                    let max = self.view_range;

                    let mut origins = Vec::new();
                    let mut tips = Vec::new();

                    let mut x = min;
                    while x <= max {
                        let mut y = min;
                        while y <= max {
                            let v = self.field.velocity(x, y);
                            let magnitude = v.norm();

                            // Scale arrow for visibility, but cap it to avoid huge arrows near singularities
                            let arrow_len = (magnitude * 0.1).min(step * 0.8);
                            if magnitude > 1e-6 {
                                let direction = v.normalize();
                                let start = [x, y];
                                let end =
                                    [x + direction.x * arrow_len, y + direction.y * arrow_len];
                                origins.push(start);
                                tips.push(end);
                            }

                            y += step;
                        }
                        x += step;
                    }

                    plot_ui.arrows(
                        Arrows::new("Vector Field", origins, tips).color(egui::Color32::LIGHT_BLUE),
                    );
                });
        });
    }
}

impl PotentialFlowTool {
    fn add_element(&mut self) {
        let element: Box<dyn FlowElement> = match self.selected_element_type {
            ElementType::Uniform => Box::new(UniformFlow::new(
                self.new_element_strength,
                self.new_element_angle,
            )),
            ElementType::Source => Box::new(Source::new(
                self.new_element_strength,
                self.new_element_x,
                self.new_element_y,
            )),
            ElementType::Sink => Box::new(Source::new(
                -self.new_element_strength,
                self.new_element_x,
                self.new_element_y,
            )),
            ElementType::Vortex => Box::new(Vortex::new(
                self.new_element_strength,
                self.new_element_x,
                self.new_element_y,
            )),
            ElementType::Doublet => Box::new(Doublet::new(
                self.new_element_strength,
                self.new_element_x,
                self.new_element_y,
            )),
        };
        self.field.add(element);
    }

    fn clear(&mut self) {
        self.field.clear();
    }
}

// [cite:modular_polynomials_review]


inventory::submit! {
    crate::framework::ToolMetadata {
        id: "PotentialFlowTool",
        display_name: "Potential Flow",
        domain: "fluid_dynamics",
        tags: &[],
        build: || Box::new(PotentialFlowTool::default()),
    }
}

impl scientific_metadata::theory::TheoryDescribable for PotentialFlowTool {
    fn theory_description(&self) -> String {
        "Two-dimensional incompressible irrotational potential flow governed by Laplace's equation $\\nabla^2 \\phi = 0$ for velocity potential $\\phi$, where fluid velocity is given by $\\vec{V} = \\nabla \\phi = (u, v)$ with linear superposition of elementary solutions.".into()
    }
    fn phonetic_description(&self) -> String { "Incompressible irrotational potential flow and Laplace equation theory".into() }
    fn theory_citation(&self) -> String { "Elementary Fluid Dynamics (Acheson, 1990); An Introduction to Fluid Dynamics (Batchelor, 1967)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert("Uniform".into(), "Uniform freestream flow phi = U (x cos(alpha) + y sin(alpha))".into());
        map.insert("Source/Sink".into(), "Radial flow phi = (m / 2 pi) ln(r)".into());
        map.insert("Vortex".into(), "Circulation flow phi = (Gamma / 2 pi) theta".into());
        map.insert("Doublet".into(), "Source-sink dipole limit phi = (mu / 2 pi) cos(theta) / r".into());
        map
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "vector_field",
        title: "Vector Field",
        description: "ψ = U·y·(1 - R²/r²): Potential streamlines and 2D velocity vector field past a cylinder.",
        category: "Fluid Mechanics",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_VECTOR_FIELD,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "source_sink",
        title: "Source-Sink Pair",
        description: "ψ = (q/2π)·(θ₁ - θ₂): Superposition of equal source and sink singularities.",
        category: "Fluid Mechanics",
        icon: "🔄",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || crate::starter_gallery::PRESET_SOURCE_SINK,
    }
}
