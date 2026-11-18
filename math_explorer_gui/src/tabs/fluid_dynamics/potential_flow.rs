use crate::accessibility::AccessibleHoverText;
use eframe::egui;
use egui_plot::{Arrows, Plot};
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex,
};
use std::f64::consts::PI;

pub struct PotentialFlowTool {
    field: PotentialFlowField,
    elements: Vec<SerializedFlowElement>,
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

#[derive(PartialEq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
enum ElementType {
    Uniform,
    #[default]
    Source,
    Sink,
    Vortex,
    Doublet,
}

#[derive(PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedFlowElement {
    element_type: ElementType,
    strength: f64,
    #[serde(default)]
    x: f64,
    #[serde(default)]
    y: f64,
    #[serde(default)]
    angle: f64,
}

const fn default_element_strength() -> f64 {
    10.0
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct PotentialFlowState {
    elements: Vec<SerializedFlowElement>,
    grid_size: usize,
    view_range: f64,
    #[serde(default)]
    selected_element_type: ElementType,
    #[serde(default = "default_element_strength")]
    new_element_strength: f64,
    #[serde(default)]
    new_element_x: f64,
    #[serde(default)]
    new_element_y: f64,
    #[serde(default)]
    new_element_angle: f64,
}

impl Default for PotentialFlowTool {
    fn default() -> Self {
        let kappa = 2.0 * PI * 10.0 * 2.0 * 2.0;
        let elements = vec![
            SerializedFlowElement {
                element_type: ElementType::Uniform,
                strength: 10.0,
                x: 0.0,
                y: 0.0,
                angle: 0.0,
            },
            SerializedFlowElement {
                element_type: ElementType::Doublet,
                strength: kappa,
                x: 0.0,
                y: 0.0,
                angle: 0.0,
            },
        ];
        let mut tool = Self {
            field: PotentialFlowField::new(),
            elements,
            grid_size: 20,
            view_range: 10.0,
            selected_element_type: ElementType::Source,
            new_element_strength: 10.0,
            new_element_x: 0.0,
            new_element_y: 0.0,
            new_element_angle: 0.0,
        };
        tool.rebuild_field();
        tool
    }
}

use crate::framework::InteractiveTool;

impl InteractiveTool for PotentialFlowTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable { self }
    fn name(&self) -> &'static str {
        "Potential Flow"
    }

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn save_state(&self) -> Option<String> {
        serde_json::to_string(&PotentialFlowState {
            elements: self.elements.clone(),
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

    fn try_load_state(&mut self, state: &str) -> Result<(), String> {
        let loaded: PotentialFlowState = serde_json::from_str(state)
            .map_err(|e| format!("Invalid PotentialFlowTool state: {}", e))?;
        if loaded.grid_size < 2
            || loaded.grid_size > math_commons::registry::MAX_GRID_SIZE
            || !loaded.view_range.is_finite()
            || loaded.view_range <= 0.0
            || !loaded.new_element_strength.is_finite()
            || !loaded.new_element_x.is_finite()
            || !loaded.new_element_y.is_finite()
            || !loaded.new_element_angle.is_finite()
            || loaded.elements.iter().any(|el| {
                !el.strength.is_finite()
                    || !el.x.is_finite()
                    || !el.y.is_finite()
                    || !el.angle.is_finite()
            })
        {
            return Err("PotentialFlowTool parameters must be finite and within valid grid/view bounds".to_string());
        }
        self.elements = loaded.elements;
        self.grid_size = loaded.grid_size;
        self.view_range = loaded.view_range;
        self.selected_element_type = loaded.selected_element_type;
        self.new_element_strength = loaded.new_element_strength;
        self.new_element_x = loaded.new_element_x;
        self.new_element_y = loaded.new_element_y;
        self.new_element_angle = loaded.new_element_angle;
        self.rebuild_field();
        Ok(())
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    fn show(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("potential_flow_controls").show(ctx, |ui| {
            ui.heading("Potential Flow Controls");
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
    fn rebuild_field(&mut self) {
        let mut field = PotentialFlowField::new();
        for el in &self.elements {
            let elem: Box<dyn FlowElement> = match el.element_type {
                ElementType::Uniform => Box::new(UniformFlow::new(el.strength, el.angle)),
                ElementType::Source => Box::new(Source::new(el.strength, el.x, el.y)),
                ElementType::Sink => Box::new(Source::new(-el.strength, el.x, el.y)),
                ElementType::Vortex => Box::new(Vortex::new(el.strength, el.x, el.y)),
                ElementType::Doublet => Box::new(Doublet::new(el.strength, el.x, el.y)),
            };
            field.add(elem);
        }
        self.field = field;
    }

    fn add_element(&mut self) {
        self.elements.push(SerializedFlowElement {
            element_type: self.selected_element_type,
            strength: self.new_element_strength,
            x: self.new_element_x,
            y: self.new_element_y,
            angle: self.new_element_angle,
        });
        self.rebuild_field();
    }

    fn clear(&mut self) {
        self.elements.clear();
        self.field.clear();
    }
}

// [cite:modular_polynomials_review]


inventory::submit! {
    crate::framework::ToolMetadata {
        name: "PotentialFlowTool",
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
