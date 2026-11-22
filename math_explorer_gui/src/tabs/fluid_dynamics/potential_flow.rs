#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::accessibility::AccessibleHoverText;
use crate::framework::{bounded_drag_value, InteractiveTool};
use eframe::egui;
use egui_plot::{Arrows, Plot};
use math_explorer::physics::fluid_dynamics::potential_flow::{
    Doublet, FlowElement, PotentialFlowField, Source, UniformFlow, Vortex,
};
use std::f64::consts::PI;

pub(crate) const FLOW_STRENGTH_MIN: f64 = -1_000.0;
pub(crate) const FLOW_STRENGTH_MAX: f64 = 1_000.0;
pub(crate) const FLOW_POS_MIN: f64 = -50.0;
pub(crate) const FLOW_POS_MAX: f64 = 50.0;
const FLOW_ANGLE_MIN: f64 = -180.0;
const FLOW_ANGLE_MAX: f64 = 180.0;
const FLOW_VIEW_MIN: f64 = 0.1;
const FLOW_VIEW_MAX: f64 = 50.0;

pub struct PotentialFlowTool {
    pub(crate) field: PotentialFlowField,
    pub(crate) elements: Vec<SerializedFlowElement>,
    pub(crate) grid_size: usize,
    pub(crate) view_range: f64,
    pub(crate) selected_element_type: ElementType,
    pub(crate) new_element_strength: f64,
    pub(crate) new_element_x: f64,
    pub(crate) new_element_y: f64,
    pub(crate) new_element_angle: f64,
}

#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) enum ElementType {
    Uniform,
    #[default]
    Source,
    Sink,
    Vortex,
    Doublet,
}

#[derive(PartialEq, Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SerializedFlowElement {
    pub(crate) element_type: ElementType,
    pub(crate) strength: f64,
    #[serde(default)]
    pub(crate) x: f64,
    #[serde(default)]
    pub(crate) y: f64,
    #[serde(default)]
    pub(crate) angle: f64,
}

impl SerializedFlowElement {
    fn is_valid(&self) -> bool {
        self.strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.strength)
            && self.x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.x)
            && self.y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.y)
            && self.angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.angle)
    }
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

impl PotentialFlowState {
    fn is_valid(&self) -> bool {
        (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
    }
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

impl InteractiveTool for PotentialFlowTool {
    fn theory(&self) -> &dyn scientific_metadata::theory::TheoryDescribable {
        self
    }

    fn name(&self) -> &'static str {
        "Potential Flow"
    }

    fn domain(&self) -> &'static str {
        "fluid_dynamics"
    }

    fn has_finite_parameters(&self) -> bool {
        (2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
            && self.view_range.is_finite()
            && (FLOW_VIEW_MIN..=FLOW_VIEW_MAX).contains(&self.view_range)
            && self.new_element_strength.is_finite()
            && (FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX).contains(&self.new_element_strength)
            && self.new_element_x.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_x)
            && self.new_element_y.is_finite()
            && (FLOW_POS_MIN..=FLOW_POS_MAX).contains(&self.new_element_y)
            && self.new_element_angle.is_finite()
            && (FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX).contains(&self.new_element_angle)
            && self.elements.len() <= crate::framework::MAX_JSON_ARRAY_LEN
            && self.elements.iter().all(SerializedFlowElement::is_valid)
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
        if !loaded.is_valid() {
            return Err(
                "PotentialFlowTool parameters must be finite and within valid grid/view bounds"
                    .to_string(),
            );
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

    #[cfg(test)]
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }

    #[cfg(test)]
    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }

    fn show_inside(&mut self, ui: &mut egui::Ui) {
        self.draw_controls(ui);
        let (origins, tips) = self.compute_arrow_field();
        egui::CentralPanel::default().show_inside(ui, |ui| {
            Plot::new("potential_flow_plot")
                .data_aspect(1.0)
                .view_aspect(1.0)
                .show(ui, |plot_ui| {
                    let arrow_color =
                        crate::app::SemanticPalette::from_visuals(&plot_ui.ctx().style().visuals)
                            .series_color(0);
                    plot_ui.arrows(Arrows::new("Vector Field", origins, tips).color(arrow_color));
                });
        });
    }
}

impl PotentialFlowTool {
    fn draw_add_element_controls(&mut self, ui: &mut egui::Ui) {
        ui.collapsing("Add Element", |ui| {
            for (elem, label) in [
                (ElementType::Uniform, "Uniform Flow"),
                (ElementType::Source, "Source"),
                (ElementType::Sink, "Sink"),
                (ElementType::Vortex, "Vortex"),
                (ElementType::Doublet, "Doublet"),
            ] {
                ui.radio_value(&mut self.selected_element_type, elem, label);
            }

            ui.separator();
            ui.label("Strength / Velocity");
            ui.add(bounded_drag_value(
                &mut self.new_element_strength,
                FLOW_STRENGTH_MIN..=FLOW_STRENGTH_MAX,
                0.1,
            ));

            if self.selected_element_type == ElementType::Uniform {
                ui.add(
                    egui::Slider::new(&mut self.new_element_angle, FLOW_ANGLE_MIN..=FLOW_ANGLE_MAX)
                        .text("Angle (degrees)"),
                );
            } else {
                ui.label("Position X");
                ui.add(bounded_drag_value(
                    &mut self.new_element_x,
                    FLOW_POS_MIN..=FLOW_POS_MAX,
                    0.1,
                ));
                ui.label("Position Y");
                ui.add(bounded_drag_value(
                    &mut self.new_element_y,
                    FLOW_POS_MIN..=FLOW_POS_MAX,
                    0.1,
                ));
            }

            if ui
                .button("➕ Add Element")
                .accessible_hover_text("Add the selected flow element to the field")
                .clicked()
            {
                self.add_element();
            }
        });
    }

    fn draw_controls(&mut self, parent_ui: &mut egui::Ui) {
        let panel_id = crate::framework::scoped_panel_id(parent_ui, "potential_flow_controls");
        crate::framework::PaneSidePanel::left(panel_id).show_inside(parent_ui, |ui| {
            ui.heading("Potential Flow Controls");
            ui.separator();
            self.draw_add_element_controls(ui);

            ui.separator();
            if ui
                .button("🔄 Clear All")
                .accessible_hover_text("Remove all potential flow elements and reset the field")
                .clicked()
            {
                self.clear();
            }

            ui.separator();
            ui.add(
                egui::Slider::new(
                    &mut self.grid_size,
                    10..=math_commons::registry::MAX_GRID_SIZE,
                )
                .text("Visualization - Grid Density"),
            );
            ui.add(egui::Slider::new(&mut self.view_range, 5.0..=FLOW_VIEW_MAX).text("View Range"));
        });
    }

    pub(crate) fn compute_arrow_field(&self) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        if !self.view_range.is_finite()
            || self.view_range <= 0.0
            || !(2..=math_commons::registry::MAX_GRID_SIZE).contains(&self.grid_size)
        {
            return (Vec::new(), Vec::new());
        }
        let step = (2.0 * self.view_range) / (self.grid_size as f64);
        if !step.is_finite() || step <= 0.0 {
            return (Vec::new(), Vec::new());
        }
        let min = -self.view_range;
        let cap = (self.grid_size + 1) * (self.grid_size + 1);
        let mut origins = Vec::with_capacity(cap);
        let mut tips = Vec::with_capacity(cap);

        for ix in 0..=self.grid_size {
            let x = min + (ix as f64) * step;
            for iy in 0..=self.grid_size {
                let y = min + (iy as f64) * step;
                let v = self.field.velocity(x, y);
                if !v.x.is_finite() || !v.y.is_finite() {
                    continue;
                }
                let magnitude = v.norm();
                if !magnitude.is_finite() || magnitude <= 1e-6 {
                    continue;
                }
                let arrow_len = (magnitude * 0.1).min(step * 0.8);
                if !arrow_len.is_finite() || arrow_len <= 0.0 {
                    continue;
                }
                let direction = v / magnitude;
                let end = [x + direction.x * arrow_len, y + direction.y * arrow_len];
                if end[0].is_finite() && end[1].is_finite() {
                    origins.push([x, y]);
                    tips.push(end);
                }
            }
        }
        (origins, tips)
    }

    pub(crate) fn rebuild_field(&mut self) {
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

    pub(crate) fn add_element(&mut self) {
        self.elements.push(SerializedFlowElement {
            element_type: self.selected_element_type,
            strength: self
                .new_element_strength
                .clamp(FLOW_STRENGTH_MIN, FLOW_STRENGTH_MAX),
            x: self.new_element_x.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            y: self.new_element_y.clamp(FLOW_POS_MIN, FLOW_POS_MAX),
            angle: self.new_element_angle.clamp(FLOW_ANGLE_MIN, FLOW_ANGLE_MAX),
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

    fn phonetic_description(&self) -> String {
        "Incompressible irrotational potential flow and Laplace equation theory".into()
    }

    fn theory_citation(&self) -> String {
        "Elementary Fluid Dynamics (Acheson, 1990); An Introduction to Fluid Dynamics (Batchelor, 1967)".into()
    }

    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Uniform".into(),
            "Uniform freestream flow phi = U (x cos(alpha) + y sin(alpha))".into(),
        );
        map.insert(
            "Source/Sink".into(),
            "Radial flow phi = (m / 2 pi) ln(r)".into(),
        );
        map.insert(
            "Vortex".into(),
            "Circulation flow phi = (Gamma / 2 pi) theta".into(),
        );
        map.insert(
            "Doublet".into(),
            "Source-sink dipole limit phi = (mu / 2 pi) cos(theta) / r".into(),
        );
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
