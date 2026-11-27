//! Aerodynamic potential-flow presets, theory descriptions, and Starter Template Gallery registrations.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::{ElementType, SerializedFlowElement};
use crate::starter_gallery::{
    StarterPresetMetadata, PRESET_LIFTING_CYLINDER, PRESET_RANKINE_OVAL, PRESET_SOURCE_SINK,
    PRESET_VECTOR_FIELD, PRESET_VORTEX_STREET,
};
use std::collections::HashMap;
use std::f64::consts::PI;

pub(crate) fn theory_element_descriptions() -> HashMap<String, String> {
    let mut map = HashMap::new();
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

pub(crate) fn symmetric_cylinder_elements() -> Vec<SerializedFlowElement> {
    let u = 10.0;
    let r = 2.0;
    vec![
        SerializedFlowElement {
            element_type: ElementType::Uniform,
            strength: u,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        },
        SerializedFlowElement {
            element_type: ElementType::Doublet,
            strength: 2.0 * PI * u * r * r,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        },
    ]
}

pub(crate) fn lifting_cylinder_elements() -> Vec<SerializedFlowElement> {
    let u = 10.0;
    let r = 2.0;
    let mut els = symmetric_cylinder_elements();
    els.push(SerializedFlowElement {
        element_type: ElementType::Vortex,
        strength: -2.0 * PI * u * r,
        x: 0.0,
        y: 0.0,
        angle: 0.0,
    });
    els
}

pub(crate) fn rankine_oval_elements() -> Vec<SerializedFlowElement> {
    vec![
        SerializedFlowElement {
            element_type: ElementType::Uniform,
            strength: 10.0,
            x: 0.0,
            y: 0.0,
            angle: 0.0,
        },
        SerializedFlowElement {
            element_type: ElementType::Source,
            strength: 75.0,
            x: -2.5,
            y: 0.0,
            angle: 0.0,
        },
        SerializedFlowElement {
            element_type: ElementType::Sink,
            strength: 75.0,
            x: 2.5,
            y: 0.0,
            angle: 0.0,
        },
    ]
}

pub(crate) fn vortex_street_elements() -> Vec<SerializedFlowElement> {
    let mut els = vec![SerializedFlowElement {
        element_type: ElementType::Uniform,
        strength: 8.0,
        x: 0.0,
        y: 0.0,
        angle: 0.0,
    }];
    for k in 0..3 {
        let x_top = -6.0 + 4.0 * k as f64;
        els.push(SerializedFlowElement {
            element_type: ElementType::Vortex,
            strength: -45.0,
            x: x_top,
            y: 1.4,
            angle: 0.0,
        });
        els.push(SerializedFlowElement {
            element_type: ElementType::Vortex,
            strength: 45.0,
            x: x_top + 2.0,
            y: -1.4,
            angle: 0.0,
        });
    }
    els
}

inventory::submit! {
    StarterPresetMetadata {
        id: "vector_field",
        title: "Vector Field",
        description: "ψ = U·y·(1 - R²/r²): Potential streamlines and 2D velocity vector field past a cylinder.",
        category: "Fluid Mechanics",
        icon: "🌊",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || PRESET_VECTOR_FIELD,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "source_sink",
        title: "Source-Sink Pair",
        description: "ψ = (q/2π)·(θ₁ - θ₂): Superposition of equal source and sink singularities.",
        category: "Fluid Mechanics",
        icon: "🔄",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || PRESET_SOURCE_SINK,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "potential_flow_lifting_cylinder",
        title: "Kutta-Joukowski Lifting Cylinder",
        description: "Spinning cylinder circulation shifting stagnation points downward via Magnus lift.",
        category: "Fluid Mechanics",
        icon: "🛫",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || PRESET_LIFTING_CYLINDER,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "potential_flow_rankine_oval",
        title: "Rankine Oval Body",
        description: "Closed dividing streamline formed by a source-sink pair in uniform freestream.",
        category: "Fluid Mechanics",
        icon: "🥚",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || PRESET_RANKINE_OVAL,
    }
}

inventory::submit! {
    StarterPresetMetadata {
        id: "potential_flow_vortex_street",
        title: "Staggered Vortex Street",
        description: "Counter-rotating staggered vortex wake with meandering RK4 dye streaks.",
        category: "Fluid Mechanics",
        icon: "🌪",
        domain: "fluid_dynamics",
        tool_name: Some("PotentialFlowTool"),
        preset_generator: || PRESET_VORTEX_STREET,
    }
}

impl scientific_metadata::theory::TheoryDescribable for super::PotentialFlowTool {
    fn theory_description(&self) -> String {
        "Two-dimensional incompressible irrotational potential flow governed by Laplace's equation $\\nabla^2 \\phi = 0$ for velocity potential $\\phi$, where fluid velocity is given by $\\vec{V} = \\nabla \\phi = (u, v)$ with linear superposition of elementary solutions.".into()
    }

    fn phonetic_description(&self) -> String {
        "Incompressible irrotational potential flow and Laplace equation theory".into()
    }

    fn theory_citation(&self) -> String {
        "Elementary Fluid Dynamics (Acheson, 1990); An Introduction to Fluid Dynamics (Batchelor, 1967)".into()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        theory_element_descriptions()
    }
}
