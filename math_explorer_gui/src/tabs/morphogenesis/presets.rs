#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::types::MorphogenesisKineticsMode;
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Deterministic PCG-style PRNG used to seed reproducible initial perturbations.
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Creates a new deterministic PRNG seeded with `seed`.
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Samples a uniform `f64` in `[0.0, 1.0]`.
    pub fn next_f64(&mut self) -> f64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x = self.state;
        let x = x ^ x >> 18;
        let rot = (x >> 27) as u32;
        let val = (x as u32).rotate_right(rot);
        (val as f64) / (u32::MAX as f64)
    }

    /// Samples a uniform `f64` in `[min, max]`.
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        min + self.next_f64() * (max - min)
    }
}

/// GUI [`TheoryDescribable`] wrapper switching dynamically between [`SchnakenbergKinetics`]
/// and [`GrayScottKinetics`].
#[derive(Clone, Debug)]
pub struct MorphogenesisTheory {
    mode: MorphogenesisKineticsMode,
    schnakenberg: SchnakenbergKinetics,
    gray_scott: GrayScottKinetics,
}

impl Default for MorphogenesisTheory {
    fn default() -> Self {
        Self {
            mode: MorphogenesisKineticsMode::Schnakenberg,
            schnakenberg: SchnakenbergKinetics { a: 0.1, b: 0.9 },
            gray_scott: GrayScottKinetics::default(),
        }
    }
}

impl MorphogenesisTheory {
    /// Synchronizes active kinetics variant for Theory Portal inspection.
    pub fn set_mode(&mut self, mode: MorphogenesisKineticsMode) {
        self.mode = mode;
    }
}

impl TheoryDescribable for MorphogenesisTheory {
    fn theory_description(&self) -> String {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.schnakenberg.theory_description(),
            MorphogenesisKineticsMode::GrayScott => self.gray_scott.theory_description(),
        }
    }

    fn phonetic_description(&self) -> String {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.schnakenberg.phonetic_description(),
            MorphogenesisKineticsMode::GrayScott => self.gray_scott.phonetic_description(),
        }
    }

    fn theory_citation(&self) -> String {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.schnakenberg.theory_citation(),
            MorphogenesisKineticsMode::GrayScott => self.gray_scott.theory_citation(),
        }
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => self.schnakenberg.available_descriptions(),
            MorphogenesisKineticsMode::GrayScott => self.gray_scott.available_descriptions(),
        };
        descs.insert(
            "peclet_number".to_string(),
            "Dimensionless Péclet number Pe = U L / D_u comparing advective transport to activator diffusion."
                .to_string(),
        );
        descs.insert(
            "damkohler_number".to_string(),
            "Dimensionless Damköhler number Da comparing hydrodynamic residence time to chemical reaction timescale."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => Some(
                r"\frac{\partial u}{\partial t} + \nabla\cdot(\mathbf{v}u) = D_u \nabla^2 u + a - u + u^2 v,\quad \frac{\partial v}{\partial t} + \nabla\cdot(\mathbf{v}v) = D_v \nabla^2 v + b - u^2 v"
                    .to_string(),
            ),
            MorphogenesisKineticsMode::GrayScott => Some(
                r"\frac{\partial u}{\partial t} + \nabla\cdot(\mathbf{v}u) = D_u \nabla^2 u - u v^2 + f(1 - u),\quad \frac{\partial v}{\partial t} + \nabla\cdot(\mathbf{v}v) = D_v \nabla^2 v + u v^2 - (f + k)v"
                    .to_string(),
            ),
        }
    }
}

/// Advected Schnakenberg starter preset coupled to a D2Q9 Kármán vortex street.
pub const PRESET_KARMAN_TURING_WAKE: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Karman Vortex Turing Wake",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.12,
    "b": 0.88,
    "d_u": 0.4,
    "d_v": 16.0,
    "dt": 0.02,
    "width": 96,
    "height": 64,
    "kinetics": "Schnakenberg",
    "flow_mode": "LbmVortexWake",
    "inlet_velocity": 0.09,
    "viscosity": 0.018,
    "circulation": 0.0,
    "overlay_mode": "DualOverlay",
    "brush_mode": "Obstacle",
    "brush_radius": 4
  }
}"#;

/// Advected Gray-Scott starter preset sheared by an analytical circulation potential flow.
pub const PRESET_GRAY_SCOTT_SPIRAL_FLOW: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gray-Scott Potential Swirl",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.037,
    "b": 0.06,
    "d_u": 0.2,
    "d_v": 0.1,
    "dt": 0.5,
    "width": 96,
    "height": 64,
    "kinetics": "GrayScott",
    "flow_mode": "PotentialFlow",
    "inlet_velocity": 0.08,
    "viscosity": 0.02,
    "circulation": 6.5,
    "overlay_mode": "DualOverlay",
    "brush_mode": "Reagent",
    "brush_radius": 4
  }
}"#;

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_spots",
        title: "Turing Spots",
        description: "∂u/∂t = D_u∇²u + a - u + u²v: Activator-inhibitor reaction-diffusion generating spot morphology.",
        category: "Reaction-Diffusion",
        icon: "🐆",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || crate::starter_gallery::PRESET_TURING_SPOTS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_stripes",
        title: "Turing Stripes",
        description: "∂v/∂t = D_v∇²v + b - u²v: Differential diffusion instability yielding labyrinthine stripe patterns.",
        category: "Morphogenesis",
        icon: "🦓",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || crate::starter_gallery::PRESET_TURING_STRIPES,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "karman_turing_wake",
        title: "Karman Vortex Turing Wake",
        description: "∂u/∂t + ∇·(vu) = D_u∇²u + R(u,v): D2Q9 Lattice Boltzmann vortex street shearing Turing spots.",
        category: "Advected Morphogenesis",
        icon: "🌀",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_KARMAN_TURING_WAKE,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_spiral_flow",
        title: "Gray-Scott Potential Swirl",
        description: "∂v/∂t + ∇·(vv) = D_v∇²v + uv² - (f+k)v: Autocatalytic filaments wound by potential-flow circulation Γ.",
        category: "Advected Morphogenesis",
        icon: "🌊",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_SPIRAL_FLOW,
    }
}
