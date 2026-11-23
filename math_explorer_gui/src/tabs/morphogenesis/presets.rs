#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::biology::morphogenesis::SchnakenbergKinetics;
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Deterministic PCG-style PRNG used to seed reproducible initial perturbations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    /// Creates a new deterministic PRNG seeded with `seed`.
    pub const fn new(seed: u64) -> Self {
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

/// GUI [`TheoryDescribable`] wrapper around [`SchnakenbergKinetics`] (`domain_biology`).
#[derive(Clone, Debug)]
pub struct MorphogenesisTheory {
    kinetics: SchnakenbergKinetics,
}

impl Default for MorphogenesisTheory {
    fn default() -> Self {
        Self {
            kinetics: SchnakenbergKinetics { a: 0.1, b: 0.9 },
        }
    }
}

impl TheoryDescribable for MorphogenesisTheory {
    fn theory_description(&self) -> String {
        self.kinetics.theory_description()
    }

    fn phonetic_description(&self) -> String {
        self.kinetics.phonetic_description()
    }

    fn theory_citation(&self) -> String {
        self.kinetics.theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = self.kinetics.available_descriptions();
        descs.insert(
            "activator_diffusion".to_string(),
            "Short-range spatial diffusion coefficient D_u of autocatalytic activator u."
                .to_string(),
        );
        descs.insert(
            "inhibitor_diffusion".to_string(),
            "Long-range spatial diffusion coefficient D_v driving Turing instability.".to_string(),
        );
        descs.insert(
            "dt".to_string(),
            "Explicit Forward Euler integration time step bounded by the 2D diffusion CFL limit."
                .to_string(),
        );
        descs
    }

    fn theory_parameters(
        &self,
    ) -> HashMap<String, scientific_metadata::theory::ParameterConstraint> {
        use scientific_metadata::theory::ParameterConstraint;
        let mut map = HashMap::with_capacity(5);
        map.insert(
            "a".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
        );
        map.insert(
            "b".to_string(),
            ParameterConstraint {
                min: 0.0,
                max: 2.0,
                step: 0.01,
            },
        );
        map.insert(
            "d_u".to_string(),
            ParameterConstraint {
                min: 0.1,
                max: 5.0,
                step: 0.05,
            },
        );
        map.insert(
            "d_v".to_string(),
            ParameterConstraint {
                min: 10.0,
                max: 200.0,
                step: 1.0,
            },
        );
        map.insert(
            "dt".to_string(),
            ParameterConstraint {
                min: 1e-4,
                max: 0.1,
                step: 5e-4,
            },
        );
        map
    }

    fn theory_latex_equation(&self) -> Option<String> {
        self.kinetics.theory_latex_equation().or_else(|| {
            Some(
                r"\frac{\partial u}{\partial t} = D_u \nabla^2 u + a - u + u^2 v,\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + b - u^2 v"
                    .to_string(),
            )
        })
    }
}

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
        id: "morphogenesis_inhibitor_fork",
        title: "Morphogenesis Inhibitor Fork",
        description: "Counterfactual Schnakenberg twin forked at step 60 lowering D_v from 40 to 18 with |u_B - u_A| difference heatmap.",
        category: "Counterfactual Biology",
        icon: "⑂",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || crate::starter_gallery::PRESET_MORPHOGENESIS_INHIBITOR_FORK,
    }
}
