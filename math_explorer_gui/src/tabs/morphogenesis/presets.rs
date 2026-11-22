#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use math_explorer::biology::morphogenesis::SchnakenbergKinetics;
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
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
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
        descs
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

fn generate_hybrid_turing_mask_preset() -> &'static str {
    static HYBRID_PRESET: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    HYBRID_PRESET
        .get_or_init(|| {
            let mut mask = vec![0u8; 100 * 100];
            for y in 15..85 {
                for x in 50..90 {
                    mask[y * 100 + x] = 1;
                }
            }
            let rle = crate::framework::encode_u8_grid(100, 100, &mask, 2)
                .expect("valid morphogenesis hybrid RLE");
            serde_json::json!({
                "metadata": {
                    "schema_version": "1.0",
                    "title": "Turing Spot-Stripe Mask",
                    "domain": "biology",
                    "timestamp": "2026-11-21T00:00:00Z"
                },
                "config": {
                    "a": 0.08,
                    "b": 1.12,
                    "d_u": 1.0,
                    "d_v": 40.0,
                    "dt": 0.005,
                    "width": 100,
                    "height": 100,
                    "mask_rle": rle
                }
            })
            .to_string()
        })
        .as_str()
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "turing_spot_stripe_mask",
        title: "Turing Spot-Stripe Mask",
        description: "Spatially heterogeneous Schnakenberg (a, b) RLE reaction mask transitioning spots into labyrinthine stripes.",
        category: "Morphogenesis",
        icon: "🎭",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: generate_hybrid_turing_mask_preset,
    }
}
