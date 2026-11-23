//! Deterministic seeding PRNG, dynamic [`TheoryDescribable`] metadata, and federated
//! [`crate::starter_gallery::StarterPresetMetadata`] registrations for morphogenesis.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::runner::MorphogenesisSharedParams;
use super::state::{MorphogenesisKinetics, MorphogenesisToolState};
use super::MorphogenesisTool;
use crate::async_sim::SimWorkerState;
use crate::framework::InteractiveTool;
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Re-exported deterministic PCG-style PRNG used to seed reproducible initial perturbations.
pub use crate::volumetric_scene::presets::SimpleRng;

/// Dynamic [`TheoryDescribable`] adapter reflecting the active [`MorphogenesisKinetics`] model.
#[derive(Clone, Debug)]
pub struct MorphogenesisTheory {
    /// Active kinetics mode when unbound, kept synchronized by [`MorphogenesisTool`].
    pub mode: MorphogenesisKinetics,
    params: Option<Arc<RwLock<MorphogenesisSharedParams>>>,
}

impl MorphogenesisTheory {
    /// Binds theory metadata to shared runtime parameters.
    #[must_use]
    pub fn new(params: Arc<RwLock<MorphogenesisSharedParams>>) -> Self {
        let mode = params.read().unwrap_or_else(|e| e.into_inner()).kinetics;
        Self {
            mode,
            params: Some(params),
        }
    }

    /// Returns a standalone theory descriptor pinned to `mode`.
    #[must_use]
    pub fn with_mode(mut self, mode: MorphogenesisKinetics) -> Self {
        self.mode = mode;
        self.params = None;
        self
    }

    fn snapshot(
        &self,
    ) -> (
        MorphogenesisKinetics,
        SchnakenbergKinetics,
        GrayScottKinetics,
    ) {
        if let Some(params) = &self.params {
            let p = params.read().unwrap_or_else(|e| e.into_inner());
            return (
                p.kinetics,
                SchnakenbergKinetics { a: p.a, b: p.b },
                GrayScottKinetics { f: p.f, k: p.k },
            );
        }
        (
            self.mode,
            SchnakenbergKinetics::new(0.1, 0.9),
            GrayScottKinetics::mitosis(),
        )
    }
}

impl Default for MorphogenesisTheory {
    fn default() -> Self {
        Self {
            mode: MorphogenesisKinetics::Schnakenberg,
            params: None,
        }
    }
}

impl TheoryDescribable for MorphogenesisTheory {
    fn theory_description(&self) -> String {
        let (mode, schnak, gs) = self.snapshot();
        match mode {
            MorphogenesisKinetics::Schnakenberg => schnak.theory_description(),
            MorphogenesisKinetics::GrayScott => gs.theory_description(),
        }
    }

    fn phonetic_description(&self) -> String {
        let (mode, schnak, gs) = self.snapshot();
        match mode {
            MorphogenesisKinetics::Schnakenberg => schnak.phonetic_description(),
            MorphogenesisKinetics::GrayScott => gs.phonetic_description(),
        }
    }

    fn theory_citation(&self) -> String {
        let (mode, schnak, gs) = self.snapshot();
        match mode {
            MorphogenesisKinetics::Schnakenberg => schnak.theory_citation(),
            MorphogenesisKinetics::GrayScott => gs.theory_citation(),
        }
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let (mode, schnak, gs) = self.snapshot();
        let mut descs = match mode {
            MorphogenesisKinetics::Schnakenberg => {
                let mut d = schnak.available_descriptions();
                d.insert(
                    "activator_diffusion".to_string(),
                    "Short-range spatial diffusion coefficient D_u of autocatalytic activator u."
                        .to_string(),
                );
                d.insert(
                    "inhibitor_diffusion".to_string(),
                    "Long-range spatial diffusion coefficient D_v driving Turing instability."
                        .to_string(),
                );
                d
            }
            MorphogenesisKinetics::GrayScott => {
                let mut d = gs.available_descriptions();
                d.insert(
                    "substrate_diffusion".to_string(),
                    "Fast substrate diffusion coefficient D_u replenishing autocatalytic fronts."
                        .to_string(),
                );
                d.insert(
                    "catalyst_diffusion".to_string(),
                    "Slow autocatalyst diffusion coefficient D_v localizing spots and waves."
                        .to_string(),
                );
                d
            }
        };
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
        let constraint =
            |min, max, step| scientific_metadata::theory::ParameterConstraint { min, max, step };
        let (mode, schnak, gs) = self.snapshot();
        match mode {
            MorphogenesisKinetics::Schnakenberg => {
                let mut map = schnak.theory_parameters();
                map.entry("a".to_string())
                    .or_insert_with(|| constraint(0.0, 1.0, 0.01));
                map.entry("b".to_string())
                    .or_insert_with(|| constraint(0.0, 2.0, 0.01));
                map.insert("d_u".to_string(), constraint(0.1, 5.0, 0.05));
                map.insert("d_v".to_string(), constraint(10.0, 200.0, 1.0));
                map.insert("dt".to_string(), constraint(1e-4, 0.1, 5e-4));
                map
            }
            MorphogenesisKinetics::GrayScott => {
                let mut map = gs.theory_parameters();
                map.entry("f".to_string())
                    .or_insert_with(|| constraint(0.005, 0.10, 0.001));
                map.entry("k".to_string())
                    .or_insert_with(|| constraint(0.01, 0.09, 0.001));
                map.insert("d_u".to_string(), constraint(0.02, 0.50, 0.005));
                map.insert("d_v".to_string(), constraint(0.01, 0.25, 0.005));
                map.insert("dt".to_string(), constraint(0.05, 1.60, 0.01));
                map
            }
        }
    }

    fn theory_latex_equation(&self) -> Option<String> {
        let (mode, _, _) = self.snapshot();
        match mode {
            MorphogenesisKinetics::Schnakenberg => Some(
                r"\frac{\partial u}{\partial t} = D_u \nabla^2 u + a - u + u^2 v,\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + b - u^2 v"
                    .to_string(),
            ),
            MorphogenesisKinetics::GrayScott => Some(
                r"\frac{\partial u}{\partial t} = D_u \nabla^2 u - u v^2 + F(1 - u),\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + u v^2 - (F + k) v"
                    .to_string(),
            ),
        }
    }
}

/// JSON preset envelope for Gray-Scott self-replicating mitosis spots (`F = 0.0367, k = 0.0649`).
pub const PRESET_GRAY_SCOTT_MITOSIS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Mitosis",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "kinetics": "GrayScott",
    "a": 0.1,
    "b": 0.9,
    "f": 0.0367,
    "k": 0.0649,
    "d_u": 0.2,
    "d_v": 0.1,
    "dt": 1.0,
    "auto_dt": true,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Gray-Scott coral dendritic growth (`F = 0.0545, k = 0.062`).
pub const PRESET_GRAY_SCOTT_CORAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Coral",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "kinetics": "GrayScott",
    "a": 0.1,
    "b": 0.9,
    "f": 0.0545,
    "k": 0.062,
    "d_u": 0.2,
    "d_v": 0.1,
    "dt": 1.0,
    "auto_dt": true,
    "width": 100,
    "height": 100
  }
}"#;

/// JSON preset envelope for Gray-Scott localized solitons (`F = 0.030, k = 0.060`).
pub const PRESET_GRAY_SCOTT_SOLITONS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Solitons",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "kinetics": "GrayScott",
    "a": 0.1,
    "b": 0.9,
    "f": 0.03,
    "k": 0.06,
    "d_u": 0.2,
    "d_v": 0.1,
    "dt": 1.0,
    "auto_dt": true,
    "width": 100,
    "height": 100
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
        id: "gray_scott_mitosis",
        title: "Gray-Scott Mitosis",
        description: "∂v/∂t = D_v∇²v + uv² - (F+k)v: Self-replicating autocatalytic spots undergoing symmetric cell division.",
        category: "Reaction-Diffusion",
        icon: "🦠",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_MITOSIS,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_coral",
        title: "Gray-Scott Coral",
        description: "∂u/∂t = D_u∇²u - uv² + F(1-u): Substrate-limited tip branching generating dendritic coral fronts.",
        category: "Morphogenesis",
        icon: "🪸",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_CORAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "gray_scott_solitons",
        title: "Gray-Scott Solitons",
        description: "F = 0.030, k = 0.060: Excitable subcritical pulses forming persistent localized chemical solitons.",
        category: "Reaction-Diffusion",
        icon: "💠",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_SOLITONS,
    }
}

impl MorphogenesisTool {
    pub(crate) fn apply_named_preset(&mut self, a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) {
        let f = self.params.read().unwrap_or_else(|e| e.into_inner()).f;
        let k = self.params.read().unwrap_or_else(|e| e.into_inner()).k;
        self.load_preset_state(
            MorphogenesisKinetics::Schnakenberg,
            a,
            b,
            f,
            k,
            d_u,
            d_v,
            dt,
        );
    }

    pub(crate) fn apply_gray_scott_preset(&mut self, f: f64, k: f64, d_u: f64, d_v: f64, dt: f64) {
        let a = self.params.read().unwrap_or_else(|e| e.into_inner()).a;
        let b = self.params.read().unwrap_or_else(|e| e.into_inner()).b;
        self.load_preset_state(MorphogenesisKinetics::GrayScott, a, b, f, k, d_u, d_v, dt);
    }

    #[allow(clippy::too_many_arguments)]
    fn load_preset_state(
        &mut self,
        kinetics: MorphogenesisKinetics,
        a: f64,
        b: f64,
        f: f64,
        k: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) {
        let target_state = if self.worker.is_running() {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let pending_initial_start =
            target_state != SimWorkerState::Running && self.pending_initial_start;
        let mask_rle = self
            .mask_rle
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (xk, yk) = self.atlas.axis_keys();
        let allowed_pairs = match kinetics {
            MorphogenesisKinetics::Schnakenberg => super::cartographer::MORPHO_ATLAS_AXIS_PAIRS,
            MorphogenesisKinetics::GrayScott => {
                super::gray_scott_atlas::GRAY_SCOTT_ATLAS_AXIS_PAIRS
            }
        };
        let atlas_axes = if self.atlas.has_default_persisted_state()
            || !allowed_pairs.iter().any(|&(ax, ay)| ax == xk && ay == yk)
        {
            super::cartographer::default_morpho_atlas_axes()
        } else {
            [xk.to_string(), yk.to_string()]
        };
        let preset_state = MorphogenesisToolState {
            kinetics_mode: kinetics,
            a,
            b,
            f,
            k,
            d_u,
            d_v,
            dt,
            auto_dt: true,
            width: self.width,
            height: self.height,
            state: Some(target_state),
            paused: None,
            pending_initial_start,
            mask_rle,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes,
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }
}
