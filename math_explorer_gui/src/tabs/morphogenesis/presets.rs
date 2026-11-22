#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::tool::MorphogenesisTool;
use crate::async_sim::SimWorkerState;
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::presets::MorphogenesisKineticsMode;
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

impl FiniteStateGuard for MorphogenesisTool {
    fn is_finite_state(&self) -> bool {
        self.worker.state() != SimWorkerState::Faulted && self.worker.diagnostic_report().is_none()
    }

    fn diagnostic_report(&self) -> Option<&DiagnosticReport> {
        self.worker.diagnostic_report()
    }

    fn diverged(&self) -> bool {
        !FiniteStateGuard::is_finite_state(self)
    }

    fn auto_scale_step_size(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.dt = (p.dt * 0.5).max(1e-4);
        drop(p);
        self.reset_system();
    }

    fn clamp_parameters(&mut self) {
        let mode = {
            let mut ks = self.kinetics_state.write().unwrap_or_else(|e| e.into_inner());
            ks.f = ks.f.clamp(0.0, 0.12);
            ks.k = ks.k.clamp(0.0, 0.12);
            ks.mode
        };
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        match mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                p.d_u = p.d_u.clamp(0.1, 5.0);
                p.d_v = p.d_v.clamp(10.0, 200.0);
                p.dt = p.dt.clamp(0.001, 0.1);
            }
            MorphogenesisKineticsMode::GrayScott => {
                p.d_u = p.d_u.clamp(0.01, 1.0);
                p.d_v = p.d_v.clamp(0.005, 1.0);
                p.dt = p.dt.clamp(0.001, 2.0);
            }
        }
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let gs = GrayScottKinetics::mitosis();
        {
            let mut ks = self.kinetics_state.write().unwrap_or_else(|e| e.into_inner());
            ks.mode = MorphogenesisKineticsMode::Schnakenberg;
            ks.f = gs.f;
            ks.k = gs.k;
        }
        {
            let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
            p.a = 0.1;
            p.b = 0.9;
            p.d_u = 1.0;
            p.d_v = 100.0;
            p.dt = 0.05;
        }
        self.theory.mode = MorphogenesisKineticsMode::Schnakenberg;
        self.reset_system();
    }
}

/// JSON preset envelope for Gray-Scott Mitosis reaction-diffusion.
pub const PRESET_GRAY_SCOTT_MITOSIS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Turing Morphogenesis - Gray-Scott Mitosis",
    "domain": "biology",
    "timestamp": "2026-11-14T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.16,
    "d_v": 0.08,
    "dt": 1.0,
    "width": 100,
    "height": 100,
    "kinetics_mode": "GrayScott",
    "f": 0.0367,
    "k": 0.0649
  }
}"#;

fn default_grid_dim() -> usize {
    100
}

fn default_morpho_a() -> f64 {
    0.1
}

fn default_morpho_b() -> f64 {
    0.9
}

fn default_gray_scott_f() -> f64 {
    GrayScottKinetics::mitosis().f
}

fn default_gray_scott_k() -> f64 {
    GrayScottKinetics::mitosis().k
}

/// Complete validated state schema for [`super::MorphogenesisTool`], compatible with `PRESET_TURING_*`
/// and `GrayScott` (`Mitosis`, `Coral`) modes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    #[serde(default = "default_morpho_a")]
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    #[serde(default = "default_morpho_b")]
    pub b: f64,
    /// Species `u` diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Species `v` diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Integration time step `dt`.
    pub dt: f64,
    /// Spatial grid width (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub width: usize,
    /// Spatial grid height (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub height: usize,
    /// Reaction kinetics family (`Schnakenberg` or `GrayScott`).
    #[serde(default)]
    pub kinetics_mode: MorphogenesisKineticsMode,
    /// Gray-Scott substrate feed rate `f` (`0.0..=0.12`).
    #[serde(default = "default_gray_scott_f")]
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k` (`0.0..=0.12`).
    #[serde(default = "default_gray_scott_k")]
    pub k: f64,
    /// Optional worker execution state (`Running` when omitted unless `paused == Some(true)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag accepted on deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Deferred start flag keeping hidden tabs idle until first rendered frame.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_initial_start: bool,
}

impl MorphogenesisToolState {
    /// Validates finiteness and mode-specific domain bounds for reaction-diffusion parameters.
    pub fn validate(&self) -> Result<(), String> {
        let base_checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("f", self.f, 0.0, 0.12),
            ("k", self.k, 0.0, 0.12),
        ];
        for (name, val, min, max) in base_checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Morphogenesis parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        let transport_checks = match self.kinetics_mode {
            MorphogenesisKineticsMode::Schnakenberg => [
                ("d_u", self.d_u, 0.1, 5.0),
                ("d_v", self.d_v, 10.0, 200.0),
                ("dt", self.dt, 1e-4, 0.1),
            ],
            MorphogenesisKineticsMode::GrayScott => [
                ("d_u", self.d_u, 0.01, 1.0),
                ("d_v", self.d_v, 0.005, 1.0),
                ("dt", self.dt, 1e-4, 2.0),
            ],
        };
        for (name, val, min, max) in transport_checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Morphogenesis parameter '{name}' = {val} must be finite within [{min}, {max}]"
                ));
            }
        }
        if self.width == 0 || self.width > 512 || self.height == 0 || self.height > 512 {
            return Err(format!(
                "Morphogenesis dimensions {}x{} must be within 1..=512",
                self.width, self.height
            ));
        }
        Ok(())
    }

    pub(crate) fn resolve_target_state(&self) -> SimWorkerState {
        self.state.unwrap_or_else(|| {
            if self.paused.unwrap_or(false) {
                SimWorkerState::Ready
            } else {
                SimWorkerState::Running
            }
        })
    }
}

/// Deterministic PCG-style PRNG used to seed reproducible Schnakenberg initial perturbations.
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

/// GUI [`TheoryDescribable`] wrapper around [`SchnakenbergKinetics`] and [`GrayScottKinetics`].
#[derive(Clone, Debug)]
pub struct MorphogenesisTheory {
    pub(crate) mode: MorphogenesisKineticsMode,
    schnakenberg: SchnakenbergKinetics,
    gray_scott: GrayScottKinetics,
}

impl Default for MorphogenesisTheory {
    fn default() -> Self {
        Self {
            mode: MorphogenesisKineticsMode::Schnakenberg,
            schnakenberg: SchnakenbergKinetics { a: 0.1, b: 0.9 },
            gray_scott: GrayScottKinetics::mitosis(),
        }
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
            "activator_diffusion".to_string(),
            "Spatial diffusion coefficient D_u of morphogen species u.".to_string(),
        );
        descs.insert(
            "inhibitor_diffusion".to_string(),
            "Spatial diffusion coefficient D_v of morphogen species v.".to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        match self.mode {
            MorphogenesisKineticsMode::Schnakenberg => {
                self.schnakenberg.theory_latex_equation().or_else(|| {
                    Some(
                        r"\frac{\partial u}{\partial t} = D_u \nabla^2 u + a - u + u^2 v,\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + b - u^2 v"
                            .to_string(),
                    )
                })
            }
            MorphogenesisKineticsMode::GrayScott => {
                self.gray_scott.theory_latex_equation().or_else(|| {
                    Some(
                        r"\frac{\partial u}{\partial t} = D_u \nabla^2 u - u v^2 + f(1 - u),\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + u v^2 - (f + k)v"
                            .to_string(),
                    )
                })
            }
        }
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
        id: "gray_scott_mitosis",
        title: "Gray-Scott Mitosis",
        description: "∂v/∂t = D_v∇²v + uv² - (f+k)v: Autocatalytic spot division and self-replicating wavefronts.",
        category: "Reaction-Diffusion",
        icon: "🦠",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_MITOSIS,
    }
}
