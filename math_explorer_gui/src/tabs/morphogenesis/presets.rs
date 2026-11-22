#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::async_sim::SimWorkerState;
use math_explorer::biology::morphogenesis::{GrayScottKinetics, SchnakenbergKinetics};
use scientific_metadata::theory::TheoryDescribable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub(crate) fn default_grid_dim() -> usize {
    100
}

pub(crate) fn default_gs_f() -> f64 {
    0.037
}

pub(crate) fn default_gs_k() -> f64 {
    0.060
}

/// Selects the active reaction-diffusion solver backend inside [`super::MorphogenesisTool`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum MorphogenesisModelKind {
    /// Two-species Schnakenberg activator-inhibitor system (`TuringSystem`).
    #[default]
    Schnakenberg,
    /// Two-species Gray-Scott cubic autocatalytic system (`TuringSystem`).
    GrayScott,
    /// Generic $N$-species `ReactionDiffusionSystem` (`ChemicalState` + `GrayScottKinetics`).
    GenericReactionDiffusion,
}

/// Complete validated state schema for [`super::MorphogenesisTool`], backward-compatible with `PRESET_TURING_*`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    pub b: f64,
    /// Activator diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Inhibitor diffusion coefficient `D_v`.
    pub d_v: f64,
    /// Integration time step `dt`.
    pub dt: f64,
    /// Spatial grid width (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub width: usize,
    /// Spatial grid height (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub height: usize,
    /// Reaction-diffusion solver backend (`Schnakenberg`, `GrayScott`, or `GenericReactionDiffusion`).
    #[serde(default)]
    pub model: MorphogenesisModelKind,
    /// Gray-Scott dimensionless feed rate `f` (`0.005..=0.12`).
    #[serde(default = "default_gs_f")]
    pub f: f64,
    /// Gray-Scott dimensionless kill rate `k` (`0.03..=0.08`).
    #[serde(default = "default_gs_k")]
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
    /// Validates finiteness and domain bounds for all reaction-diffusion parameters.
    pub fn validate(&self) -> Result<(), String> {
        let (du_min, dv_min, dt_max) = match self.model {
            MorphogenesisModelKind::Schnakenberg => (0.1, 10.0, 0.1),
            MorphogenesisModelKind::GrayScott
            | MorphogenesisModelKind::GenericReactionDiffusion => (0.01, 0.01, 1.0),
        };
        let checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("f", self.f, 0.005, 0.12),
            ("k", self.k, 0.03, 0.08),
            ("d_u", self.d_u, du_min, 5.0),
            ("d_v", self.d_v, dv_min, 200.0),
            ("dt", self.dt, 1e-4, dt_max),
        ];
        for (name, val, min, max) in checks {
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

/// GUI [`TheoryDescribable`] wrapper around [`SchnakenbergKinetics`] and [`GrayScottKinetics`].
#[derive(Clone, Debug)]
pub struct MorphogenesisTheory {
    schnakenberg: SchnakenbergKinetics,
    gray_scott: GrayScottKinetics,
}

impl Default for MorphogenesisTheory {
    fn default() -> Self {
        Self {
            schnakenberg: SchnakenbergKinetics { a: 0.1, b: 0.9 },
            gray_scott: GrayScottKinetics::default(),
        }
    }
}

impl TheoryDescribable for MorphogenesisTheory {
    fn theory_description(&self) -> String {
        format!(
            "{} {}",
            self.schnakenberg.theory_description(),
            self.gray_scott.theory_description()
        )
    }

    fn phonetic_description(&self) -> String {
        self.schnakenberg.phonetic_description()
    }

    fn theory_citation(&self) -> String {
        format!(
            "{}; {}",
            self.schnakenberg.theory_citation(),
            self.gray_scott.theory_citation()
        )
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut descs = self.schnakenberg.available_descriptions();
        for (k, v) in self.gray_scott.available_descriptions() {
            descs.insert(k, v);
        }
        descs.insert(
            "activator_diffusion".to_string(),
            "Short-range spatial diffusion coefficient D_u of autocatalytic species u.".to_string(),
        );
        descs.insert(
            "inhibitor_diffusion".to_string(),
            "Spatial diffusion coefficient D_v driving differential-diffusion instability."
                .to_string(),
        );
        descs
    }

    fn theory_latex_equation(&self) -> Option<String> {
        Some(
            r"\frac{\partial u}{\partial t} = D_u \nabla^2 u + R_u(u,v),\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + R_v(u,v)"
                .to_string(),
        )
    }
}

/// Starter preset JSON envelope for Gray-Scott Coral morphogenesis.
pub const PRESET_GRAY_SCOTT_CORAL: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Gray-Scott Coral Morphogenesis",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "a": 0.1,
    "b": 0.9,
    "d_u": 0.2,
    "d_v": 0.1,
    "dt": 0.5,
    "width": 100,
    "height": 100,
    "model": "GrayScott",
    "f": 0.037,
    "k": 0.06
  }
}"#;

/// Starter preset JSON envelope for Lenia `Orbium` glider.
pub const PRESET_LENIA_ORBIUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Continuous Lenia - Orbium Glider",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mu": 0.15,
    "sigma": 0.015,
    "dt": 0.1,
    "kernel_radius": 13,
    "preset": "Orbium",
    "width": 128,
    "height": 128
  }
}"#;

/// Starter preset JSON envelope for Lenia `Hydrogeminium` self-dividing organism.
pub const PRESET_LENIA_HYDROGEMINIUM: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Continuous Lenia - Hydrogeminium",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "mu": 0.26,
    "sigma": 0.036,
    "dt": 0.1,
    "kernel_radius": 18,
    "preset": "Hydrogeminium",
    "width": 128,
    "height": 128
  }
}"#;

/// Starter preset JSON envelope for Vogel Golden-Angle Sunflower phyllotaxis.
pub const PRESET_PHYLLOTAXIS_SUNFLOWER: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Phyllotaxis - Golden-Angle Sunflower",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "divergence_angle_deg": 137.50776405003785,
    "scale_c": 0.45,
    "primordia_count": 220,
    "substrate": 2.5,
    "v_max": 1.0,
    "k_m": 1.0,
    "hill_n": 2.5,
    "kinetics": "MichaelisMenten",
    "lsystem_preset": "BotanicalShoot",
    "lsystem_iterations": 4,
    "branch_angle_deg": 25.0
  }
}"#;

/// Starter preset JSON envelope for Cooperative Hill Nutrient L-System Shoot.
pub const PRESET_PHYLLOTAXIS_HILL_SHOOT: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Phyllotaxis - Cooperative Hill Shoot",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "divergence_angle_deg": 137.50776405003785,
    "scale_c": 0.48,
    "primordia_count": 144,
    "substrate": 3.2,
    "v_max": 1.2,
    "k_m": 1.0,
    "hill_n": 3.2,
    "kinetics": "Hill",
    "lsystem_preset": "SympodialTree",
    "lsystem_iterations": 4,
    "branch_angle_deg": 22.5
  }
}"#;

/// Starter preset JSON envelope for Hawk-Dove Replicator ESS & Hill Kinetics.
pub const PRESET_EVOLUTION_KINETICS_ESS: &str = r#"{
  "metadata": {
    "schema_version": "1.0",
    "title": "Evolution & Enzyme Kinetics - Hawk-Dove ESS",
    "domain": "biology",
    "timestamp": "2026-11-21T00:00:00Z"
  },
  "config": {
    "resource_value": 2.0,
    "cost_of_injury": 4.0,
    "initial_hawk_freq": 0.15,
    "time_horizon": 25.0,
    "dt": 0.05,
    "v_max": 10.0,
    "k_m": 1.5,
    "hill_n": 3.0,
    "substrate_probe": 2.0
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
        id: "gray_scott_coral",
        title: "Gray-Scott Coral",
        description: "∂u/∂t = D_u∇²u - uv² + f(1-u): Cubic autocatalysis forming self-replicating coral labyrinths.",
        category: "Reaction-Diffusion",
        icon: "🪸",
        domain: "biology",
        tool_name: Some("MorphogenesisTool"),
        preset_generator: || PRESET_GRAY_SCOTT_CORAL,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lenia_orbium",
        title: "Lenia Orbium Glider",
        description: "U = F⁻¹(F(K)·F(A)): Spectral continuous cellular automaton glider translating across a periodic torus.",
        category: "Artificial Life",
        icon: "🦠",
        domain: "biology",
        tool_name: Some("LeniaTool"),
        preset_generator: || PRESET_LENIA_ORBIUM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "lenia_hydrogeminium",
        title: "Lenia Hydrogeminium",
        description: "Multi-ring concentric bump kernel lifeform exhibiting oscillatory elongation and fission.",
        category: "Artificial Life",
        icon: "🫧",
        domain: "biology",
        tool_name: Some("LeniaTool"),
        preset_generator: || PRESET_LENIA_HYDROGEMINIUM,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "phyllotaxis_golden_sunflower",
        title: "Golden-Angle Sunflower",
        description: "r_k = c√(k+0.5), θ_k = k·137.508°: Vogel floret packing with consecutive Fibonacci parastichies.",
        category: "Phyllotaxis",
        icon: "🌻",
        domain: "biology",
        tool_name: Some("PhyllotaxisLSystemTool"),
        preset_generator: || PRESET_PHYLLOTAXIS_SUNFLOWER,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "phyllotaxis_hill_shoot",
        title: "Cooperative Hill Shoot",
        description: "Parallel L-system branching coupled to sigmoidal Hill enzyme nutrient uptake kinetics.",
        category: "L-Systems",
        icon: "🌿",
        domain: "biology",
        tool_name: Some("PhyllotaxisLSystemTool"),
        preset_generator: || PRESET_PHYLLOTAXIS_HILL_SHOOT,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "evolution_hawk_dove_ess",
        title: "Hawk-Dove & Hill Kinetics",
        description: "dp/dt = p(1-p)(W_H - W_D) & v = V_max[S]^n/(K_m^n + [S]^n): ESS replicator & enzyme saturation.",
        category: "Evolution & Kinetics",
        icon: "🦅",
        domain: "biology",
        tool_name: Some("EvolutionKineticsTool"),
        preset_generator: || PRESET_EVOLUTION_KINETICS_ESS,
    }
}
