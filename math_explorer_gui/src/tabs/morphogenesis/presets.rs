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
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
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
            let mut ks = self
                .kinetics_state
                .write()
                .unwrap_or_else(|e| e.into_inner());
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
            p.mask_rle = None;
        }
        self.mask_cells.fill(0);
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
    /// Optional RLE-encoded spatial `(a, b)` reaction mask (`0`=base, `1`=stripe, `2`=spot).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mask_rle: Option<String>,
    /// Persisted visibility of the progressive 2D parameter atlas (never raster pixels).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub atlas_visible: bool,
    /// Persisted `(x, y)` axis parameter keys for the 2D parameter atlas.
    #[serde(
        default = "super::atlas::default_morpho_atlas_axes",
        skip_serializing_if = "super::atlas::is_default_morpho_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
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
        if let Some(ref rle) = self.mask_rle {
            let _ = crate::framework::decode_u8_grid(rle, self.width, self.height, 2)?;
        }
        let valid_axes = super::atlas::MORPHO_ATLAS_AXIS_PAIRS
            .iter()
            .any(|&(xk, yk)| xk == self.atlas_axes[0] && yk == self.atlas_axes[1]);
        if !valid_axes {
            return Err(format!(
                "Invalid Morphogenesis atlas axes ('{}', '{}')",
                self.atlas_axes[0], self.atlas_axes[1]
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

pub use crate::volumetric_scene::presets::SimpleRng;

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
