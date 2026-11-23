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
    /// Validates finiteness, parameter domain bounds, and atlas axis keys.
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
        Some(
            r"\frac{\partial u}{\partial t} = D_u \nabla^2 u + R_u(u,v),\quad \frac{\partial v}{\partial t} = D_v \nabla^2 v + R_v(u,v)"
                .to_string(),
        )
    }
}

#[cfg(test)]
pub use super::starter_presets::*;
