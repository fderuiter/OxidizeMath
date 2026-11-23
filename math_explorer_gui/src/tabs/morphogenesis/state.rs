//! Validated serializable state schema for `MorphogenesisTool`.

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    default_morpho_atlas_axes, is_default_morpho_atlas_axes, MORPHO_ATLAS_AXIS_PAIRS,
};
use super::types::{
    default_brush_radius, default_viscosity, is_default_brush_radius, is_default_viscosity,
    is_zero_f64, MorphogenesisBrushMode, MorphogenesisFlowMode, MorphogenesisKineticsMode,
    MorphogenesisOverlayMode,
};
use crate::async_sim::SimWorkerState;
use crate::framework::decode_rle_mask;
use math_explorer::biology::morphogenesis::GrayScottKinetics;
use serde::{Deserialize, Serialize};

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

/// Complete validated state schema for [`super::MorphogenesisTool`], compatible with `PRESET_TURING_*`,
/// `PRESET_GRAY_SCOTT_MITOSIS`, microfluidic `mask_rle` sharing, and advected flow presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    #[serde(default = "default_morpho_a")]
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    #[serde(default = "default_morpho_b")]
    pub b: f64,
    /// Primary species diffusion coefficient `D_u`.
    pub d_u: f64,
    /// Secondary species diffusion coefficient `D_v`.
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
    #[serde(default, alias = "kinetics")]
    pub kinetics_mode: MorphogenesisKineticsMode,
    /// Gray-Scott substrate feed rate `f` (`0.0..=0.12`).
    #[serde(default = "default_gray_scott_f")]
    pub f: f64,
    /// Gray-Scott autocatalyst kill rate `k` (`0.0..=0.12`).
    #[serde(default = "default_gray_scott_k")]
    pub k: f64,
    /// Hydrodynamic velocity coupling mode (`None`, `LbmVortexWake`, `PotentialFlow`).
    #[serde(default, skip_serializing_if = "MorphogenesisFlowMode::is_default")]
    pub flow_mode: MorphogenesisFlowMode,
    /// Horizontal inlet / freestream velocity magnitude (`0.0..=2.0`).
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub inlet_velocity: f64,
    /// Kinematic fluid viscosity `\nu` (`0.003..=0.5`).
    #[serde(default = "default_viscosity", skip_serializing_if = "is_default_viscosity")]
    pub viscosity: f64,
    /// Analytical vortex circulation `\Gamma` (`-25.0..=25.0`).
    #[serde(default, skip_serializing_if = "is_zero_f64")]
    pub circulation: f64,
    /// Visualization overlay mode (`Concentration`, `Vorticity`, `DualOverlay`).
    #[serde(default, skip_serializing_if = "MorphogenesisOverlayMode::is_default")]
    pub overlay_mode: MorphogenesisOverlayMode,
    /// Coupled obstacle/reagent sculpting brush mode.
    #[serde(default, skip_serializing_if = "MorphogenesisBrushMode::is_default")]
    pub brush_mode: MorphogenesisBrushMode,
    /// Coupled sculpting brush radius in cells (`1..=25`).
    #[serde(
        default = "default_brush_radius",
        skip_serializing_if = "is_default_brush_radius"
    )]
    pub brush_radius: i32,
    /// Optional worker execution state (`Running` when omitted unless `paused == Some(true)`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state: Option<SimWorkerState>,
    /// Legacy boolean pause flag accepted on deserialization.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paused: Option<bool>,
    /// Deferred start flag keeping hidden tabs idle until first rendered frame.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub pending_initial_start: bool,
    /// Optional RLE-encoded `ReactionCellRole` mask (`[role, run_len]` pairs).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mask_rle: Vec<[u32; 2]>,
    /// Persisted visibility of the progressive 2D parameter atlas (never raster pixels).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub atlas_visible: bool,
    /// Persisted `(x, y)` axis parameter keys for the 2D parameter atlas.
    #[serde(
        default = "default_morpho_atlas_axes",
        skip_serializing_if = "is_default_morpho_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
}

impl Default for MorphogenesisToolState {
    fn default() -> Self {
        Self {
            a: default_morpho_a(),
            b: default_morpho_b(),
            d_u: 1.0,
            d_v: 100.0,
            dt: 0.05,
            width: default_grid_dim(),
            height: default_grid_dim(),
            kinetics_mode: MorphogenesisKineticsMode::Schnakenberg,
            f: default_gray_scott_f(),
            k: default_gray_scott_k(),
            flow_mode: MorphogenesisFlowMode::None,
            inlet_velocity: 0.0,
            viscosity: default_viscosity(),
            circulation: 0.0,
            overlay_mode: MorphogenesisOverlayMode::Concentration,
            brush_mode: MorphogenesisBrushMode::Obstacle,
            brush_radius: default_brush_radius(),
            state: None,
            paused: None,
            pending_initial_start: false,
            mask_rle: Vec::new(),
            atlas_visible: false,
            atlas_axes: default_morpho_atlas_axes(),
        }
    }
}

impl MorphogenesisToolState {
    /// Validates finiteness, mode-specific domain bounds, RLE mask bounds, and atlas axis keys.
    pub fn validate(&self) -> Result<(), String> {
        let base_checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("f", self.f, 0.0, 0.12),
            ("k", self.k, 0.0, 0.12),
            ("inlet_velocity", self.inlet_velocity, 0.0, 2.0),
            ("viscosity", self.viscosity, 0.003, 0.5),
            ("circulation", self.circulation, -25.0, 25.0),
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
        if !(1..=25).contains(&self.brush_radius) {
            return Err(format!(
                "Morphogenesis brush_radius = {} must be within 1..=25",
                self.brush_radius
            ));
        }
        if self.width == 0 || self.width > 512 || self.height == 0 || self.height > 512 {
            return Err(format!(
                "Morphogenesis dimensions {}x{} must be within 1..=512",
                self.width, self.height
            ));
        }
        if !self.mask_rle.is_empty() {
            decode_rle_mask(&self.mask_rle, self.width * self.height)?;
        }
        let valid_axes = MORPHO_ATLAS_AXIS_PAIRS
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
