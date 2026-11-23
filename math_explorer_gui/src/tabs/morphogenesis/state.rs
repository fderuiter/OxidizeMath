#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    default_morpho_atlas_axes, is_default_morpho_atlas_axes, MORPHO_ATLAS_AXIS_PAIRS,
};
use super::types::{
    default_brush_radius, default_viscosity, MorphogenesisBrushMode, MorphogenesisFlowMode,
    MorphogenesisKineticsMode, MorphogenesisOverlayMode,
};
use crate::async_sim::SimWorkerState;
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
/// `PRESET_GRAY_SCOTT_MITOSIS`, and advected flow presets.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    #[serde(default = "default_morpho_a")]
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    #[serde(default = "default_morpho_b")]
    pub b: f64,
    /// Activator/substrate diffusion coefficient `D_u` (`0.005..=5.0`).
    pub d_u: f64,
    /// Inhibitor/autocatalyst diffusion coefficient `D_v` (`0.005..=200.0`).
    pub d_v: f64,
    /// Integration time step `dt` (`1e-4..=2.0`).
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
    #[serde(default)]
    pub flow_mode: MorphogenesisFlowMode,
    /// Horizontal inlet / freestream velocity magnitude (`0.0..=2.0`).
    #[serde(default)]
    pub inlet_velocity: f64,
    /// Kinematic fluid viscosity `\nu` (`0.003..=0.5`).
    #[serde(default = "default_viscosity")]
    pub viscosity: f64,
    /// Analytical vortex circulation `\Gamma` (`-25.0..=25.0`).
    #[serde(default)]
    pub circulation: f64,
    /// Visualization overlay mode (`Concentration`, `Vorticity`, `DualOverlay`).
    #[serde(default)]
    pub overlay_mode: MorphogenesisOverlayMode,
    /// Active pointer sculpting brush mode.
    #[serde(default)]
    pub brush_mode: MorphogenesisBrushMode,
    /// Pointer sculpting brush radius in cells (`1..=25`).
    #[serde(default = "default_brush_radius")]
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

impl MorphogenesisToolState {
    /// Validates finiteness, domain bounds, and atlas axis keys for all morphogenesis parameters.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("f", self.f, 0.0, 0.12),
            ("k", self.k, 0.0, 0.12),
            ("d_u", self.d_u, 0.005, 5.0),
            ("d_v", self.d_v, 0.005, 200.0),
            ("dt", self.dt, 1e-4, 2.0),
            ("inlet_velocity", self.inlet_velocity, 0.0, 2.0),
            ("viscosity", self.viscosity, 0.003, 0.5),
            ("circulation", self.circulation, -25.0, 25.0),
        ];
        for (name, val, min, max) in checks {
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

impl super::MorphogenesisTool {
    pub(crate) fn apply_named_preset(&mut self, a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.apply_full_preset(
            MorphogenesisKineticsMode::Schnakenberg,
            a,
            b,
            p.f,
            p.k,
            d_u,
            d_v,
            dt,
        );
    }

    pub(crate) fn apply_gray_scott_preset(&mut self, f: f64, k: f64) {
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        self.apply_full_preset(
            MorphogenesisKineticsMode::GrayScott,
            p.a,
            p.b,
            f,
            k,
            0.16,
            0.08,
            1.0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_full_preset(
        &mut self,
        kinetics_mode: MorphogenesisKineticsMode,
        a: f64,
        b: f64,
        f: f64,
        k: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) {
        use crate::framework::InteractiveTool;
        let target_state = if self.worker.is_running() {
            SimWorkerState::Running
        } else {
            SimWorkerState::Ready
        };
        let p = self
            .params
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone();
        let (xk, yk) = self.atlas.axis_keys();
        let preset_state = MorphogenesisToolState {
            a,
            b,
            d_u,
            d_v,
            dt,
            width: self.width,
            height: self.height,
            kinetics_mode,
            f,
            k,
            flow_mode: p.flow_mode,
            inlet_velocity: p.inlet_velocity,
            viscosity: p.viscosity,
            circulation: p.circulation,
            overlay_mode: p.overlay_mode,
            brush_mode: p.brush_mode,
            brush_radius: p.brush_radius,
            state: Some(target_state),
            paused: None,
            pending_initial_start: false,
            atlas_visible: self.atlas.is_visible(),
            atlas_axes: [xk.to_string(), yk.to_string()],
        };
        if let Ok(json) = serde_json::to_string(&preset_state) {
            let _ = self.try_load_state(&json);
        }
    }
}
