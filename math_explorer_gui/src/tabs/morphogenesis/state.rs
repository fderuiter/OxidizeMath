#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::types::{
    default_brush_radius, default_viscosity, MorphogenesisBrushMode, MorphogenesisFlowMode,
    MorphogenesisKineticsMode, MorphogenesisOverlayMode,
};
use crate::async_sim::SimWorkerState;
use serde::{Deserialize, Serialize};

fn default_grid_dim() -> usize {
    100
}

/// Complete validated state schema for [`super::MorphogenesisTool`], compatible with `PRESET_TURING_*`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Reaction kinetic parameter `a` (`0.0..=1.0`, or Gray-Scott feed rate `f`).
    pub a: f64,
    /// Reaction kinetic parameter `b` (`0.0..=2.0`, or Gray-Scott kill rate `k`).
    pub b: f64,
    /// Activator/substrate diffusion coefficient `D_u` (`0.01..=5.0`).
    pub d_u: f64,
    /// Inhibitor/autocatalyst diffusion coefficient `D_v` (`0.01..=200.0`).
    pub d_v: f64,
    /// Integration time step `dt` (`1e-4..=1.0`).
    pub dt: f64,
    /// Spatial grid width (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub width: usize,
    /// Spatial grid height (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub height: usize,
    /// Reaction kinetics strategy (`Schnakenberg` or `GrayScott`).
    #[serde(default)]
    pub kinetics: MorphogenesisKineticsMode,
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
}

impl MorphogenesisToolState {
    /// Validates finiteness and domain bounds for all reaction, diffusion, and flow parameters.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("d_u", self.d_u, 0.01, 5.0),
            ("d_v", self.d_v, 0.01, 200.0),
            ("dt", self.dt, 1e-4, 1.0),
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
