#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{
    default_morpho_atlas_axes, is_default_morpho_atlas_axes, MORPHO_ATLAS_AXIS_PAIRS,
};
use super::MorphogenesisTool;
use crate::async_sim::{SimCommand, SimWorkerState};
use crate::framework::recovery::{DiagnosticReport, FiniteStateGuard};
use crate::framework::{
    decode_u8_grid, draw_brush_palette_controls, encode_u8_grid, RasterBrushStroke,
};
use eframe::egui;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SpatialSchnakenbergKinetics, TuringSystem};
use serde::{Deserialize, Serialize};

fn default_grid_dim() -> usize {
    100
}

/// Complete validated state schema for [`MorphogenesisTool`], compatible with `PRESET_TURING_*`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    pub b: f64,
    /// Activator diffusion coefficient `D_u` (`0.1..=5.0`).
    pub d_u: f64,
    /// Inhibitor diffusion coefficient `D_v` (`10.0..=200.0`).
    pub d_v: f64,
    /// Integration time step `dt` (`1e-4..=0.1`).
    pub dt: f64,
    /// Spatial grid width (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub width: usize,
    /// Spatial grid height (`1..=512`).
    #[serde(default = "default_grid_dim")]
    pub height: usize,
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
        default = "default_morpho_atlas_axes",
        skip_serializing_if = "is_default_morpho_atlas_axes"
    )]
    pub atlas_axes: [String; 2],
}

impl MorphogenesisToolState {
    /// Validates finiteness, parameter domain bounds, spatial mask RLE, and atlas axis keys.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("d_u", self.d_u, 0.1, 5.0),
            ("d_v", self.d_v, 10.0, 200.0),
            ("dt", self.dt, 1e-4, 0.1),
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
        if let Some(ref rle) = self.mask_rle {
            let _ = decode_u8_grid(rle, self.width, self.height, 2)?;
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

/// Schnakenberg `(a, b)` kinetic rates stamped by [`MorphoBrushMode::StripeMask`].
pub const STRIPE_MASK_RATES: (f64, f64) = (0.14, 0.86);
/// Schnakenberg `(a, b)` kinetic rates stamped by [`MorphoBrushMode::SpotMask`].
pub const SPOT_MASK_RATES: (f64, f64) = (0.08, 1.12);

/// Interactive chemical perturbation and spatial reaction-rate brush modes for [`MorphogenesisTool`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MorphoBrushMode {
    /// Injects autocatalytic activator `u` concentration seeds.
    #[default]
    ActivatorSeed,
    /// Injects long-range inhibitor `v` concentration perturbations.
    InhibitorSeed,
    /// Stamps spatial stripe-regime `(a, b)` overrides onto `SpatialSchnakenbergKinetics`.
    StripeMask,
    /// Stamps spatial spot-regime `(a, b)` overrides onto `SpatialSchnakenbergKinetics`.
    SpotMask,
    /// Clears spatial kinetic overrides and equilibrates touched cells.
    EraseMask,
}

impl MorphoBrushMode {
    /// All selectable morphogenesis brush modes in UI display order.
    pub const ALL: [Self; 5] = [
        Self::ActivatorSeed,
        Self::InhibitorSeed,
        Self::StripeMask,
        Self::SpotMask,
        Self::EraseMask,
    ];

    /// User-facing label for side-panel controls.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ActivatorSeed => "Seed Activator (u+)",
            Self::InhibitorSeed => "Seed Inhibitor (v+)",
            Self::StripeMask => "Stripe Kinetics Mask (a, b)",
            Self::SpotMask => "Spot Kinetics Mask (a, b)",
            Self::EraseMask => "Clear Mask / Equilibrate",
        }
    }

    /// Wire command byte passed via [`SimCommand::ApplyBrushCells`].
    #[must_use]
    pub const fn to_code(self) -> u8 {
        match self {
            Self::EraseMask => 0,
            Self::ActivatorSeed => 1,
            Self::InhibitorSeed => 2,
            Self::StripeMask => 3,
            Self::SpotMask => 4,
        }
    }

    /// Decodes a wire command byte back to [`MorphoBrushMode`].
    #[must_use]
    pub const fn from_code(code: u8) -> Self {
        match code {
            1 => Self::ActivatorSeed,
            2 => Self::InhibitorSeed,
            3 => Self::StripeMask,
            4 => Self::SpotMask,
            _ => Self::EraseMask,
        }
    }
}

/// Applies an RLE-encoded spatial mask (`0` = baseline, `1` = stripe, `2` = spot) onto `kinetics`.
pub fn apply_mask_rle_to_kinetics(
    kinetics: &mut SpatialSchnakenbergKinetics,
    width: usize,
    height: usize,
    mask_rle: Option<&str>,
) {
    kinetics.clear_mask();
    let Some(rle) = mask_rle else {
        return;
    };
    let Ok(decoded) = decode_u8_grid(rle, width, height, 2) else {
        return;
    };
    for (idx, sym) in decoded.into_iter().enumerate() {
        match sym {
            1 => kinetics.set_cell_rates(idx, STRIPE_MASK_RATES.0, STRIPE_MASK_RATES.1),
            2 => kinetics.set_cell_rates(idx, SPOT_MASK_RATES.0, SPOT_MASK_RATES.1),
            _ => {}
        }
    }
}

/// Applies a [`MorphoBrushMode`] cell batch directly to `TuringSystem` slices and kinetic masks.
pub fn apply_morpho_cells_to_system(
    system: &mut TuringSystem<2, SpatialSchnakenbergKinetics, FiniteDifference2D>,
    width: usize,
    height: usize,
    cells: &[(usize, usize)],
    mode: MorphoBrushMode,
) {
    let total = width * height;
    for &(x, y) in cells {
        if x >= width || y >= height {
            continue;
        }
        let idx = y * width + x;
        if idx >= total {
            continue;
        }
        match mode {
            MorphoBrushMode::ActivatorSeed => {
                let next_u = (system.u()[idx] + 0.75).clamp(0.0, 8.0);
                system.u_mut()[idx] = next_u;
            }
            MorphoBrushMode::InhibitorSeed => {
                let next_v = (system.v()[idx] + 0.75).clamp(0.0, 8.0);
                system.v_mut()[idx] = next_v;
            }
            MorphoBrushMode::StripeMask => {
                system
                    .kinetics
                    .set_cell_rates(idx, STRIPE_MASK_RATES.0, STRIPE_MASK_RATES.1);
                system.u_mut()[idx] = (system.u()[idx] + 0.15).clamp(0.0, 8.0);
            }
            MorphoBrushMode::SpotMask => {
                system
                    .kinetics
                    .set_cell_rates(idx, SPOT_MASK_RATES.0, SPOT_MASK_RATES.1);
                system.u_mut()[idx] = (system.u()[idx] + 0.25).clamp(0.0, 8.0);
            }
            MorphoBrushMode::EraseMask => {
                system.kinetics.clear_cell_override(idx);
                let a = system.kinetics.a;
                let b = system.kinetics.b;
                let u_eq = (a + b).max(1e-3);
                system.u_mut()[idx] = u_eq;
                system.v_mut()[idx] = b / (u_eq * u_eq);
            }
        }
    }
}

impl MorphogenesisTool {
    /// Applies a [`RasterBrushStroke`] to the local spatial mask mirror and dispatches to the worker.
    pub fn apply_brush_stroke(&mut self, stroke: RasterBrushStroke, ctx: Option<&egui::Context>) {
        let effective_mode = if stroke.secondary {
            MorphoBrushMode::EraseMask
        } else {
            self.brush_mode
        };
        if self.mask_cells.len() != self.width * self.height {
            self.mask_cells.resize(self.width * self.height, 0);
        }
        let maybe_sym = match effective_mode {
            MorphoBrushMode::StripeMask => Some(1u8),
            MorphoBrushMode::SpotMask => Some(2u8),
            MorphoBrushMode::EraseMask => Some(0u8),
            MorphoBrushMode::ActivatorSeed | MorphoBrushMode::InhibitorSeed => None,
        };
        if let Some(sym) = maybe_sym {
            for &(x, y) in &stroke.cells {
                if x < self.width && y < self.height {
                    self.mask_cells[y * self.width + x] = sym;
                }
            }
            self.sync_mask_rle_from_cells();
        }
        let _ = self.worker.send_command(SimCommand::ApplyBrushCells {
            cells: stroke.cells,
            mode: effective_mode.to_code(),
        });
        self.step_runner = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn sync_mask_rle_from_cells(&mut self) {
        let next_rle = if self.mask_cells.iter().any(|&s| s != 0) {
            encode_u8_grid(self.width, self.height, &self.mask_cells, 2).ok()
        } else {
            None
        };
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mask_rle = next_rle;
    }

    /// Clears all spatial `(a, b)` reaction mask overrides on both UI and worker threads.
    pub fn clear_spatial_mask(&mut self, ctx: Option<&egui::Context>) {
        self.mask_cells.fill(0);
        self.params
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .mask_rle = None;
        let _ = self
            .worker
            .send_command(SimCommand::Custom("clear_mask".to_string()));
        self.step_runner = None;
        self.dirty = true;
        if let Some(c) = ctx {
            c.request_repaint();
        }
    }

    pub(crate) fn draw_brush_section(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.separator();
        ui.label("Chemical & Mask Brush");
        egui::ComboBox::from_id_salt("morpho_brush_mode")
            .selected_text(self.brush_mode.label())
            .show_ui(ui, |ui| {
                for mode in MorphoBrushMode::ALL {
                    ui.selectable_value(&mut self.brush_mode, mode, mode.label());
                }
            });
        draw_brush_palette_controls(ui, "morphogenesis", &mut self.brush_state, 12);
        if ui.button("Clear Spatial Mask").clicked() {
            self.clear_spatial_mask(Some(ctx));
        }
    }
}

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
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = p.a.clamp(0.0, 1.0);
        p.b = p.b.clamp(0.0, 2.0);
        p.d_u = p.d_u.clamp(0.1, 5.0);
        p.d_v = p.d_v.clamp(10.0, 200.0);
        p.dt = p.dt.clamp(0.001, 0.1);
        drop(p);
        self.reset_system();
    }

    fn reset_parameters(&mut self) {
        let mut p = self.params.write().unwrap_or_else(|e| e.into_inner());
        p.a = 0.1;
        p.b = 0.9;
        p.d_u = 1.0;
        p.d_v = 100.0;
        p.dt = 0.05;
        p.mask_rle = None;
        drop(p);
        self.mask_cells.fill(0);
        self.reset_system();
    }
}

