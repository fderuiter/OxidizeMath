//! Validated state schema, deterministic seeding, chemical brushing, and heatmap helpers
//! for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimpleRng;
use crate::async_sim::SimWorkerState;
pub use crate::presets::{
    default_auto_dt, default_gray_scott_f, default_gray_scott_k, MorphogenesisKinetics,
    DEFAULT_GRAY_SCOTT_F, DEFAULT_GRAY_SCOTT_K,
};
use eframe::egui::{self, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_commons::math_kernel::types::flatten_2d_index;
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{SchnakenbergKinetics, TuringSystem};
use serde::{Deserialize, Serialize};

const fn default_grid_dim() -> usize {
    100
}

const fn default_schnakenberg_a() -> f64 {
    0.1
}

const fn default_schnakenberg_b() -> f64 {
    0.9
}

/// Pointer interaction mode on the 2D morphogenesis plot canvas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CanvasInteractionMode {
    /// Primary drag injects reagent; secondary drag restores local equilibrium.
    #[default]
    Brush,
    /// Pointer drag and scroll pan/zoom the `egui_plot` viewport.
    Navigate,
}

/// Complete validated state schema for [`super::MorphogenesisTool`], backward-compatible with `PRESET_TURING_*`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisToolState {
    /// Active reaction kinetics model (`Schnakenberg` when omitted).
    #[serde(default)]
    pub kinetics: MorphogenesisKinetics,
    /// Schnakenberg activator production rate `a` (`0.0..=1.0`).
    #[serde(default = "default_schnakenberg_a")]
    pub a: f64,
    /// Schnakenberg inhibitor production rate `b` (`0.0..=2.0`).
    #[serde(default = "default_schnakenberg_b")]
    pub b: f64,
    /// Gray-Scott substrate feed rate `F` (`0.0..=0.2`).
    #[serde(default = "default_gray_scott_f")]
    pub f: f64,
    /// Gray-Scott catalyst kill rate `k` (`0.0..=0.2`).
    #[serde(default = "default_gray_scott_k")]
    pub k: f64,
    /// Species `u` diffusion coefficient `D_u` (`0.01..=5.0`).
    pub d_u: f64,
    /// Species `v` diffusion coefficient `D_v` (`0.01..=200.0`).
    pub d_v: f64,
    /// Integration time step `dt` (`1e-5..=2.0`).
    pub dt: f64,
    /// Automatic 2D CFL time-step clamping and substepping (`true` when omitted).
    #[serde(default = "default_auto_dt")]
    pub auto_dt: bool,
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
}

impl MorphogenesisToolState {
    /// Validates finiteness and domain bounds for all reaction-diffusion parameters.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("a", self.a, 0.0, 1.0),
            ("b", self.b, 0.0, 2.0),
            ("f", self.f, 0.0, 0.2),
            ("k", self.k, 0.0, 0.2),
            ("d_u", self.d_u, 0.01, 5.0),
            ("d_v", self.d_v, 0.01, 200.0),
            ("dt", self.dt, 1e-5, 2.0),
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

    pub(crate) fn resolve_target_state(
        &self,
        was_pending_initial_start: bool,
    ) -> (SimWorkerState, bool) {
        if let Some(explicit) = self.state {
            let pending = explicit != SimWorkerState::Running && self.pending_initial_start;
            return (explicit, pending);
        }
        if let Some(paused) = self.paused {
            return if paused {
                (SimWorkerState::Ready, self.pending_initial_start)
            } else {
                (SimWorkerState::Running, false)
            };
        }
        if was_pending_initial_start {
            (SimWorkerState::Ready, true)
        } else {
            (SimWorkerState::Running, false)
        }
    }
}

pub(crate) fn initialize_schnakenberg_system(
    system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    width: usize,
    height: usize,
) {
    let n = width * height;
    let mut rng = SimpleRng::new(12345);
    let a = system.kinetics.a;
    let b = system.kinetics.b;
    let u_eq = (a + b).max(1e-6);
    let v_eq = b / (u_eq * u_eq);

    for i in 0..n {
        system.u_mut()[i] = (u_eq + rng.range(-0.1, 0.1)).max(0.0);
        system.v_mut()[i] = (v_eq + rng.range(-0.1, 0.1)).max(0.0);
    }
}

pub(crate) fn initialize_gray_scott_system(
    system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    width: usize,
    height: usize,
) {
    system.u_mut().fill(1.0);
    system.v_mut().fill(0.0);
    if width == 0 || height == 0 {
        return;
    }
    let mut rng = SimpleRng::new(12345);
    let base_r = (width.min(height) / 16).clamp(2, 5) as i32;
    seed_gray_scott_patch(
        system,
        width,
        height,
        (width / 2) as i32,
        (height / 2) as i32,
        base_r,
        &mut rng,
    );
    for _ in 0..4 {
        let cx = (rng.range(0.22, 0.78) * (width as f64)).round() as i32;
        let cy = (rng.range(0.22, 0.78) * (height as f64)).round() as i32;
        let r = (base_r - 1).max(2);
        seed_gray_scott_patch(system, width, height, cx, cy, r, &mut rng);
    }
}

fn seed_gray_scott_patch(
    system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    width: usize,
    height: usize,
    cx: i32,
    cy: i32,
    r: i32,
    rng: &mut SimpleRng,
) {
    let x_min = (cx - r).max(0);
    let x_max = (cx + r).min(width as i32 - 1);
    let y_min = (cy - r).max(0);
    let y_max = (cy + r).min(height as i32 - 1);
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let idx = flatten_2d_index(x as usize, y as usize, width);
            system.u_mut()[idx] = (0.50 + rng.range(-0.04, 0.04)).clamp(0.0, 1.0);
            system.v_mut()[idx] = (0.25 + rng.range(-0.04, 0.04)).clamp(0.0, 1.0);
        }
    }
}

pub(crate) fn initialize_for_kinetics(
    system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    kinetics: MorphogenesisKinetics,
    width: usize,
    height: usize,
) {
    match kinetics {
        MorphogenesisKinetics::Schnakenberg => {
            initialize_schnakenberg_system(system, width, height);
        }
        MorphogenesisKinetics::GrayScott => {
            initialize_gray_scott_system(system, width, height);
        }
    }
}

/// Computes finite population spatial variance of `data` without heap allocation.
#[must_use]
pub fn spatial_variance(data: &[f64]) -> f64 {
    let mut count = 0_usize;
    let mut mean = 0.0_f64;
    let mut m2 = 0.0_f64;
    for &val in data {
        if val.is_finite() {
            count += 1;
            let delta = val - mean;
            mean += delta / (count as f64);
            let delta2 = val - mean;
            m2 += delta * delta2;
        }
    }
    if count == 0 {
        0.0
    } else {
        m2 / (count as f64)
    }
}

pub(crate) fn ensure_nucleated_for_kinetics(
    system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    kinetics: MorphogenesisKinetics,
    width: usize,
    height: usize,
) {
    let var_u = spatial_variance(system.u());
    let catalyst_extinct = kinetics == MorphogenesisKinetics::GrayScott
        && system
            .v()
            .iter()
            .copied()
            .fold(0.0_f64, |acc, v| if v.is_finite() { acc.max(v) } else { acc })
            < 1e-3;
    if var_u < 1e-7 || catalyst_extinct {
        initialize_for_kinetics(system, kinetics, width, height);
    }
}

pub(crate) fn apply_chemical_brush(
    system: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    kinetics: MorphogenesisKinetics,
    dims: (usize, usize),
    circle: (i32, i32, i32),
    inject_reagent: bool,
) {
    let (width, height) = dims;
    let (cx, cy, r) = circle;
    if width == 0 || height == 0 {
        return;
    }
    let radius = r.clamp(0, 64);
    let r_sq = radius * radius;
    let (target_u, target_v) = match (kinetics, inject_reagent) {
        (MorphogenesisKinetics::GrayScott, true) => (0.50, 0.25),
        (MorphogenesisKinetics::GrayScott, false) => (1.0, 0.0),
        (MorphogenesisKinetics::Schnakenberg, true) => {
            let u_eq = (system.kinetics.a + system.kinetics.b).max(1e-6);
            let v_eq = system.kinetics.b / (u_eq * u_eq);
            ((u_eq + 0.85).min(6.0), (v_eq * 0.5).max(0.0))
        }
        (MorphogenesisKinetics::Schnakenberg, false) => {
            let u_eq = (system.kinetics.a + system.kinetics.b).max(1e-6);
            let v_eq = system.kinetics.b / (u_eq * u_eq);
            (u_eq, v_eq)
        }
    };
    let x_min = (cx - radius).max(0);
    let x_max = (cx + radius).min(width as i32 - 1);
    let y_min = (cy - radius).max(0);
    let y_max = (cy + radius).min(height as i32 - 1);
    for y in y_min..=y_max {
        for x in x_min..=x_max {
            let dx = x - cx;
            let dy = y - cy;
            if dx * dx + dy * dy <= r_sq {
                let idx = flatten_2d_index(x as usize, y as usize, width);
                system.u_mut()[idx] = target_u;
                system.v_mut()[idx] = target_v;
            }
        }
    }
}

pub(crate) fn plot_concentration_with_contrast(
    data: &[f64],
    width: usize,
    height: usize,
    high_contrast: bool,
) -> ColorImage {
    let num_pixels = width * height;
    let mut pixels = Vec::with_capacity(num_pixels);
    let (min, max) = data
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &val| {
            if val.is_finite() {
                (lo.min(val), hi.max(val))
            } else {
                (lo, hi)
            }
        });
    let (min, max) = if min.is_finite() && max.is_finite() && max >= min {
        (min, max)
    } else {
        (0.0, 1.0)
    };
    let range = max - min;
    let inv_range = if range > 1e-6 { 1.0 / range } else { 1.0 };

    for &val in data {
        let safe_val = if val.is_finite() { val } else { min };
        let norm = (safe_val - min) * inv_range;
        let safe_norm = if norm.is_finite() {
            norm.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let rgb = if high_contrast {
            heatmap_color_high_contrast(safe_norm)
        } else {
            heatmap_color(safe_norm)
        };
        pixels.push(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
    }
    ColorImage {
        size: [width, height],
        pixels,
        ..Default::default()
    }
}
