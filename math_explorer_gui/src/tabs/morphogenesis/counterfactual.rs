//! Counterfactual shadow twin state and spatial difference heatmaps for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::SimpleRng;
use crate::app::ThemeMode;
use crate::framework::{
    field_difference_norms, sanitize_perturbation_delta, DivergenceSnapshot,
    DomainInterventionDelta, SeparationTracker,
};
use eframe::egui::{Color32, ColorImage};
use math_commons::math_kernel::colormap::{heatmap_color, heatmap_color_high_contrast};
use math_explorer::biology::diffusion::FiniteDifference2D;
use math_explorer::biology::morphogenesis::{DiffusionCoeff, SchnakenbergKinetics, TuringSystem};
use math_explorer::math_kernel::types::{Dimension, StepSize};
use serde::{Deserialize, Serialize};

/// Spatial heatmap visualization mode when a counterfactual Turing fork is active.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MorphogenesisHeatmapMode {
    /// Active perturbed activator concentration `u_B(x, y)`.
    #[default]
    ActiveUb,
    /// Side-by-side spatial split `u_A | u_B` separated by a vertical divider seam.
    SplitUaUb,
    /// Pointwise absolute activator difference `|u_B(x, y) - u_A(x, y)|`.
    DifferenceAbs,
}

impl MorphogenesisHeatmapMode {
    /// All selectable spatial heatmap modes in toolbar display order.
    pub const ALL: [Self; 3] = [Self::ActiveUb, Self::SplitUaUb, Self::DifferenceAbs];

    /// Human-readable UI label for segmented mode buttons and accessibility announcements.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::ActiveUb => "Active u_B",
            Self::SplitUaUb => "Split u_A | u_B",
            Self::DifferenceAbs => "Difference |u_B - u_A|",
        }
    }
}

/// Compact serializable fork descriptor for `MorphogenesisTool` (`< 320` bytes JSON).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MorphogenesisForkDescriptor {
    /// Step index `k > 0` where the reaction-diffusion field was forked.
    pub fork_step: usize,
    /// Number of lockstep `dt` steps evolved after `fork_step`.
    #[serde(default)]
    pub post_fork_steps: usize,
    /// Frozen baseline activator production rate `a` on `Branch A`.
    pub baseline_a: f64,
    /// Frozen baseline inhibitor production rate `b` on `Branch A`.
    pub baseline_b: f64,
    /// Frozen baseline activator diffusion `D_u` on `Branch A`.
    pub baseline_du: f64,
    /// Frozen baseline inhibitor diffusion `D_v` on `Branch A`.
    pub baseline_dv: f64,
    /// Spatial activator perturbation amplitude `δu` applied at `t_k` on `Branch B`.
    pub delta_u: f64,
    /// Active spatial heatmap comparison mode.
    #[serde(default)]
    pub heatmap_mode: MorphogenesisHeatmapMode,
}

impl MorphogenesisForkDescriptor {
    /// Validates finiteness and domain bounds of a serialized fork descriptor.
    pub fn validate(&self) -> Result<(), String> {
        let checks = [
            ("baseline_a", self.baseline_a, 0.0, 1.0),
            ("baseline_b", self.baseline_b, 0.0, 2.0),
            ("baseline_du", self.baseline_du, 0.1, 5.0),
            ("baseline_dv", self.baseline_dv, 10.0, 200.0),
            ("delta_u", self.delta_u, 0.0, 1.0),
        ];
        if self.fork_step == 0 || self.fork_step > 20_000 || self.post_fork_steps > 20_000 {
            return Err("Morphogenesis fork steps must be in 1..=20000".to_string());
        }
        for (name, val, min, max) in checks {
            if !val.is_finite() || val < min || val > max {
                return Err(format!(
                    "Invalid morphogenesis fork field '{name}' = {val} (expected [{min}, {max}])"
                ));
            }
        }
        Ok(())
    }
}

/// Structured worker payload carrying `u_B`, optional baseline `u_A`, and twin metrics.
#[derive(Clone, Debug)]
pub struct MorphogenesisSnapshotData {
    /// Active `Branch B` activator field `u_B`.
    pub u_b: Vec<f64>,
    /// Unperturbed `Branch A` activator field `u_A` when forked.
    pub u_a: Option<Vec<f64>>,
    /// Current step index `k`.
    pub step_count: usize,
    /// Live spatial divergence snapshot when forked.
    pub divergence: Option<DivergenceSnapshot>,
}

/// Pre-allocated counterfactual shadow twin state for `MorphogenesisRunner` (zero-alloc `step()`).
pub struct MorphogenesisTwinState {
    /// Frozen baseline `TuringSystem` (`Branch A`) with pre-allocated double buffers.
    pub system_a: TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    /// Cloned baseline PRNG state at `t_k`.
    pub rng_a: SimpleRng,
    /// Step index `k > 0` where the fork was anchored.
    pub fork_step: usize,
    /// Number of lockstep steps taken since `fork_step`.
    pub post_fork_steps: usize,
    /// Spatial activator perturbation amplitude `δu` applied at `t_k`.
    pub delta_u: f64,
    /// Latest spatial Euclidean difference `||u_B - u_A||_2`.
    pub l2_norm: f64,
    /// Latest pointwise supremum difference `||u_B - u_A||_∞`.
    pub linf_norm: f64,
    /// Live separation and logarithmic slope tracker.
    pub tracker: SeparationTracker,
}

/// Clones a 2D `TuringSystem` (pre-allocating both `state` and `next_state` buffers).
pub fn clone_turing_system(
    src: &TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
    width: usize,
    height: usize,
) -> TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D> {
    let mut dst = TuringSystem::new_with_kinetics(
        Dimension(width * height),
        DiffusionCoeff(src.diffusion_coeffs[0]),
        DiffusionCoeff(src.diffusion_coeffs[1]),
        SchnakenbergKinetics {
            a: src.kinetics.a,
            b: src.kinetics.b,
        },
        FiniteDifference2D::new(
            Dimension(width),
            Dimension(height),
            StepSize(1.0),
            StepSize(1.0),
        ),
    );
    dst.u_mut().copy_from_slice(src.u());
    dst.v_mut().copy_from_slice(src.v());
    dst
}

impl MorphogenesisTwinState {
    /// Clones `(fork_step, system_b, rng_b)` into `Branch A` and perturbs `system_b` by `delta_u`.
    pub fn fork_from(
        fork_step: usize,
        dt: f64,
        width: usize,
        height: usize,
        system_b: &mut TuringSystem<2, SchnakenbergKinetics, FiniteDifference2D>,
        rng_b: &mut SimpleRng,
        delta_u: f64,
    ) -> Self {
        let clean_delta = sanitize_perturbation_delta(delta_u);
        let system_a = clone_turing_system(system_b, width, height);
        let rng_a = *rng_b;

        if clean_delta > 0.0 {
            let u_b = system_b.u_mut();
            let center = u_b.len() / 2;
            for (idx, cell) in u_b.iter_mut().enumerate() {
                let jitter = if idx == center {
                    1.0
                } else {
                    rng_b.range(-1.0, 1.0)
                };
                *cell += clean_delta * jitter;
            }
        }

        let (l2_norm, linf_norm) = field_difference_norms(system_a.u(), system_b.u());
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            0.005
        };
        let fork_time = (fork_step as f64) * safe_dt;
        Self {
            system_a,
            rng_a,
            fork_step,
            post_fork_steps: 0,
            delta_u: clean_delta,
            l2_norm,
            linf_norm,
            tracker: SeparationTracker::new(fork_time, l2_norm),
        }
    }

    /// Clones `MorphogenesisTwinState` outside the verified step loop.
    #[must_use]
    pub fn clone_with_dims(&self, width: usize, height: usize) -> Self {
        Self {
            system_a: clone_turing_system(&self.system_a, width, height),
            rng_a: self.rng_a,
            fork_step: self.fork_step,
            post_fork_steps: self.post_fork_steps,
            delta_u: self.delta_u,
            l2_norm: self.l2_norm,
            linf_norm: self.linf_norm,
            tracker: self.tracker,
        }
    }

    /// Steps frozen baseline `Branch A` by `dt` with zero heap allocations.
    #[inline]
    pub fn step_branch_a(&mut self, dt: f64) {
        self.system_a.step(dt);
    }

    /// Updates spatial `L2` and `L_∞` norms after a lockstep tick with zero heap allocations.
    #[inline]
    pub fn observe_lockstep(&mut self, step_count: usize, dt: f64, u_b: &[f64]) {
        self.post_fork_steps = step_count.saturating_sub(self.fork_step);
        let (l2, linf) = field_difference_norms(self.system_a.u(), u_b);
        self.l2_norm = l2;
        self.linf_norm = linf;
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            0.005
        };
        self.tracker.observe((step_count as f64) * safe_dt, l2);
    }

    /// Builds a live [`DivergenceSnapshot`] reporting spatial `L2` and `L_∞` norms.
    #[must_use]
    pub fn snapshot(&self, step_count: usize, dt: f64) -> DivergenceSnapshot {
        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt
        } else {
            0.005
        };
        DivergenceSnapshot {
            fork_step: self.fork_step,
            fork_time: self.tracker.fork_time(),
            current_time: (step_count as f64) * safe_dt,
            delta_x: self.delta_u,
            l2_separation: self.l2_norm,
            log_separation_slope: self.tracker.log_separation_slope(),
            peak_separation: self.tracker.peak_separation(),
            primary_delta: Some(DomainInterventionDelta::new(
                "||u_B - u_A||_∞",
                format!("{:.4e}", self.linf_norm),
                self.linf_norm,
                "Maximum pointwise activator difference ||u_B(x,y) - u_A(x,y)||_∞",
            )),
            secondary_delta: Some(DomainInterventionDelta::new(
                "Spatial L2",
                format!("{:.4e}", self.l2_norm),
                self.l2_norm,
                "Spatial Euclidean activator difference ||u_B - u_A||_2",
            )),
        }
    }

    /// Serializes current twin metadata into a [`MorphogenesisForkDescriptor`].
    #[must_use]
    pub fn to_descriptor(
        &self,
        heatmap_mode: MorphogenesisHeatmapMode,
    ) -> MorphogenesisForkDescriptor {
        MorphogenesisForkDescriptor {
            fork_step: self.fork_step,
            post_fork_steps: self.post_fork_steps,
            baseline_a: self.system_a.kinetics.a,
            baseline_b: self.system_a.kinetics.b,
            baseline_du: self.system_a.diffusion_coeffs[0],
            baseline_dv: self.system_a.diffusion_coeffs[1],
            delta_u: self.delta_u,
            heatmap_mode,
        }
    }
}

fn lerp_u8(a: u8, b: u8, t: f64) -> u8 {
    (f64::from(a) + (f64::from(b) - f64::from(a)) * t.clamp(0.0, 1.0)).round() as u8
}

fn concentration_rgb(norm: f64, theme: ThemeMode) -> [u8; 3] {
    let t = if norm.is_finite() {
        norm.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match theme {
        ThemeMode::Dark => heatmap_color(t),
        ThemeMode::HighContrast => heatmap_color_high_contrast(t),
        ThemeMode::Light => {
            if t < 0.5 {
                let s = t * 2.0;
                [
                    lerp_u8(244, 32, s),
                    lerp_u8(247, 138, s),
                    lerp_u8(252, 156, s),
                ]
            } else {
                let s = (t - 0.5) * 2.0;
                [
                    lerp_u8(32, 18, s),
                    lerp_u8(138, 34, s),
                    lerp_u8(156, 108, s),
                ]
            }
        }
    }
}

fn difference_rgb(norm: f64, theme: ThemeMode) -> [u8; 3] {
    let t = if norm.is_finite() {
        norm.clamp(0.0, 1.0)
    } else {
        0.0
    };
    match theme {
        ThemeMode::Dark => {
            if t < 0.5 {
                let s = t * 2.0;
                [lerp_u8(16, 198, s), lerp_u8(20, 48, s), lerp_u8(32, 108, s)]
            } else {
                let s = (t - 0.5) * 2.0;
                [
                    lerp_u8(198, 255, s),
                    lerp_u8(48, 214, s),
                    lerp_u8(108, 72, s),
                ]
            }
        }
        ThemeMode::Light => {
            if t < 0.5 {
                let s = t * 2.0;
                [
                    lerp_u8(246, 217, s),
                    lerp_u8(248, 95, s),
                    lerp_u8(252, 14, s),
                ]
            } else {
                let s = (t - 0.5) * 2.0;
                [lerp_u8(217, 127, s), lerp_u8(95, 0, s), lerp_u8(14, 38, s)]
            }
        }
        ThemeMode::HighContrast => {
            if t < 0.5 {
                let s = t * 2.0;
                [0, lerp_u8(0, 220, s), lerp_u8(0, 255, s)]
            } else {
                let s = (t - 0.5) * 2.0;
                [lerp_u8(0, 255, s), lerp_u8(220, 255, s), lerp_u8(255, 0, s)]
            }
        }
    }
}

fn finite_bounds(primary: &[f64], secondary: Option<&[f64]>) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in primary.iter().chain(secondary.unwrap_or(&[]).iter()) {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if min.is_finite() && max.is_finite() && max >= min {
        let range = max - min;
        (min, if range > 1e-6 { 1.0 / range } else { 1.0 })
    } else {
        (0.0, 1.0)
    }
}

/// Renders `Active u_B`, `Split u_A | u_B`, or `Difference |u_B - u_A|` across `Dark`, `Light`, and `HighContrast`.
#[must_use]
pub fn render_morphogenesis_heatmap(
    u_b: &[f64],
    u_a: Option<&[f64]>,
    width: usize,
    height: usize,
    mode: MorphogenesisHeatmapMode,
    theme: ThemeMode,
) -> ColorImage {
    let n = width * height;
    let mut pixels = Vec::with_capacity(n);
    let effective_ua = u_a.filter(|a| a.len() == n && u_b.len() == n);

    match (mode, effective_ua) {
        (MorphogenesisHeatmapMode::SplitUaUb, Some(ua)) if width >= 2 => {
            let (min, inv) = finite_bounds(u_b, Some(ua));
            let seam_x = width / 2;
            let seam_rgb = match theme {
                ThemeMode::Dark => [226, 232, 240],
                ThemeMode::Light => [15, 23, 42],
                ThemeMode::HighContrast => [255, 255, 0],
            };
            for idx in 0..n {
                let x = idx % width;
                let rgb = if x == seam_x {
                    seam_rgb
                } else {
                    let val = if x < seam_x { ua[idx] } else { u_b[idx] };
                    let safe = if val.is_finite() { val } else { min };
                    concentration_rgb((safe - min) * inv, theme)
                };
                pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
            }
        }
        (MorphogenesisHeatmapMode::DifferenceAbs, Some(ua)) => {
            let mut max_diff = 0.0_f64;
            for (&a, &b) in ua.iter().zip(u_b.iter()) {
                let d = (b - a).abs();
                if d.is_finite() {
                    max_diff = max_diff.max(d);
                }
            }
            let inv = if max_diff > 1e-12 {
                1.0 / max_diff
            } else {
                0.0
            };
            for (&a, &b) in ua.iter().zip(u_b.iter()) {
                let d = (b - a).abs();
                let rgb = difference_rgb(if d.is_finite() { d * inv } else { 0.0 }, theme);
                pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
            }
        }
        _ => {
            let (min, inv) = finite_bounds(u_b, None);
            for &val in u_b.iter().take(n) {
                let safe = if val.is_finite() { val } else { min };
                let rgb = concentration_rgb((safe - min) * inv, theme);
                pixels.push(Color32::from_rgb(rgb[0], rgb[1], rgb[2]));
            }
            pixels.resize(n, Color32::BLACK);
        }
    }

    ColorImage {
        size: [width, height],
        pixels,
        ..Default::default()
    }
}
