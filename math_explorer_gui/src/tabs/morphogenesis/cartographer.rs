//! Progressive 2D parameter atlas cartographer and washout re-seeding for [`super::MorphogenesisTool`].

#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::atlas::{classify_gray_scott, GrayScottRegime};
use super::state::{MorphogenesisKinetics, DEFAULT_GRAY_SCOTT_F, DEFAULT_GRAY_SCOTT_K};
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use eframe::egui;
use math_explorer::biology::morphogenesis::FusedEulerSolver;
use scientific_metadata::theory::ParameterConstraint;
use std::collections::HashMap;

/// Washout variance threshold below which activator grid `u` is treated as spatially uniform.
pub const WASHOUT_VARIANCE_THRESHOLD: f64 = 1e-8;

/// Supported `(x, y)` parameter axis pairs for [`MorphogenesisCartographer`].
pub const MORPHO_ATLAS_AXIS_PAIRS: &[(&str, &str)] =
    &[("a", "b"), ("f", "k"), ("d_v", "dt"), ("d_u", "d_v")];

/// Discrete reaction-diffusion regimes on the 2D 5-point stencil.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MorphogenesisRegime {
    /// Homogeneous steady state is linearly and numerically stable.
    StableWashout = 0,
    /// Diffusion-driven Turing or saddle-node/subcritical pattern instability.
    TuringPattern = 1,
    /// Homogeneous $k = 0$ Hopf oscillatory instability.
    BulkHopfOscillation = 2,
    /// Explicit Forward Euler amplifies damped grid modes ($|1 + \Delta t \lambda| > 1$).
    EulerCflUnstable = 3,
}

impl MorphogenesisRegime {
    /// Converts a raw `u8` cell regime code into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::StableWashout,
            1 => Self::TuringPattern,
            2 => Self::BulkHopfOscillation,
            _ => Self::EulerCflUnstable,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::StableWashout => "Stable Washout",
            Self::TuringPattern => "Turing Pattern",
            Self::BulkHopfOscillation => "Bulk Hopf Oscillation",
            Self::EulerCflUnstable => "Euler CFL Unstable",
        }
    }
}

/// Default serialized atlas axis keys for [`super::MorphogenesisToolState`].
#[must_use]
pub fn default_morpho_atlas_axes() -> [String; 2] {
    ["a".to_string(), "b".to_string()]
}

/// Predicate skipping default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_morpho_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "a" && axes[1] == "b"
}

/// Computes the population variance of `u` in two zero-allocation scalar passes.
#[must_use]
pub fn grid_u_variance(u: &[f64]) -> f64 {
    if u.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for &val in u {
        if !val.is_finite() {
            return f64::INFINITY;
        }
        sum += val;
        count += 1;
    }
    if count == 0 {
        return 0.0;
    }
    let mean = sum / (count as f64);
    let mut sq_sum = 0.0;
    for &val in u {
        let diff = val - mean;
        sq_sum += diff * diff;
    }
    sq_sum / (count as f64)
}

#[inline]
fn mode_growth_and_euler_amp(jac: [f64; 4], diff: [f64; 2], dt: f64, mu: f64) -> (f64, f64, bool) {
    let [fu, fv, gu, gv] = jac;
    let [d_u, d_v] = diff;
    let tr = (fu - mu * d_u) + (gv - mu * d_v);
    let det = (fu - mu * d_u) * (gv - mu * d_v) - fv * gu;
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        let root = disc.sqrt();
        let lam_plus = 0.5 * (tr + root);
        let lam_minus = 0.5 * (tr - root);
        let amp_plus = (1.0 + dt * lam_plus).abs();
        let amp_minus = (1.0 + dt * lam_minus).abs();
        let cfl_bad = (lam_minus <= 0.0 && 1.0 + dt * lam_minus < -1.0)
            || (lam_plus <= 0.0 && 1.0 + dt * lam_plus < -1.0);
        (lam_plus, amp_plus.max(amp_minus), cfl_bad)
    } else {
        let re = 0.5 * tr;
        let amp_sq = (1.0 + dt * tr + dt * dt * det).max(0.0);
        let cfl_bad = tr <= 0.0 && (tr + dt * det) > 0.0;
        (re, amp_sq.sqrt(), cfl_bad)
    }
}

#[inline]
fn dominant_washout_mode(g0: f64, gs: f64, g8: f64, mu_star: f64) -> (f64, f64) {
    if gs >= g0 && gs >= g8 {
        (gs, mu_star)
    } else if g0 >= g8 {
        (g0, 0.0)
    } else {
        (g8, 8.0)
    }
}

#[inline]
fn valid_positive_diffusion_and_dt(x: f64, y: f64, d_u: f64, d_v: f64, dt: f64) -> bool {
    [x, y, d_u, d_v, dt].into_iter().all(f64::is_finite) && d_u > 0.0 && d_v > 0.0 && dt > 0.0
}

/// Zero-allocation Schnakenberg dispersion and 2D Forward Euler CFL classifier.
#[must_use]
pub fn classify_schnakenberg_regime(a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) -> AtlasCell {
    if !valid_positive_diffusion_and_dt(a, b, d_u, d_v, dt) {
        return AtlasCell {
            regime: MorphogenesisRegime::EulerCflUnstable as u8,
            primary: f64::INFINITY,
            secondary: 8.0,
        };
    }
    let u_eq = a + b;
    if u_eq <= 1e-6 {
        let washout = b <= 1e-9;
        return AtlasCell {
            regime: (if washout {
                MorphogenesisRegime::StableWashout
            } else {
                MorphogenesisRegime::EulerCflUnstable
            }) as u8,
            primary: if washout { -1.0 } else { 2.0 },
            secondary: 0.0,
        };
    }
    let u_sq = u_eq * u_eq;
    let (fu, fv, gu, gv) = ((b - a) / u_eq, u_sq, -2.0 * b / u_eq, -u_sq);
    let jac = [fu, fv, gu, gv];
    let diff = [d_u, d_v];
    let mu_star = ((d_v * fu + d_u * gv) / (2.0 * d_u * d_v)).clamp(0.0, 8.0);
    let (g0, a0, bad0) = mode_growth_and_euler_amp(jac, diff, dt, 0.0);
    let (gs, as_, bads) = mode_growth_and_euler_amp(jac, diff, dt, mu_star);
    let (g8, a8, bad8) = mode_growth_and_euler_amp(jac, diff, dt, 8.0);

    if bad0 || bads || bad8 {
        AtlasCell {
            regime: MorphogenesisRegime::EulerCflUnstable as u8,
            primary: a0.max(as_).max(a8),
            secondary: 8.0,
        }
    } else if fu + gv > 0.0 {
        AtlasCell {
            regime: MorphogenesisRegime::BulkHopfOscillation as u8,
            primary: g0.max(gs),
            secondary: 0.0,
        }
    } else if mu_star > 0.0 && gs > 0.0 {
        AtlasCell {
            regime: MorphogenesisRegime::TuringPattern as u8,
            primary: gs,
            secondary: mu_star,
        }
    } else {
        let (dom_g, dom_mu) = dominant_washout_mode(g0, gs, g8, mu_star);
        AtlasCell {
            regime: MorphogenesisRegime::StableWashout as u8,
            primary: dom_g,
            secondary: dom_mu,
        }
    }
}

fn classify_gray_scott_cell(f: f64, k: f64, d_u: f64, d_v: f64, dt: f64) -> AtlasCell {
    if !valid_positive_diffusion_and_dt(f, k, d_u, d_v, dt) {
        return AtlasCell {
            regime: MorphogenesisRegime::EulerCflUnstable as u8,
            primary: f64::INFINITY,
            secondary: 8.0,
        };
    }
    let amp = (1.0 - 8.0 * d_u.max(d_v) * dt).abs();
    if amp > 1.0 {
        return AtlasCell {
            regime: MorphogenesisRegime::EulerCflUnstable as u8,
            primary: amp,
            secondary: 8.0,
        };
    }
    match classify_gray_scott(f, k, d_u, d_v) {
        GrayScottRegime::BifurcationUnstable => AtlasCell {
            regime: MorphogenesisRegime::BulkHopfOscillation as u8,
            primary: (f - 4.0 * (f + k).powi(2)).max(0.05),
            secondary: 1.0,
        },
        GrayScottRegime::SubcriticalPattern => AtlasCell {
            regime: MorphogenesisRegime::TuringPattern as u8,
            primary: 0.15,
            secondary: 2.0,
        },
        GrayScottRegime::UniformStable => AtlasCell {
            regime: MorphogenesisRegime::StableWashout as u8,
            primary: -(f + k),
            secondary: 0.0,
        },
    }
}

/// Returns `true` when `(a, b, d_u, d_v, dt)` lies in a CFL-safe pattern-forming or oscillatory regime.
#[inline]
#[must_use]
pub fn is_pattern_forming_regime(a: f64, b: f64, d_u: f64, d_v: f64, dt: f64) -> bool {
    matches!(
        MorphogenesisRegime::from_u8(classify_schnakenberg_regime(a, b, d_u, d_v, dt).regime),
        MorphogenesisRegime::TuringPattern | MorphogenesisRegime::BulkHopfOscillation
    )
}

fn morpho_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let (label, min, max, step) = match key {
        "a" => ("Activator Rate (a)", 0.0, 1.0, 0.01),
        "b" => ("Inhibitor Rate (b)", 0.0, 2.0, 0.01),
        "f" => ("Feed Rate (F)", 0.005, 0.085, 0.001),
        "k" => ("Kill Rate (k)", 0.030, 0.070, 0.001),
        "d_u" => ("Activator Diffusion (D_u)", 0.1, 5.0, 0.05),
        "d_v" => ("Inhibitor Diffusion (D_v)", 10.0, 200.0, 1.0),
        _ => ("Time Step (dt)", 1e-4, 0.1, 5e-4),
    };
    let resolved_key = if matches!(key, "a" | "b" | "f" | "k" | "d_u" | "d_v" | "dt") {
        key
    } else {
        "dt"
    };
    AtlasAxisSpec::from_theory(
        theory_params,
        resolved_key,
        label,
        ParameterConstraint { min, max, step },
    )
}

/// Zero-allocation 2D cartographer for Schnakenberg and Gray-Scott reaction-diffusion morphogenesis.
#[derive(Clone, Debug)]
pub struct MorphogenesisCartographer {
    axes: AtlasAxisPair,
    kinetics: MorphogenesisKinetics,
    base_a: f64,
    base_b: f64,
    base_f: f64,
    base_k: f64,
    base_du: f64,
    base_dv: f64,
    base_dt: f64,
    auto_dt: bool,
}

impl MorphogenesisCartographer {
    /// Builds a [`MorphogenesisCartographer`] bounded by [`super::presets::MorphogenesisTheory`].
    #[must_use]
    pub fn new(
        theory_params: &HashMap<String, ParameterConstraint>,
        axis_keys: (&'static str, &'static str),
        a: f64,
        b: f64,
        d_u: f64,
        d_v: f64,
        dt: f64,
    ) -> Self {
        Self {
            axes: AtlasAxisPair {
                x: morpho_axis_spec(theory_params, axis_keys.0),
                y: morpho_axis_spec(theory_params, axis_keys.1),
            },
            kinetics: MorphogenesisKinetics::Schnakenberg,
            base_a: a,
            base_b: b,
            base_f: DEFAULT_GRAY_SCOTT_F,
            base_k: DEFAULT_GRAY_SCOTT_K,
            base_du: d_u,
            base_dv: d_v,
            base_dt: dt,
            auto_dt: false,
        }
    }

    /// Attaches active [`MorphogenesisKinetics`], Gray-Scott `(f, k)`, and `auto_dt` context.
    #[must_use]
    pub const fn with_kinetics(
        mut self,
        kinetics: MorphogenesisKinetics,
        f: f64,
        k: f64,
        auto_dt: bool,
    ) -> Self {
        self.kinetics = kinetics;
        self.base_f = f;
        self.base_k = k;
        self.auto_dt = auto_dt;
        self
    }

    #[inline]
    fn resolve_params(&self, p1: f64, p2: f64) -> (f64, f64, f64, f64, f64, f64, f64) {
        let (mut a, mut b, mut f, mut k) = (self.base_a, self.base_b, self.base_f, self.base_k);
        let (mut du, mut dv, mut dt) = (self.base_du, self.base_dv, self.base_dt);
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "a" => a = val,
                "b" => b = val,
                "f" => f = val,
                "k" => k = val,
                "d_u" => du = val,
                "d_v" => dv = val,
                "dt" => dt = val,
                _ => {}
            }
        }
        if self.auto_dt && self.axes.x.key != "dt" && self.axes.y.key != "dt" {
            dt =
                FusedEulerSolver::cfl_substeps_2d(dt, 1.0, du, dv, super::runner::MAX_CFL_SUBSTEPS)
                    .1;
        }
        (a, b, f, k, du, dv, dt)
    }
}

impl RegimeCartographer for MorphogenesisCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (a, b, f, k, du, dv, dt) = self.resolve_params(p1, p2);
        let use_gs = self.axes.x.key == "f"
            || self.axes.y.key == "k"
            || (self.kinetics == MorphogenesisKinetics::GrayScott
                && self.axes.x.key != "a"
                && self.axes.y.key != "b");
        if use_gs {
            classify_gray_scott_cell(f, k, du, dv, dt)
        } else {
            classify_schnakenberg_regime(a, b, du, dv, dt)
        }
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        let intensity = match MorphogenesisRegime::from_u8(cell.regime) {
            MorphogenesisRegime::StableWashout => {
                (1.0 / (1.0 - cell.primary).max(1.0)).clamp(0.0, 1.0)
            }
            MorphogenesisRegime::TuringPattern => (cell.primary * 4.0).clamp(0.15, 1.0),
            MorphogenesisRegime::BulkHopfOscillation => (cell.primary * 2.0).clamp(0.2, 1.0),
            MorphogenesisRegime::EulerCflUnstable => ((cell.primary - 1.0) * 0.25).clamp(0.2, 1.0),
        };
        atlas_regime_color(cell.regime, 4, intensity, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        MorphogenesisRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Dominant Growth Rate Re(λ)"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let reg = MorphogenesisRegime::from_u8(cell.regime);
        if reg == MorphogenesisRegime::EulerCflUnstable {
            format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nEuler Amplification |G|: {:.3} (> 1.0 unstable)",
                self.axes.x.key, p1, self.axes.y.key, p2, reg.label(), cell.primary
            )
        } else {
            format!(
                "{} = {:.4}, {} = {:.4}\nRegime: {}\nDominant Growth Re(λ): {:+.4} (k² = {:.3})",
                self.axes.x.key,
                p1,
                self.axes.y.key,
                p2,
                reg.label(),
                cell.primary,
                cell.secondary
            )
        }
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        [
            MorphogenesisRegime::StableWashout,
            MorphogenesisRegime::TuringPattern,
            MorphogenesisRegime::BulkHopfOscillation,
            MorphogenesisRegime::EulerCflUnstable,
        ]
        .into_iter()
        .map(|r| {
            (
                r.label(),
                atlas_regime_color(r as u8, 4, 0.7, high_contrast),
            )
        })
        .collect()
    }
}
