#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation 2D parameter regime cartographer (`OdeRegimeCartographer`) for [`OdeSolverTool`].

#[path = "atlas/kernel.rs"]
mod kernel;
#[path = "atlas/panel.rs"]
mod panel;

use super::OdeSolverTool;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, RegimeCartographer,
};
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::presets::OdePreset;
use eframe::egui;
use kernel::{classify_stack_dynamics, StackOdeDynamics};
use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::ode::{ArrayState, IntegrationMethod};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;

const REGIME_COUNT: u8 = 4;

/// Supported `(x, y)` parameter axis pairs for [`OdeRegimeCartographer`].
pub const ODE_ATLAS_AXIS_PAIRS: &[(&str, &str)] = &[
    ("k", "c"),
    ("k", "dt"),
    ("y0", "v0"),
    ("r", "cap_k"),
    ("mu", "dt"),
    ("a", "b"),
    ("f", "w"),
];

/// Default serialized atlas axis keys for [`OdeSolverTool`].
#[must_use]
pub fn default_ode_atlas_axes() -> [String; 2] {
    ["k".to_string(), "c".to_string()]
}

/// Predicate omitting default `atlas_axes` during JSON serialization for backward compatibility.
#[must_use]
pub fn is_default_ode_atlas_axes(axes: &[String; 2]) -> bool {
    axes[0] == "k" && axes[1] == "c"
}

/// Discrete dynamical and numerical stability regimes classified by [`OdeRegimeCartographer`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum OdeAtlasRegime {
    /// Monotonic / overdamped decay toward equilibrium ($c^2 \ge 4k$ or $\lambda \in \mathbb{R}_{<0}$).
    OverdampedDecay = 0,
    /// Oscillatory / underdamped spiral toward equilibrium ($0 < c^2 < 4k$).
    UnderdampedSpiral = 1,
    /// Sustained periodic orbit or nonlinear limit cycle ($c = 0$, Van der Pol, or forced cycle).
    SustainedCycle = 2,
    /// Unbounded exponential growth or numerical solver divergence.
    UnstableDivergence = 3,
}

impl OdeAtlasRegime {
    /// Converts a raw `u8` regime index into [`Self`].
    #[inline]
    #[must_use]
    pub const fn from_u8(code: u8) -> Self {
        match code {
            0 => Self::OverdampedDecay,
            1 => Self::UnderdampedSpiral,
            2 => Self::SustainedCycle,
            _ => Self::UnstableDivergence,
        }
    }

    /// Human-readable regime classification label.
    #[inline]
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::OverdampedDecay => "Overdamped / Monotonic Decay",
            Self::UnderdampedSpiral => "Underdamped Spiral (c² < 4k)",
            Self::SustainedCycle => "Sustained Orbit / Limit Cycle",
            Self::UnstableDivergence => "Unstable / Divergent",
        }
    }
}

fn ode_axis_spec(
    theory_params: &HashMap<String, ParameterConstraint>,
    key: &'static str,
) -> AtlasAxisSpec {
    let pc = |min: f64, max: f64, step: f64| ParameterConstraint { min, max, step };
    let (label, fallback) = match key {
        "c" => ("Damping (c)", pc(-0.5, 5.0, 0.05)),
        "dt" => ("Step Size (dt)", pc(0.005, 0.5, 0.005)),
        "y0" => ("Initial y(0)", pc(-4.0, 4.0, 0.1)),
        "v0" => ("Initial v(0)", pc(-4.0, 4.0, 0.1)),
        "r" => ("Growth Rate (r)", pc(-1.0, 5.0, 0.05)),
        "cap_k" => ("Carrying Capacity (K)", pc(1.0, 50.0, 0.5)),
        "mu" => ("Nonlinearity (μ)", pc(-0.5, 5.0, 0.05)),
        "a" => ("Parameter (a)", pc(-3.0, 5.0, 0.05)),
        "b" => ("Parameter (b)", pc(-2.0, 5.0, 0.05)),
        "f" => ("Forcing Amplitude (f)", pc(0.0, 5.0, 0.05)),
        "w" => ("Forcing Frequency (ω)", pc(0.1, 5.0, 0.05)),
        _ => ("Stiffness / Rate (k)", pc(0.05, 5.0, 0.05)),
    };
    AtlasAxisSpec::from_theory(theory_params, key, label, fallback)
}

/// Zero-allocation 2D parameter-space cartographer for [`OdeSolverTool`].
#[derive(Clone, Debug)]
pub struct OdeRegimeCartographer {
    axes: AtlasAxisPair,
    preset: OdePreset,
    method: IntegrationMethod,
    custom_coupled: bool,
    base_dt: f64,
    base_k: f64,
    base_c: f64,
    base_r: f64,
    base_cap_k: f64,
    base_y0: f64,
    base_v0: f64,
    matrix_a: [[f64; 2]; 2],
    planar_dx: Option<CompiledExpr>,
    planar_dy: Option<CompiledExpr>,
    planar_params: [f64; MAX_SYNTHESIZED_PARAMS],
    planar_len: usize,
    planar_slot_x: Option<usize>,
    planar_slot_y: Option<usize>,
    scalar_expr: Option<CompiledExpr>,
    scalar_params: [f64; MAX_SYNTHESIZED_PARAMS],
    scalar_len: usize,
    scalar_slot_x: Option<usize>,
    scalar_slot_y: Option<usize>,
    coupled_dy: Option<CompiledExpr>,
    coupled_dv: Option<CompiledExpr>,
    coupled_params: [f64; MAX_SYNTHESIZED_PARAMS],
    coupled_len: usize,
    coupled_slot_x: Option<usize>,
    coupled_slot_y: Option<usize>,
}

pub(super) fn find_param_slot(names: &[String], len: usize, target: &str) -> Option<usize> {
    names
        .iter()
        .take(len)
        .position(|n| n == target || (target == "cap_k" && n == "K"))
}

fn apply_axis_slots(
    mut base: [f64; MAX_SYNTHESIZED_PARAMS],
    slots: (Option<usize>, Option<usize>),
    vals: (f64, f64),
) -> [f64; MAX_SYNTHESIZED_PARAMS] {
    if let Some(sx) = slots.0 {
        base[sx] = vals.0;
    }
    if let Some(sy) = slots.1 {
        base[sy] = vals.1;
    }
    base
}

struct PlanarCartographerSnapshot {
    dx: Option<CompiledExpr>,
    dy: Option<CompiledExpr>,
    params: [f64; MAX_SYNTHESIZED_PARAMS],
    len: usize,
    slot_x: Option<usize>,
    slot_y: Option<usize>,
    base_c: f64,
}

fn snapshot_planar_expr(
    tool: &OdeSolverTool,
    axis_keys: (&str, &str),
) -> PlanarCartographerSnapshot {
    let (dx, dy, params, len) = tool.planar_bank.snapshot_compiled_and_params(
        &tool.planar_x_input,
        &tool.planar_y_input,
        [tool.param_k, tool.param_r, tool.param_cap_k],
        tool.synced_legacy_params,
    );
    let slot_x = find_param_slot(dx.param_names(), len, axis_keys.0);
    let slot_y = find_param_slot(dx.param_names(), len, axis_keys.1);
    let base_c = find_param_slot(dx.param_names(), len, "c").map_or(0.0, |idx| params[idx]);
    PlanarCartographerSnapshot {
        dx: Some(dx),
        dy: Some(dy),
        params,
        len,
        slot_x,
        slot_y,
        base_c,
    }
}

fn snapshot_scalar_expr(
    tool: &OdeSolverTool,
    axis_keys: (&str, &str),
) -> (
    Option<CompiledExpr>,
    [f64; MAX_SYNTHESIZED_PARAMS],
    usize,
    Option<usize>,
    Option<usize>,
) {
    let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
    let expr = tool.math_input.active_compiled().cloned();
    let Some(compiled) = &expr else {
        return (None, params, 0, None, None);
    };
    let slice = tool.math_input.param_slice();
    let len = compiled
        .param_names()
        .len()
        .min(MAX_SYNTHESIZED_PARAMS)
        .min(slice.len());
    params[..len].copy_from_slice(&slice[..len]);
    let sx = find_param_slot(compiled.param_names(), len, axis_keys.0);
    let sy = find_param_slot(compiled.param_names(), len, axis_keys.1);
    (expr, params, len, sx, sy)
}

impl OdeRegimeCartographer {
    /// Snapshots [`OdeSolverTool`] state into a thread-safe zero-allocation [`OdeRegimeCartographer`].
    #[must_use]
    pub fn from_tool(tool: &OdeSolverTool, axis_keys: (&'static str, &'static str)) -> Self {
        let theory_params = tool.theory_parameters();
        let axes = AtlasAxisPair {
            x: ode_axis_spec(&theory_params, axis_keys.0),
            y: ode_axis_spec(&theory_params, axis_keys.1),
        };
        let (scalar_expr, scalar_params, scalar_len, scalar_slot_x, scalar_slot_y) =
            snapshot_scalar_expr(tool, axis_keys);
        let planar = snapshot_planar_expr(tool, axis_keys);
        let compiled = tool.coupled_bank.active_compiled();
        let c_slice = tool.coupled_bank.param_slice();
        let coupled_len = c_slice.len().min(MAX_SYNTHESIZED_PARAMS);
        let mut coupled_params = [0.0; MAX_SYNTHESIZED_PARAMS];
        coupled_params[..coupled_len].copy_from_slice(&c_slice[..coupled_len]);
        let names = tool.coupled_bank.param_names();
        let base_c = if tool.preset == OdePreset::PlanarCustom {
            planar.base_c
        } else {
            find_param_slot(names, coupled_len, "c").map_or(0.0, |idx| coupled_params[idx])
        };

        Self {
            axes,
            preset: tool.preset,
            method: tool.method,
            custom_coupled: tool.custom_coupled,
            base_dt: tool.dt,
            base_k: tool.param_k,
            base_c,
            base_r: tool.param_r,
            base_cap_k: tool.param_cap_k,
            base_y0: tool.ic_y0,
            base_v0: tool.ic_v0,
            matrix_a: tool.matrix_a,
            planar_dx: planar.dx,
            planar_dy: planar.dy,
            planar_params: planar.params,
            planar_len: planar.len,
            planar_slot_x: planar.slot_x,
            planar_slot_y: planar.slot_y,
            scalar_expr,
            scalar_params,
            scalar_len,
            scalar_slot_x,
            scalar_slot_y,
            coupled_dy: compiled.first().cloned(),
            coupled_dv: compiled.get(1).cloned(),
            coupled_params,
            coupled_len,
            coupled_slot_x: find_param_slot(names, coupled_len, axis_keys.0),
            coupled_slot_y: find_param_slot(names, coupled_len, axis_keys.1),
        }
    }

    fn resolve_custom_dynamics(&self, p1: f64, p2: f64, k: f64, c: f64) -> StackOdeDynamics<'_> {
        if self.custom_coupled {
            if let (Some(dy), Some(dv)) = (&self.coupled_dy, &self.coupled_dv) {
                let params = apply_axis_slots(
                    self.coupled_params,
                    (self.coupled_slot_x, self.coupled_slot_y),
                    (p1, p2),
                );
                return StackOdeDynamics::CoupledExpr {
                    dy,
                    dv,
                    params,
                    param_len: self.coupled_len,
                };
            }
            StackOdeDynamics::DampedHarmonic { k, c }
        } else if let Some(expr) = &self.scalar_expr {
            let params = apply_axis_slots(
                self.scalar_params,
                (self.scalar_slot_x, self.scalar_slot_y),
                (p1, p2),
            );
            StackOdeDynamics::ScalarExpr {
                expr,
                params,
                param_len: self.scalar_len,
            }
        } else {
            StackOdeDynamics::Exponential { k }
        }
    }

    fn resolve_dynamics(&self, p1: f64, p2: f64) -> (StackOdeDynamics<'_>, ArrayState<2>, f64) {
        let (mut k, mut c, mut r, mut cap_k) =
            (self.base_k, self.base_c, self.base_r, self.base_cap_k);
        let (mut y0, mut v0, mut dt) = (self.base_y0, self.base_v0, self.base_dt.clamp(0.01, 0.08));
        for (key, val) in [(self.axes.x.key, p1), (self.axes.y.key, p2)] {
            match key {
                "k" => k = val,
                "c" => c = val,
                "r" => r = val,
                "cap_k" => cap_k = val,
                "y0" => y0 = val,
                "v0" => v0 = val,
                "dt" => dt = val,
                _ => {}
            }
        }
        let dyn_sys = match self.preset {
            OdePreset::Exponential => StackOdeDynamics::Exponential { k },
            OdePreset::HarmonicOscillator => StackOdeDynamics::DampedHarmonic { k, c },
            OdePreset::LogisticGrowth => StackOdeDynamics::Logistic { r, cap_k },
            OdePreset::LinearSystem2D => StackOdeDynamics::Linear2D {
                matrix_a: self.matrix_a,
            },
            OdePreset::PlanarCustom => {
                if let (Some(dy), Some(dv)) = (&self.planar_dx, &self.planar_dy) {
                    let params = apply_axis_slots(
                        self.planar_params,
                        (self.planar_slot_x, self.planar_slot_y),
                        (p1, p2),
                    );
                    StackOdeDynamics::CoupledExpr {
                        dy,
                        dv,
                        params,
                        param_len: self.planar_len,
                    }
                } else {
                    StackOdeDynamics::DampedHarmonic { k, c }
                }
            }
            OdePreset::Custom => self.resolve_custom_dynamics(p1, p2, k, c),
        };
        let init_v = if dyn_sys.is_2d() { v0 } else { 0.0 };
        (dyn_sys, ArrayState([y0, init_v]), dt)
    }
}

impl RegimeCartographer for OdeRegimeCartographer {
    fn axes(&self) -> AtlasAxisPair {
        self.axes.clone()
    }

    fn evaluate_cell(&self, p1: f64, p2: f64) -> AtlasCell {
        let (dyn_sys, init, dt) = self.resolve_dynamics(p1, p2);
        classify_stack_dynamics(dyn_sys, self.method, init, dt)
    }

    fn color_for_cell(&self, cell: AtlasCell, high_contrast: bool) -> egui::Color32 {
        atlas_regime_color(cell.regime, REGIME_COUNT, cell.primary, high_contrast)
    }

    fn regime_label(&self, regime: u8) -> &'static str {
        OdeAtlasRegime::from_u8(regime).label()
    }

    fn primary_metric_label(&self) -> &'static str {
        "Normalized Orbit Envelope"
    }

    fn format_hover_tooltip(&self, p1: f64, p2: f64, cell: AtlasCell) -> String {
        let disc_txt = if cell.secondary.is_finite() {
            format!("{:.3}", cell.secondary)
        } else {
            "∞".to_string()
        };
        format!(
            "{} = {:.3}, {} = {:.3}\nRegime: {}\nEnvelope: {:.2}  |  Discriminant/Metric: {}",
            self.axes.x.key,
            p1,
            self.axes.y.key,
            p2,
            self.regime_label(cell.regime),
            cell.primary,
            disc_txt
        )
    }

    fn overlay_contour(&self) -> Option<(&'static str, Vec<[f64; 2]>)> {
        if self.axes.x.key == "k" && self.axes.y.key == "c" {
            let k_min = self.axes.x.constraint.min.max(0.0);
            let k_max = self.axes.x.constraint.max;
            if k_max > k_min {
                let samples = 64_usize;
                let mut pts = Vec::with_capacity(samples + 1);
                for i in 0..=samples {
                    let t = (i as f64) / (samples as f64);
                    let k = k_min + t * (k_max - k_min);
                    let c = 2.0 * k.sqrt();
                    if c >= self.axes.y.constraint.min && c <= self.axes.y.constraint.max {
                        pts.push([k, c]);
                    }
                }
                if pts.len() >= 2 {
                    return Some(("Critical Damping (c² = 4k)", pts));
                }
            }
        }
        None
    }

    fn legend_entries(&self, high_contrast: bool) -> Vec<(&'static str, egui::Color32)> {
        vec![
            (
                OdeAtlasRegime::OverdampedDecay.label(),
                atlas_regime_color(0, REGIME_COUNT, 0.5, high_contrast),
            ),
            (
                OdeAtlasRegime::UnderdampedSpiral.label(),
                atlas_regime_color(1, REGIME_COUNT, 0.5, high_contrast),
            ),
            (
                OdeAtlasRegime::SustainedCycle.label(),
                atlas_regime_color(2, REGIME_COUNT, 0.5, high_contrast),
            ),
            (
                OdeAtlasRegime::UnstableDivergence.label(),
                atlas_regime_color(3, REGIME_COUNT, 0.5, high_contrast),
            ),
        ]
    }
}
