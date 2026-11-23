#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Zero-allocation 2D parameter regime cartographer (`OdeRegimeCartographer`) for [`OdeSolverTool`].

#[path = "atlas/kernel.rs"]
mod kernel;

use super::OdeSolverTool;
use crate::async_sim::atlas::{
    atlas_regime_color, AtlasAxisPair, AtlasAxisSpec, AtlasCell, AtlasTeleportEvent,
    RegimeCartographer,
};
use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::presets::OdePreset;
use eframe::egui;
use kernel::{classify_stack_dynamics, StackOdeDynamics};
use math_commons::ast::CompiledExpr;
use math_explorer::pure_math::analysis::ode::{ArrayState, IntegrationMethod};
use scientific_metadata::theory::{ParameterConstraint, TheoryDescribable};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

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

fn find_param_slot(names: &[String], len: usize, target: &str) -> Option<usize> {
    names.iter().take(len).position(|n| n == target)
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
        let mut scalar_params = [0.0; MAX_SYNTHESIZED_PARAMS];
        let mut scalar_len = 0;
        let (mut scalar_slot_x, mut scalar_slot_y) = (None, None);
        let scalar_expr = tool.math_input.active_compiled().cloned();
        if let Some(expr) = &scalar_expr {
            let slice = tool.math_input.param_slice();
            scalar_len = expr
                .param_names()
                .len()
                .min(MAX_SYNTHESIZED_PARAMS)
                .min(slice.len());
            scalar_params[..scalar_len].copy_from_slice(&slice[..scalar_len]);
            scalar_slot_x = find_param_slot(expr.param_names(), scalar_len, axis_keys.0);
            scalar_slot_y = find_param_slot(expr.param_names(), scalar_len, axis_keys.1);
        }

        let compiled = tool.coupled_bank.active_compiled();
        let c_slice = tool.coupled_bank.param_slice();
        let coupled_len = c_slice.len().min(MAX_SYNTHESIZED_PARAMS);
        let mut coupled_params = [0.0; MAX_SYNTHESIZED_PARAMS];
        coupled_params[..coupled_len].copy_from_slice(&c_slice[..coupled_len]);
        let names = tool.coupled_bank.param_names();
        let base_c =
            find_param_slot(names, coupled_len, "c").map_or(0.0, |idx| coupled_params[idx]);

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
            OdePreset::Custom if self.custom_coupled => {
                if let (Some(dy), Some(dv)) = (&self.coupled_dy, &self.coupled_dv) {
                    let mut params = self.coupled_params;
                    if let Some(sx) = self.coupled_slot_x {
                        params[sx] = p1;
                    }
                    if let Some(sy) = self.coupled_slot_y {
                        params[sy] = p2;
                    }
                    StackOdeDynamics::CoupledExpr {
                        dy,
                        dv,
                        params,
                        param_len: self.coupled_len,
                    }
                } else {
                    StackOdeDynamics::DampedHarmonic { k, c }
                }
            }
            OdePreset::Custom => {
                if let Some(expr) = &self.scalar_expr {
                    let mut params = self.scalar_params;
                    if let Some(sx) = self.scalar_slot_x {
                        params[sx] = p1;
                    }
                    if let Some(sy) = self.scalar_slot_y {
                        params[sy] = p2;
                    }
                    StackOdeDynamics::ScalarExpr {
                        expr,
                        params,
                        param_len: self.scalar_len,
                    }
                } else {
                    StackOdeDynamics::Exponential { k }
                }
            }
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

impl OdeSolverTool {
    pub(crate) fn atlas_param_value(&self, key: &str) -> f64 {
        match key {
            "k" => self.param_k,
            "r" => self.param_r,
            "cap_k" => self.param_cap_k,
            "y0" => self.ic_y0,
            "v0" => self.ic_v0,
            "dt" => self.dt,
            _ => {
                if self.custom_coupled {
                    if let Some(pos) = self
                        .coupled_bank
                        .param_names()
                        .iter()
                        .position(|n| n == key)
                    {
                        return self
                            .coupled_bank
                            .param_slice()
                            .get(pos)
                            .copied()
                            .unwrap_or(1.0);
                    }
                }
                self.math_input.active_param_value(key).unwrap_or(0.2)
            }
        }
    }

    fn apply_single_atlas_key(&mut self, key: &str, val: f64) {
        match key {
            "k" => {
                self.param_k = val;
                self.math_input.set_param("k", val);
                self.coupled_bank.set_param("k", val);
            }
            "c" => {
                if self.preset == OdePreset::HarmonicOscillator && val.abs() > 1e-6 {
                    self.preset = OdePreset::Custom;
                    self.custom_coupled = true;
                    self.coupled_bank
                        .set_expressions(&["v".to_string(), "-k * y - c * v".to_string()]);
                    self.coupled_bank.set_param("k", self.param_k);
                }
                self.coupled_bank.set_param("c", val);
            }
            "r" => {
                self.param_r = val;
                self.math_input.set_param("r", val);
                self.coupled_bank.set_param("r", val);
            }
            "cap_k" => {
                self.param_cap_k = val;
                self.math_input.set_param("cap_k", val);
                self.math_input.set_param("K", val);
            }
            "y0" => self.ic_y0 = val,
            "v0" => self.ic_v0 = val,
            "dt" => {
                let min_dt =
                    (self.total_time.max(0.1) / crate::presets::MAX_ODE_STEPS as f64).max(1e-4);
                self.dt = val.clamp(min_dt, 1.0);
            }
            _ => {
                self.coupled_bank.set_param(key, val);
                self.math_input.set_param(key, val);
            }
        }
    }

    /// Teleports active ODE parameters to `ev`, pushing the prior run to `snapshot_manager`.
    pub fn apply_atlas_teleport(&mut self, ev: AtlasTeleportEvent) {
        let prev_summary = self.last_param_summary.clone();
        let prev_series = self.build_current_trace_series();
        self.snapshot_manager
            .push_history_run(prev_summary, prev_series);
        self.apply_single_atlas_key(ev.x_key, ev.x_val);
        self.apply_single_atlas_key(ev.y_key, ev.y_val);
        self.synced_legacy_params = [self.param_k, self.param_r, self.param_cap_k];
        self.recalculate();
    }

    fn atlas_context_signature(&self, xk: &str, yk: &str) -> u64 {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8).hash(&mut h);
        (self.method as u8).hash(&mut h);
        self.custom_coupled.hash(&mut h);
        for (k, v) in [
            ("dt", self.dt),
            ("k", self.param_k),
            ("r", self.param_r),
            ("cap_k", self.param_cap_k),
            ("y0", self.ic_y0),
            ("v0", self.ic_v0),
        ] {
            if k != xk && k != yk {
                v.to_bits().hash(&mut h);
            }
        }
        if self.preset == OdePreset::Custom {
            self.math_input.raw_text.hash(&mut h);
            self.coupled_bank.raw_text(0).hash(&mut h);
            self.coupled_bank.raw_text(1).hash(&mut h);
        }
        h.finish()
    }

    pub(super) fn render_ode_atlas_panel(&mut self, ui: &mut egui::Ui, changed: &mut bool) {
        if !self.atlas.is_visible() {
            return;
        }
        ui.add_space(6.0);
        let (xk, yk) = self.atlas.axis_keys();
        let active_x = self.atlas_param_value(xk);
        let active_y = self.atlas_param_value(yk);
        let sig = self.atlas_context_signature(xk, yk);
        let cartographer_seed = OdeRegimeCartographer::from_tool(self, (xk, yk));
        let mut atlas = std::mem::replace(
            &mut self.atlas,
            crate::async_sim::atlas::ParameterAtlasController::new(Vec::new()),
        );
        let teleport = atlas.show_panel(ui, "ode", active_x, active_y, sig, |_keys| {
            Arc::new(cartographer_seed)
        });
        self.atlas = atlas;
        if let Some(ev) = teleport {
            self.apply_atlas_teleport(ev);
            *changed = true;
        }
    }
}
