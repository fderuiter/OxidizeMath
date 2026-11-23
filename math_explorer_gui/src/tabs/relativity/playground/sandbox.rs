#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! Custom `MathExpressionInputBar` + `CompiledExpr` spacetime metric sandbox backed by
//! stack-allocated [`StaticRiemannianMetric`] and
//! [`math_explorer::pure_math::tensor::static_christoffel_symbols`].

use std::collections::BTreeMap;
use std::f64::consts::FRAC_PI_2;

use crate::framework::math_input::MAX_SYNTHESIZED_PARAMS;
use crate::framework::MathExpressionInputBar;
use eframe::egui;
use math_commons::ast::CompiledExpr;
use math_explorer::nalgebra::{SMatrix, SVector};
use math_explorer::pure_math::analysis::ode::{ArrayState, RungeKutta4};
use math_explorer::pure_math::tensor::{
    GeodesicState, GeodesicStepStatus, GeodesicSystem, StaticMetric, StaticRiemannianMetric,
    TensorError,
};

use super::sim::refine_halt_4d;
use super::telemetry::{push_custom_lapse_sample, push_custom_wormhole_sample, CustomSampleCtx};
use super::{GeodesicMode, GeodesicPlaygroundTool, TrajectorySample};

/// Default custom 4D radial lapse expression $f(r)$ ($g_{00} = -f(r)$, $g_{11} = 1/f(r)$).
pub const DEFAULT_CUSTOM_LAPSE_EXPR: &str = "1 - Rs / r";

/// Default custom 3D wormhole circumferential profile $b(l)$ ($g_{22} = b(l)^2$).
pub const DEFAULT_CUSTOM_WORMHOLE_EXPR: &str = "sqrt(b0^2 + l^2)";

/// Maximum UTF-8 byte length permitted for persisted custom metric expressions.
pub const MAX_CUSTOM_METRIC_EXPR_BYTES: usize = 512;

/// Minimum regular radial lapse threshold guarding against horizon division by zero.
pub const MIN_REGULAR_LAPSE: f64 = 0.02;

/// Minimum regular circumferential throat radius guarding against pinch-off singularities.
pub const MIN_REGULAR_THROAT: f64 = 0.10;

pub(crate) fn default_custom_lapse_expr() -> String {
    DEFAULT_CUSTOM_LAPSE_EXPR.to_string()
}

pub(crate) fn is_default_custom_lapse_expr(s: &str) -> bool {
    s == DEFAULT_CUSTOM_LAPSE_EXPR
}

pub(crate) fn default_custom_wormhole_expr() -> String {
    DEFAULT_CUSTOM_WORMHOLE_EXPR.to_string()
}

pub(crate) fn is_default_custom_wormhole_expr(s: &str) -> bool {
    s == DEFAULT_CUSTOM_WORMHOLE_EXPR
}

/// Interactive AST sandbox holding compiled radial lapse $f(r)$ and wormhole profile $b(l)$ bars.
#[derive(Clone, Debug)]
pub struct MetricSandbox {
    /// Expression bar for 4D static radial lapse $f(r)$ with axes `["r", "Rs"]`.
    pub lapse_bar: MathExpressionInputBar,
    /// Expression bar for 3D wormhole areal radius $b(l)$ with axes `["l", "b0"]`.
    pub wormhole_bar: MathExpressionInputBar,
}

impl Default for MetricSandbox {
    fn default() -> Self {
        let lapse_bar = MathExpressionInputBar::new(
            "f(r)",
            DEFAULT_CUSTOM_LAPSE_EXPR,
            vec![
                ("Schwarzschild (1 - Rs/r)", "1 - Rs / r"),
                ("Unit Mass (1 - 2/r)", "1 - 2 / r"),
                ("Reissner-Nordstrom (Charged)", "1 - Rs / r + q^2 / r^2"),
                ("Yukawa Screened Gravity", "1 - (Rs / r) * exp(-m * r)"),
                ("Schwarzschild-de Sitter", "1 - Rs / r - k * r^2"),
            ],
        )
        .with_axes(&["r", "Rs"])
        .with_param_range(-2.0..=2.0, 0.5)
        .with_param_spec("q", "Charge q", -1.8..=1.8, 0.65)
        .with_param_spec("m", "Screening m", 0.0..=2.0, 0.25)
        .with_param_spec("k", "Cosmological k", -0.02..=0.02, 0.004);

        let wormhole_bar = MathExpressionInputBar::new(
            "b(l)",
            DEFAULT_CUSTOM_WORMHOLE_EXPR,
            vec![
                ("Morris-Thorne sqrt(b0^2 + l^2)", "sqrt(b0^2 + l^2)"),
                ("Quartic Throat Flare", "sqrt(b0^2 + l^2 + a * l^4)"),
                ("Hyperbolic Cosh Throat", "b0 * cosh(l / b0)"),
            ],
        )
        .with_axes(&["l", "b0"])
        .with_param_range(0.0..=2.0, 0.1)
        .with_param_spec("a", "Quartic Flare a", 0.0..=0.5, 0.05);

        Self {
            lapse_bar,
            wormhole_bar,
        }
    }
}

impl MetricSandbox {
    /// Validates and restores persisted formula strings and free-parameter slider values.
    pub fn apply_persisted(
        &mut self,
        lapse_expr: &str,
        wormhole_expr: &str,
        params: &BTreeMap<String, f64>,
        require_lapse_valid: bool,
        require_wormhole_valid: bool,
    ) -> Result<(), String> {
        if lapse_expr.len() > MAX_CUSTOM_METRIC_EXPR_BYTES
            || wormhole_expr.len() > MAX_CUSTOM_METRIC_EXPR_BYTES
        {
            return Err("Custom metric expression exceeds byte limit".to_string());
        }
        if params.len() > MAX_SYNTHESIZED_PARAMS || params.values().any(|v| !v.is_finite()) {
            return Err("Invalid or non-finite custom metric parameters".to_string());
        }
        self.lapse_bar.set_expression(lapse_expr);
        self.wormhole_bar.set_expression(wormhole_expr);
        self.lapse_bar.set_params_map(params);
        self.wormhole_bar.set_params_map(params);
        if require_lapse_valid && !self.lapse_bar.is_valid() {
            return Err(self
                .lapse_bar
                .error_message()
                .unwrap_or_else(|| "Invalid custom lapse expression".to_string()));
        }
        if require_wormhole_valid && !self.wormhole_bar.is_valid() {
            return Err(self
                .wormhole_bar
                .error_message()
                .unwrap_or_else(|| "Invalid custom wormhole expression".to_string()));
        }
        Ok(())
    }

    /// Collects merged active free parameters from both bars for compact JSON persistence.
    #[must_use]
    pub fn merged_params_map(&self) -> BTreeMap<String, f64> {
        let mut out = self.lapse_bar.params_map();
        for (k, v) in self.wormhole_bar.params_map() {
            out.entry(k).or_insert(v);
        }
        out
    }

    /// Returns `true` when all synthesized parameter values across both bars are finite.
    #[must_use]
    pub fn has_finite_params(&self) -> bool {
        self.lapse_bar.param_slice().iter().all(|v| v.is_finite())
            && self
                .wormhole_bar
                .param_slice()
                .iter()
                .all(|v| v.is_finite())
    }

    /// Evaluates active custom radial lapse $f(r)$ with zero heap allocations.
    #[inline]
    #[must_use]
    pub fn eval_lapse(&self, r: f64, rs: f64) -> f64 {
        self.lapse_bar
            .active_compiled()
            .map_or(1.0 - rs / r.max(0.1), |c| {
                c.eval_2d(r, rs, self.lapse_bar.param_slice())
            })
    }

    /// Evaluates active custom wormhole profile $b(l)$ with zero heap allocations.
    #[inline]
    #[must_use]
    pub fn eval_wormhole_b(&self, l: f64, b0: f64) -> f64 {
        self.wormhole_bar
            .active_compiled()
            .map_or((b0 * b0 + l * l).sqrt(), |c| {
                c.eval_2d(l, b0, self.wormhole_bar.param_slice())
            })
    }

    /// Estimates the outermost radial horizon where $f(r) \le 0.02$ on $[0.15, 12.0]$.
    #[must_use]
    pub fn estimate_lapse_horizon(&self, rs: f64) -> f64 {
        let mut horizon = 0.0_f64;
        for idx in 1..=96 {
            let r = 0.15 + 11.85 * (idx as f64) / 96.0;
            let f = self.eval_lapse(r, rs);
            if !f.is_finite() || f <= MIN_REGULAR_LAPSE {
                horizon = r;
            }
        }
        horizon
    }
}

/// Wraps a compiled $f(r; R_s, \text{params})$ into a stack-allocated 4D [`StaticRiemannianMetric`].
///
/// Returns `Err(TensorError::SingularMetric)` whenever $f(r) \le 0.02$, $r \le 0.05$,
/// $|\sin\theta| \le 10^{-4}$, or any coordinate/bytecode output is non-finite.
pub(crate) fn build_lapse_static_metric<'a>(
    compiled: &'a CompiledExpr,
    rs: f64,
    params: &'a [f64],
) -> StaticRiemannianMetric<
    impl Fn(&SVector<f64, 4>) -> Result<SMatrix<f64, 4, 4>, TensorError> + 'a,
    4,
> {
    StaticRiemannianMetric::new(move |pt: &SVector<f64, 4>| {
        if !pt.iter().all(|v| v.is_finite()) || !rs.is_finite() {
            return Err(TensorError::SingularMetric);
        }
        let r = pt[1];
        let sin_t = pt[2].sin();
        if r <= 0.05 || sin_t.abs() <= 1e-4 {
            return Err(TensorError::SingularMetric);
        }
        let f = compiled.eval_2d(r, rs, params);
        if !f.is_finite() || f <= MIN_REGULAR_LAPSE {
            return Err(TensorError::SingularMetric);
        }
        let mut g = SMatrix::<f64, 4, 4>::zeros();
        g[(0, 0)] = -f;
        g[(1, 1)] = 1.0 / f;
        g[(2, 2)] = r * r;
        g[(3, 3)] = r * r * sin_t * sin_t;
        Ok(g)
    })
}

/// Wraps a compiled $b(l; b_0, \text{params})$ into a stack-allocated 3D [`StaticRiemannianMetric`].
pub(crate) fn build_wormhole_static_metric<'a>(
    compiled: &'a CompiledExpr,
    b0: f64,
    params: &'a [f64],
) -> StaticRiemannianMetric<
    impl Fn(&SVector<f64, 3>) -> Result<SMatrix<f64, 3, 3>, TensorError> + 'a,
    3,
> {
    StaticRiemannianMetric::new(move |pt: &SVector<f64, 3>| {
        if !pt.iter().all(|v| v.is_finite()) || !b0.is_finite() {
            return Err(TensorError::SingularMetric);
        }
        let b_val = compiled.eval_2d(pt[1], b0, params);
        if !b_val.is_finite() || b_val <= MIN_REGULAR_THROAT {
            return Err(TensorError::SingularMetric);
        }
        let r2 = b_val * b_val;
        if !r2.is_finite() {
            return Err(TensorError::SingularMetric);
        }
        Ok(SMatrix::<f64, 3, 3>::new(
            -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, r2,
        ))
    })
}

impl GeodesicPlaygroundTool {
    pub(super) fn draw_sandbox_controls(&mut self, ui: &mut egui::Ui) -> bool {
        let changed = match self.state.spacetime {
            super::SpacetimePreset::CustomRadialLapse => self.sandbox.lapse_bar.ui(ui),
            super::SpacetimePreset::CustomWormholeProfile => self.sandbox.wormhole_bar.ui(ui),
            _ => false,
        };
        if changed {
            self.sync_sandbox_to_state();
        }
        changed
    }

    pub(crate) fn sync_sandbox_to_state(&mut self) {
        self.state.custom_lapse_expr = self.sandbox.lapse_bar.raw_text.clone();
        self.state.custom_wormhole_expr = self.sandbox.wormhole_bar.raw_text.clone();
        self.state.custom_params = self.sandbox.merged_params_map();
    }

    pub(super) fn integrate_custom_lapse_into(
        &self,
        steps: usize,
        impact_val: f64,
        out: &mut Vec<TrajectorySample>,
    ) -> (bool, f64) {
        let Some(compiled) = self.sandbox.lapse_bar.active_compiled() else {
            return (false, 1.0);
        };
        let params = self.sandbox.lapse_bar.param_slice();
        let rs = self.state.scale_radius;
        let r0 = self.state.initial_radius.max(0.25);
        let f0 = compiled.eval_2d(r0, rs, params);
        if !f0.is_finite() || f0 <= MIN_REGULAR_LAPSE {
            return (true, 1.0);
        }
        let (pos0, vel0, e_val) =
            self.initial_diagonal_4d_state(r0, f0, impact_val, self.state.radial_bias);
        let ctx = CustomSampleCtx {
            mode: self.state.mode,
            compiled,
            scale: rs,
            params,
        };
        let halted = Self::step_custom_lapse_rk4(ctx, pos0, vel0, self.state.step_size, steps, out);
        (halted, e_val)
    }

    fn step_custom_lapse_rk4(
        ctx: CustomSampleCtx<'_>,
        pos0: SVector<f64, 4>,
        vel0: SVector<f64, 4>,
        dt: f64,
        steps: usize,
        out: &mut Vec<TrajectorySample>,
    ) -> bool {
        let sys = GeodesicSystem::new(build_lapse_static_metric(
            ctx.compiled,
            ctx.scale,
            ctx.params,
        ));
        let mut state = ArrayState::<8>::from_pos_vel(&pos0, &vel0);
        let mut solver = RungeKutta4::new(&state);
        push_custom_lapse_sample(
            out,
            ctx,
            0.0,
            &state,
            sys.metric_norm(&state).unwrap_or(0.0),
        );
        for step in 0..steps {
            let lambda = (step as f64) * dt;
            let prev = state;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    let pos = SVector::<f64, 4>::new(state.0[0], state.0[1], FRAC_PI_2, state.0[3]);
                    if sys.metric().christoffel_at(&pos).is_err() {
                        state = prev;
                        let (_, sub_lam) =
                            refine_halt_4d(&sys, &mut solver, lambda, &mut state, dt);
                        if let Ok(n_end) = sys.metric_norm(&state) {
                            push_custom_lapse_sample(out, ctx, sub_lam, &state, n_end);
                        }
                        return true;
                    }
                    push_custom_lapse_sample(out, ctx, lambda + dt, &state, metric_norm);
                    if state.0[1] > 28.0 {
                        return false;
                    }
                }
                GeodesicStepStatus::HaltedSingularity => {
                    let (did_adv, sub_lam) =
                        refine_halt_4d(&sys, &mut solver, lambda, &mut state, dt);
                    if did_adv {
                        if let Ok(n_end) = sys.metric_norm(&state) {
                            push_custom_lapse_sample(out, ctx, sub_lam, &state, n_end);
                        }
                    }
                    return true;
                }
            }
        }
        false
    }

    pub(super) fn integrate_custom_wormhole_into(
        &self,
        steps: usize,
        impact_val: f64,
        out: &mut Vec<TrajectorySample>,
    ) -> (bool, f64) {
        let Some(compiled) = self.sandbox.wormhole_bar.active_compiled() else {
            return (false, 1.0);
        };
        let params = self.sandbox.wormhole_bar.param_slice();
        let b0 = self.state.scale_radius;
        let l0 = self.state.initial_radius;
        let b_init = compiled.eval_2d(l0, b0, params);
        if !b_init.is_finite() || b_init <= MIN_REGULAR_THROAT {
            return (true, 1.0);
        }
        let (ut, ul, uphi, e_val) = Self::wormhole_initial_vel(
            self.state.mode,
            b_init * b_init,
            impact_val,
            self.state.radial_bias,
        );
        let ctx = CustomSampleCtx {
            mode: self.state.mode,
            compiled,
            scale: b0,
            params,
        };
        let sys = GeodesicSystem::new(build_wormhole_static_metric(compiled, b0, params));
        let mut state = ArrayState::<6>::from_pos_vel(
            &SVector::<f64, 3>::new(0.0, l0, 0.0),
            &SVector::<f64, 3>::new(ut, ul, uphi),
        );
        let mut solver = RungeKutta4::new(&state);
        let dt = self.state.step_size;
        push_custom_wormhole_sample(
            out,
            ctx,
            0.0,
            &state,
            sys.metric_norm(&state).unwrap_or(0.0),
        );
        for step in 0..steps {
            let lambda = (step as f64) * dt;
            let prev = state;
            match sys.step_rk4(&mut solver, lambda, &mut state, dt) {
                GeodesicStepStatus::Advanced { metric_norm } => {
                    let pos = SVector::<f64, 3>::new(state.0[0], state.0[1], state.0[2]);
                    if sys.metric().christoffel_at(&pos).is_err() {
                        state = prev;
                        let _ = state;
                        return (true, e_val);
                    }
                    push_custom_wormhole_sample(out, ctx, lambda + dt, &state, metric_norm);
                    if state.0[1].abs() > 20.0 {
                        return (false, e_val);
                    }
                }
                GeodesicStepStatus::HaltedSingularity => return (true, e_val),
            }
        }
        (false, e_val)
    }

    pub(super) fn initial_diagonal_4d_state(
        &self,
        r0: f64,
        f0: f64,
        impact_val: f64,
        radial_bias: f64,
    ) -> (SVector<f64, 4>, SVector<f64, 4>, f64) {
        let safe_f = f0.max(MIN_REGULAR_LAPSE);
        match self.state.mode {
            GeodesicMode::NullPhoton => {
                let e = 1.0;
                let max_b = r0 / safe_f.sqrt();
                let ell = impact_val.clamp(-max_b, max_b);
                let ut = e / safe_f;
                let uphi = ell / (r0 * r0);
                let ur_sq = (e * e - safe_f * (ell * ell) / (r0 * r0)).max(0.0);
                let sign = if radial_bias > 0.0 { 1.0 } else { -1.0 };
                (
                    SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, 0.0),
                    SVector::<f64, 4>::new(ut, sign * ur_sq.sqrt(), 0.0, uphi),
                    e,
                )
            }
            GeodesicMode::TimelikeParticle => {
                let ell = impact_val;
                let uphi = ell / (r0 * r0);
                let ur = radial_bias * 0.45;
                let e_sq = ur * ur + safe_f * (1.0 + (ell * ell) / (r0 * r0));
                let e = e_sq.max(1e-6).sqrt();
                let ut = e / safe_f;
                (
                    SVector::<f64, 4>::new(0.0, r0, FRAC_PI_2, 0.0),
                    SVector::<f64, 4>::new(ut, ur, 0.0, uphi),
                    e,
                )
            }
        }
    }
}
