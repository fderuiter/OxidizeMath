#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! ODE model definitions and planar AST evaluation helpers.

pub use super::custom_spec::{default_coupled_ode_bank, CustomOdeSpec};

use super::OdeSolverTool;
use crate::presets::OdePreset;
use math_commons::ast::{CompiledExpr, Expr, Instruction};
use math_explorer::pure_math::analysis::ode::{
    ArrayState, BackwardEuler, DormandPrince45, Euler, Heun, ImplicitTrapezoidal, IndexedState,
    IntegrationMethod, OdeSystem, Rosenbrock23, RungeKutta4, Solver,
};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub(super) struct CustomOdeContext<'a>(pub(super) Option<&'a CustomOdeSpec>);

impl Hash for CustomOdeContext<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.0 {
            None => 0_u8.hash(state),
            Some(CustomOdeSpec::Scalar { expr, .. }) => {
                1_u8.hash(state);
                hash_compiled_expr(expr, state);
            }
            Some(CustomOdeSpec::Coupled { dy, dv, .. }) => {
                2_u8.hash(state);
                hash_compiled_expr(dy, state);
                hash_compiled_expr(dv, state);
            }
        }
    }
}

fn hash_compiled_expr<H: Hasher>(compiled: &CompiledExpr, state: &mut H) {
    compiled.axis_count().hash(state);
    compiled.var_names().hash(state);
    compiled.instructions().len().hash(state);
    for &inst in compiled.instructions() {
        hash_instruction(inst, state);
    }
}

fn hash_instruction<H: Hasher>(inst: Instruction, state: &mut H) {
    match inst {
        Instruction::PushConst(n) => {
            1_u8.hash(state);
            crate::sensitivity::canonical_f64_bits(n).hash(state);
        }
        Instruction::PushImagUnit => 2_u8.hash(state),
        Instruction::LoadVar(slot) => (3_u8, slot).hash(state),
        Instruction::Unary(op) => (4_u8, op).hash(state),
        Instruction::Binary(op) => (5_u8, op).hash(state),
        Instruction::Call(func) => (6_u8, func).hash(state),
    }
}

/// Builds the parameter lookup map shared across planar AST ODE evaluations.
#[must_use]
pub fn build_ode_param_map(param_k: f64, param_r: f64, param_cap_k: f64) -> HashMap<String, f64> {
    let mut p_map = HashMap::with_capacity(4);
    p_map.insert("k".to_string(), param_k);
    p_map.insert("r".to_string(), param_r);
    p_map.insert("cap_k".to_string(), param_cap_k);
    p_map.insert("K".to_string(), param_cap_k);
    p_map
}

impl OdeSolverTool {
    /// Evaluates the instantaneous 2D right-hand side `[dx/dt, dy/dt]` at `(t, x, y)`.
    #[must_use]
    pub fn eval_planar_rhs_at(
        &self,
        t: f64,
        x: f64,
        y: f64,
        params: &HashMap<String, f64>,
    ) -> [f64; 2] {
        match self.preset {
            OdePreset::HarmonicOscillator => [y, -self.param_k * x],
            OdePreset::LinearSystem2D => [
                self.matrix_a[0][0] * x + self.matrix_a[0][1] * y,
                self.matrix_a[1][0] * x + self.matrix_a[1][1] * y,
            ],
            OdePreset::PlanarCustom => {
                match (
                    self.planar_x_input.cached_expr(),
                    self.planar_y_input.cached_expr(),
                ) {
                    (Some(fx), Some(fy)) => [
                        fx.eval_planar(t, x, y, params),
                        fy.eval_planar(t, x, y, params),
                    ],
                    _ => [f64::NAN, f64::NAN],
                }
            }
            OdePreset::Custom if self.custom_coupled => {
                let compiled = self.coupled_bank.active_compiled();
                let p = self.coupled_bank.param_slice();
                match (compiled.first(), compiled.get(1)) {
                    (Some(dy), Some(dv)) => [dy.eval_3d(t, x, y, p), dv.eval_3d(t, x, y, p)],
                    _ => [f64::NAN, f64::NAN],
                }
            }
            OdePreset::Exponential | OdePreset::LogisticGrowth | OdePreset::Custom => [0.0, 0.0],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ExponentialOde {
    pub k: f64,
}

impl OdeSystem<ArrayState<1>> for ExponentialOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([self.k * state.0[0]])
    }
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = self.k * state.0[0];
    }
}

#[derive(Clone, Copy, Debug)]
pub struct HarmonicOscillatorOde {
    pub k: f64,
}

impl OdeSystem<ArrayState<2>> for HarmonicOscillatorOde {
    fn derivative(&self, _t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        ArrayState([state.0[1], -self.k * state.0[0]])
    }
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        out.0[0] = state.0[1];
        out.0[1] = -self.k * state.0[0];
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LogisticGrowthOde {
    pub r: f64,
    pub cap_k: f64,
}

impl OdeSystem<ArrayState<1>> for LogisticGrowthOde {
    fn derivative(&self, _t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        let y = state.0[0];
        ArrayState([self.r * y * (1.0 - y / self.cap_k)])
    }
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        let y = state.0[0];
        out.0[0] = self.r * y * (1.0 - y / self.cap_k);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Linear2DOde {
    pub matrix_a: [[f64; 2]; 2],
}

impl OdeSystem<ArrayState<2>> for Linear2DOde {
    fn derivative(&self, _t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        let [x, y] = state.0;
        ArrayState([
            self.matrix_a[0][0] * x + self.matrix_a[0][1] * y,
            self.matrix_a[1][0] * x + self.matrix_a[1][1] * y,
        ])
    }
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        let [x, y] = state.0;
        out.0[0] = self.matrix_a[0][0] * x + self.matrix_a[0][1] * y;
        out.0[1] = self.matrix_a[1][0] * x + self.matrix_a[1][1] * y;
    }
}

#[derive(Clone, Debug)]
pub struct PlanarAstOde<'a> {
    pub x_expr: &'a Expr,
    pub y_expr: &'a Expr,
    pub params: HashMap<String, f64>,
}

impl OdeSystem<ArrayState<2>> for PlanarAstOde<'_> {
    fn derivative(&self, t: f64, state: &ArrayState<2>) -> ArrayState<2> {
        let [x, y] = state.0;
        ArrayState([
            self.x_expr.eval_planar(t, x, y, &self.params),
            self.y_expr.eval_planar(t, x, y, &self.params),
        ])
    }
    fn derivative_in_place(&self, t: f64, state: &ArrayState<2>, out: &mut ArrayState<2>) {
        let [x, y] = state.0;
        out.0[0] = self.x_expr.eval_planar(t, x, y, &self.params);
        out.0[1] = self.y_expr.eval_planar(t, x, y, &self.params);
    }
}

#[cfg(test)]
pub fn eval_ode_final_state(
    preset: OdePreset,
    method: IntegrationMethod,
    dt: f64,
    total_time: f64,
    p_map: &HashMap<String, f64>,
    custom_spec: Option<&CustomOdeSpec>,
) -> f64 {
    let spec = super::comparison::OdeSimSpec {
        preset,
        method,
        dt,
        total_time,
        adaptive_tol: super::DEFAULT_ODE_ADAPTIVE_TOL,
        param_k: p_map.get("k").copied().unwrap_or(1.0),
        param_r: p_map.get("r").copied().unwrap_or(1.0),
        param_cap_k: p_map
            .get("cap_k")
            .or_else(|| p_map.get("K"))
            .copied()
            .unwrap_or(10.0),
        ic_y0: p_map.get("y0").copied().unwrap_or(1.0),
        ic_v0: p_map.get("v0").copied().unwrap_or(0.0),
        matrix_a: super::poincare_map::DEFAULT_MATRIX_A,
        custom_spec,
        planar_x_expr: None,
        planar_y_expr: None,
    };
    eval_ode_final_state_planar(&spec, p_map)
}

pub fn eval_ode_final_state_planar(
    spec: &super::comparison::OdeSimSpec<'_>,
    p_map: &HashMap<String, f64>,
) -> f64 {
    let Some(steps) = crate::presets::checked_ode_step_count(spec.dt, spec.total_time) else {
        return f64::NAN;
    };
    let y0 = p_map.get("y0").copied().unwrap_or(spec.ic_y0);
    let v0 = p_map.get("v0").copied().unwrap_or(spec.ic_v0);
    let k = p_map.get("k").copied().unwrap_or(spec.param_k);
    let r = p_map.get("r").copied().unwrap_or(spec.param_r);
    let cap_k = p_map
        .get("cap_k")
        .or_else(|| p_map.get("K"))
        .copied()
        .unwrap_or(spec.param_cap_k);

    match spec.preset {
        OdePreset::Exponential => step_final(ArrayState([y0]), &ExponentialOde { k }, spec, steps),
        OdePreset::HarmonicOscillator => step_final(
            ArrayState([y0, v0]),
            &HarmonicOscillatorOde { k },
            spec,
            steps,
        ),
        OdePreset::LogisticGrowth => step_final(
            ArrayState([y0]),
            &LogisticGrowthOde { r, cap_k },
            spec,
            steps,
        ),
        OdePreset::LinearSystem2D => {
            let dyn_sys = Linear2DOde {
                matrix_a: [
                    [
                        p_map.get("a11").copied().unwrap_or(spec.matrix_a[0][0]),
                        p_map.get("a12").copied().unwrap_or(spec.matrix_a[0][1]),
                    ],
                    [
                        p_map.get("a21").copied().unwrap_or(spec.matrix_a[1][0]),
                        p_map.get("a22").copied().unwrap_or(spec.matrix_a[1][1]),
                    ],
                ],
            };
            step_final(ArrayState([y0, v0]), &dyn_sys, spec, steps)
        }
        OdePreset::PlanarCustom => eval_planar_custom_final([y0, v0], cap_k, spec, p_map, steps),
        OdePreset::Custom => eval_custom_spec_final([y0, v0], spec, p_map, steps),
    }
}

fn eval_custom_spec_final(
    ic: [f64; 2],
    spec: &super::comparison::OdeSimSpec<'_>,
    p_map: &HashMap<String, f64>,
    steps: usize,
) -> f64 {
    let Some(cspec) = spec.custom_spec else {
        return f64::NAN;
    };
    let dynamics = cspec.with_param_overrides(p_map);
    if dynamics.is_coupled() {
        step_final(ArrayState(ic), &dynamics, spec, steps)
    } else {
        step_final(ArrayState([ic[0]]), &dynamics, spec, steps)
    }
}

fn eval_planar_custom_final(
    ic: [f64; 2],
    cap_k: f64,
    spec: &super::comparison::OdeSimSpec<'_>,
    p_map: &HashMap<String, f64>,
    steps: usize,
) -> f64 {
    let (Some(x_expr), Some(y_expr)) = (spec.planar_x_expr, spec.planar_y_expr) else {
        return f64::NAN;
    };
    let mut params = p_map.clone();
    params.entry("cap_k".to_string()).or_insert(cap_k);
    params.insert("K".to_string(), cap_k);
    let dyn_sys = PlanarAstOde {
        x_expr,
        y_expr,
        params,
    };
    step_final(ArrayState(ic), &dyn_sys, spec, steps)
}

fn run_solver_n_steps<const N: usize, D, S>(
    state: &mut ArrayState<N>,
    dyn_sys: &D,
    solver: &mut S,
    dt: f64,
    steps: usize,
) where
    D: OdeSystem<ArrayState<N>>,
    S: Solver<ArrayState<N>>,
{
    for i in 0..steps {
        solver.step(dyn_sys, i as f64 * dt, state, dt);
    }
}

fn step_final<const N: usize, D>(
    mut state: ArrayState<N>,
    dyn_sys: &D,
    spec: &super::comparison::OdeSimSpec<'_>,
    steps: usize,
) -> f64
where
    D: OdeSystem<ArrayState<N>>,
    ArrayState<N>: IndexedState,
{
    let dt = spec.dt;
    let rtol = super::comparison::sanitize_adaptive_tol(spec.adaptive_tol);
    let atol = (rtol * 0.01).max(1e-14);
    match spec.method {
        IntegrationMethod::RungeKutta4 => {
            let mut s = RungeKutta4::new(&state);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
        IntegrationMethod::Euler => {
            let mut s = Euler::new(&state);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
        IntegrationMethod::Heun => {
            let mut s = Heun::new(&state);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
        IntegrationMethod::DormandPrince45 => {
            let mut s = DormandPrince45::new(&state).with_tolerances(rtol, atol);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
        IntegrationMethod::BackwardEuler => {
            let mut s = BackwardEuler::new(&state);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
        IntegrationMethod::ImplicitTrapezoidal => {
            let mut s = ImplicitTrapezoidal::new(&state);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
        IntegrationMethod::Rosenbrock23 => {
            let mut s = Rosenbrock23::new(&state).with_tolerances(rtol, atol);
            run_solver_n_steps(&mut state, dyn_sys, &mut s, dt, steps);
        }
    }
    state.0[0]
}

impl OdeSolverTool {
    pub(crate) fn compute_active_baseline_hash(&self, custom_spec: Option<&CustomOdeSpec>) -> u64 {
        use crate::sensitivity::canonical_f64_bits;

        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (self.preset as u8, self.method as u8).hash(&mut hasher);
        for val in [
            self.dt,
            self.total_time,
            super::comparison::sanitize_adaptive_tol(self.adaptive_tol),
            self.param_k,
            self.param_r,
            self.param_cap_k,
            self.ic_y0,
            self.ic_v0,
        ] {
            canonical_f64_bits(val).hash(&mut hasher);
        }
        for val in self.matrix_a.iter().flatten() {
            canonical_f64_bits(*val).hash(&mut hasher);
        }
        if self.preset == OdePreset::PlanarCustom {
            self.planar_x_input.raw_text.hash(&mut hasher);
            self.planar_y_input.raw_text.hash(&mut hasher);
        }
        if let Some(spec) = custom_spec {
            1_u8.hash(&mut hasher);
            spec.hash_with_params(&mut hasher);
        } else {
            0_u8.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn reusable_baseline_final_y(
        &self,
        custom_spec: Option<&CustomOdeSpec>,
        param_refs: &[(&str, f64)],
    ) -> Option<f64> {
        if self.diverged || self.has_custom_input_error() {
            return None;
        }
        let steps = crate::presets::checked_ode_step_count(self.dt, self.total_time)?;
        if self.y_series.len() != steps + 1
            || (self.has_second_state() && self.v_series.len() != steps + 1)
        {
            return None;
        }
        if self.active_baseline_hash != Some(self.compute_active_baseline_hash(custom_spec)) {
            return None;
        }
        if self.preset == OdePreset::Custom
            && !custom_spec.is_some_and(|spec| spec.matches_baseline_params(param_refs))
        {
            return None;
        }
        self.y_series.last().copied().filter(|v| v.is_finite())
    }

    pub(super) fn run_sensitivity_analysis(&mut self, defer_gesture: bool) {
        if defer_gesture {
            self.sensitivity_engine.defer_for_gesture();
            return;
        }

        use crate::sensitivity::canonical_f64_bits;

        let dynamic_params = self.collect_sensitivity_params();
        let param_refs: Vec<(&str, f64)> = dynamic_params
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();

        let custom_spec = self.build_custom_ode_spec();
        let baseline_hint = self.reusable_baseline_final_y(custom_spec.as_ref(), &param_refs);
        let px = self.planar_x_input.cached_expr().cloned();
        let py = self.planar_y_input.cached_expr().cloned();
        let base_spec =
            self.sim_spec_with_planar(self.method, custom_spec.as_ref(), px.as_ref(), py.as_ref());
        let custom_ctx =
            (self.preset == OdePreset::Custom).then_some(CustomOdeContext(custom_spec.as_ref()));
        let planar_ctx = (self.preset == OdePreset::PlanarCustom).then_some((
            self.planar_x_input.raw_text.as_str(),
            self.planar_y_input.raw_text.as_str(),
        ));
        let clean_tol = super::comparison::sanitize_adaptive_tol(base_spec.adaptive_tol);
        let context = (
            base_spec.preset as u8,
            base_spec.method as u8,
            canonical_f64_bits(base_spec.dt),
            canonical_f64_bits(base_spec.total_time),
            canonical_f64_bits(clean_tol),
            custom_ctx,
            planar_ctx,
        );

        self.sensitivity_engine.compute_with_baseline(
            &param_refs,
            &context,
            baseline_hint,
            |p_map| eval_ode_final_state_planar(&base_spec, p_map),
        );
    }
}
