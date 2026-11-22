#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! ODE model definitions and zero-per-step-allocation evaluation helpers.

use super::OdeSolverTool;
use crate::presets::OdePreset;
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{
    ArrayState, Euler, Heun, IntegrationMethod, OdeSystem, RungeKutta4, Solver,
};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub(super) struct AstContext<'a>(pub(super) Option<&'a Expr>);

impl Hash for AstContext<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let Some(root) = self.0 else {
            0_u8.hash(state);
            return;
        };
        1_u8.hash(state);
        let mut stack: [Option<&Expr>; 64] = [None; 64];
        stack[0] = Some(root);
        let mut top = 1_usize;
        for _ in 0..256 {
            if top == 0 {
                break;
            }
            top -= 1;
            if let Some(node) = stack[top] {
                push_and_hash_expr_node(node, &mut stack, &mut top, state);
            }
        }
    }
}

fn push_and_hash_expr_node<'a, H: Hasher>(
    node: &'a Expr,
    stack: &mut [Option<&'a Expr>; 64],
    top: &mut usize,
    state: &mut H,
) {
    match node {
        Expr::Number(n) => {
            1_u8.hash(state);
            crate::sensitivity::canonical_f64_bits(*n).hash(state);
        }
        Expr::Variable(v) => {
            2_u8.hash(state);
            v.hash(state);
        }
        Expr::Binary { op, left, right } => {
            3_u8.hash(state);
            op.hash(state);
            if *top + 2 <= stack.len() {
                stack[*top] = Some(right);
                stack[*top + 1] = Some(left);
                *top += 2;
            }
        }
        Expr::Unary { op, expr } => {
            4_u8.hash(state);
            op.hash(state);
            if *top < stack.len() {
                stack[*top] = Some(expr);
                *top += 1;
            }
        }
        Expr::Function { func, args } => {
            5_u8.hash(state);
            func.hash(state);
            args.len().hash(state);
            for arg in args.iter().rev() {
                if *top < stack.len() {
                    stack[*top] = Some(arg);
                    *top += 1;
                }
            }
        }
    }
}

/// Builds the parameter lookup map shared across scalar and planar AST ODE evaluations.
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
pub struct DynamicAstOde<'a> {
    pub expr: &'a Expr,
    pub params: HashMap<String, f64>,
}

impl OdeSystem<ArrayState<1>> for DynamicAstOde<'_> {
    fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
        ArrayState([self.expr.eval_2d(t, state.0[0], &self.params)])
    }
    fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
        out.0[0] = self.expr.eval_2d(t, state.0[0], &self.params);
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
    custom_expr: Option<&Expr>,
) -> f64 {
    let spec = super::comparison::OdeSimSpec {
        preset,
        method,
        dt,
        total_time,
        param_k: p_map.get("k").copied().unwrap_or(1.0),
        param_r: p_map.get("r").copied().unwrap_or(1.0),
        param_cap_k: p_map.get("cap_k").copied().unwrap_or(10.0),
        ic_y0: p_map.get("y0").copied().unwrap_or(1.0),
        ic_v0: p_map.get("v0").copied().unwrap_or(0.0),
        matrix_a: super::poincare_map::DEFAULT_MATRIX_A,
        custom_expr,
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
    let cap_k = p_map.get("cap_k").copied().unwrap_or(spec.param_cap_k);

    match spec.preset {
        OdePreset::Exponential => step_final_1d(ArrayState([y0]), &ExponentialOde { k }, spec.method, steps, spec.dt),
        OdePreset::HarmonicOscillator => {
            step_final_2d(ArrayState([y0, v0]), &HarmonicOscillatorOde { k }, spec.method, steps, spec.dt)
        }
        OdePreset::LogisticGrowth => {
            step_final_1d(ArrayState([y0]), &LogisticGrowthOde { r, cap_k }, spec.method, steps, spec.dt)
        }
        OdePreset::LinearSystem2D => {
            let a11 = p_map.get("a11").copied().unwrap_or(spec.matrix_a[0][0]);
            let a12 = p_map.get("a12").copied().unwrap_or(spec.matrix_a[0][1]);
            let a21 = p_map.get("a21").copied().unwrap_or(spec.matrix_a[1][0]);
            let a22 = p_map.get("a22").copied().unwrap_or(spec.matrix_a[1][1]);
            let dyn_sys = Linear2DOde {
                matrix_a: [[a11, a12], [a21, a22]],
            };
            step_final_2d(ArrayState([y0, v0]), &dyn_sys, spec.method, steps, spec.dt)
        }
        OdePreset::PlanarCustom => match (spec.planar_x_expr, spec.planar_y_expr) {
            (Some(fx), Some(fy)) => {
                let mut params = p_map.clone();
                params.insert("K".to_string(), cap_k);
                let dyn_sys = PlanarAstOde {
                    x_expr: fx,
                    y_expr: fy,
                    params,
                };
                step_final_2d(ArrayState([y0, v0]), &dyn_sys, spec.method, steps, spec.dt)
            }
            _ => f64::NAN,
        },
        OdePreset::Custom => match spec.custom_expr {
            Some(expr) => {
                let mut params = p_map.clone();
                params.insert("K".to_string(), cap_k);
                let dyn_sys = DynamicAstOde { expr, params };
                step_final_1d(ArrayState([y0]), &dyn_sys, spec.method, steps, spec.dt)
            }
            None => f64::NAN,
        },
    }
}

fn step_final_1d<D: OdeSystem<ArrayState<1>>>(
    mut state: ArrayState<1>,
    dyn_sys: &D,
    method: IntegrationMethod,
    steps: usize,
    dt: f64,
) -> f64 {
    match method {
        IntegrationMethod::RungeKutta4 => {
            let mut solver = RungeKutta4::new(&state);
            for i in 0..steps {
                solver.step(dyn_sys, i as f64 * dt, &mut state, dt);
            }
        }
        IntegrationMethod::Euler => {
            let mut solver = Euler::new(&state);
            for i in 0..steps {
                solver.step(dyn_sys, i as f64 * dt, &mut state, dt);
            }
        }
        IntegrationMethod::Heun => {
            let mut solver = Heun::new(&state);
            for i in 0..steps {
                solver.step(dyn_sys, i as f64 * dt, &mut state, dt);
            }
        }
    }
    state.0[0]
}

fn step_final_2d<D: OdeSystem<ArrayState<2>>>(
    mut state: ArrayState<2>,
    dyn_sys: &D,
    method: IntegrationMethod,
    steps: usize,
    dt: f64,
) -> f64 {
    match method {
        IntegrationMethod::RungeKutta4 => {
            let mut solver = RungeKutta4::new(&state);
            for i in 0..steps {
                solver.step(dyn_sys, i as f64 * dt, &mut state, dt);
            }
        }
        IntegrationMethod::Euler => {
            let mut solver = Euler::new(&state);
            for i in 0..steps {
                solver.step(dyn_sys, i as f64 * dt, &mut state, dt);
            }
        }
        IntegrationMethod::Heun => {
            let mut solver = Heun::new(&state);
            for i in 0..steps {
                solver.step(dyn_sys, i as f64 * dt, &mut state, dt);
            }
        }
    }
    state.0[0]
}

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn theory_description(&self) -> String {
        match self.preset {
            OdePreset::Exponential => "First-order linear ordinary differential equation describing exponential growth or decay proportional to current quantity.".into(),
            OdePreset::HarmonicOscillator => "Second-order ordinary differential equation modeling simple harmonic motion under a Hooke's law restoring force.".into(),
            OdePreset::LogisticGrowth => "Nonlinear ordinary differential equation modeling density-dependent population growth with an environmental carrying capacity.".into(),
            OdePreset::LinearSystem2D => "Planar 2x2 linear dynamical system classified on the Poincaré trace-determinant bifurcation diagram.".into(),
            OdePreset::PlanarCustom => "Coupled 2D planar ordinary differential equations: x' = f(t, x, y), y' = g(t, x, y).".into(),
            OdePreset::Custom => "Custom user-defined ordinary differential equation: y' = f(t, y)".into(),
        }
    }
    fn phonetic_description(&self) -> String {
        "Ordinary differential equations theory".into()
    }
    fn theory_citation(&self) -> String {
        match self.preset {
            OdePreset::Exponential | OdePreset::LinearSystem2D | OdePreset::PlanarCustom | OdePreset::Custom => {
                "Differential Equations and Dynamical Systems (Perko, 2001)".into()
            }
            OdePreset::HarmonicOscillator => {
                "Classical Mechanics (Goldstein et al., 2002); Perko (2001)".into()
            }
            OdePreset::LogisticGrowth => {
                "Elements of Mathematical Biology (Lotka, 1925); Verhulst (1838)".into()
            }
        }
    }
    fn theory_latex_equation(&self) -> Option<String> {
        let eq = match self.preset {
            OdePreset::Exponential => r"\frac{dy}{dt} = k y",
            OdePreset::HarmonicOscillator => r"\frac{d^2 y}{dt^2} = -k y",
            OdePreset::LogisticGrowth => r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
            OdePreset::LinearSystem2D => r"\dot{\mathbf{x}} = A \mathbf{x}",
            OdePreset::PlanarCustom | OdePreset::Custom => return None,
        };
        Some(eq.into())
    }
    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("Exponential".into(), "First-order linear ODE representing exponential growth or decay".into());
        map.insert("Harmonic Oscillator".into(), "Second-order linear ODE representing simple harmonic motion".into());
        map.insert("Logistic Growth".into(), "Nonlinear first-order ODE modeling population growth with carrying capacity".into());
        map.insert("Linear 2D".into(), "Planar 2x2 linear system with Poincaré stability regimes".into());
        map.insert("Planar Custom".into(), "Coupled 2D user-defined ODE phase-plane system".into());
        map.insert("Custom".into(), "Custom user-defined ordinary differential equation".into());
        map
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "harmonic_oscillator",
        title: "Harmonic Oscillator",
        description: "y'' = -k·y: Second-order differential equation modeling undamped periodic spring motion.",
        category: "Differential Equations",
        icon: "〰️",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_HARMONIC_OSCILLATOR,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "logistic_growth",
        title: "Logistic Growth",
        description: "y' = r·y·(1 - y/K): Non-linear growth constrained by carrying capacity K.",
        category: "Population Dynamics",
        icon: "📈",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_LOGISTIC_GROWTH,
    }
}

inventory::submit! {
    crate::starter_gallery::StarterPresetMetadata {
        id: "exponential_decay",
        title: "Exponential Decay",
        description: "y' = k·y: First-order differential decay process with constant half-life.",
        category: "Calculus",
        icon: "📉",
        domain: "analysis",
        tool_name: Some("OdeSolverTool"),
        preset_generator: || crate::starter_gallery::PRESET_EXPONENTIAL_DECAY,
    }
}
