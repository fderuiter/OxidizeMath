#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! ODE model definitions and evaluation helpers.

use super::OdeSolverTool;
use crate::presets::OdePreset;
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
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

#[derive(Clone, Debug)]
pub struct ExponentialOde {
    pub k: f64,
}

impl OdeSystem<VecState> for ExponentialOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = self.k * state.0[0];
    }
}

#[derive(Clone, Debug)]
pub struct HarmonicOscillatorOde {
    pub k: f64,
}

impl OdeSystem<VecState> for HarmonicOscillatorOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        out.0[0] = state.0[1];
        out.0[1] = -self.k * state.0[0];
    }
}

#[derive(Clone, Debug)]
pub struct LogisticGrowthOde {
    pub r: f64,
    pub cap_k: f64,
}

impl OdeSystem<VecState> for LogisticGrowthOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, _t: f64, state: &VecState, out: &mut VecState) {
        let y = state.0[0];
        out.0[0] = self.r * y * (1.0 - y / self.cap_k);
    }
}

#[derive(Clone, Debug)]
pub struct DynamicAstOde {
    pub expr: Expr,
    pub params: HashMap<String, f64>,
}

impl OdeSystem<VecState> for DynamicAstOde {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }
    fn derivative_in_place(&self, t: f64, state: &VecState, out: &mut VecState) {
        let y = state.0[0];
        out.0[0] = self.expr.eval_2d(t, y, &self.params);
    }
}

pub fn eval_ode_final_state(
    preset: OdePreset,
    method: IntegrationMethod,
    dt: f64,
    total_time: f64,
    p_map: &HashMap<String, f64>,
    custom_expr: Option<&Expr>,
) -> f64 {
    let Some(steps) = crate::presets::checked_ode_step_count(dt, total_time) else {
        return f64::NAN;
    };
    let y0 = p_map.get("y0").copied().unwrap_or(1.0);
    let v0 = p_map.get("v0").copied().unwrap_or(0.0);
    let k = p_map.get("k").copied().unwrap_or(1.0);
    let r = p_map.get("r").copied().unwrap_or(1.0);
    let cap_k = p_map.get("cap_k").copied().unwrap_or(10.0);

    match preset {
        OdePreset::Exponential => {
            let init = VecState(vec![y0]);
            step_model(init, ExponentialOde { k }, method, steps, dt)
        }
        OdePreset::HarmonicOscillator => {
            let init = VecState(vec![y0, v0]);
            step_model(init, HarmonicOscillatorOde { k }, method, steps, dt)
        }
        OdePreset::LogisticGrowth => {
            let init = VecState(vec![y0]);
            step_model(init, LogisticGrowthOde { r, cap_k }, method, steps, dt)
        }
        OdePreset::Custom => {
            if let Some(expr) = custom_expr {
                let init = VecState(vec![y0]);
                let dynamics = DynamicAstOde {
                    expr: expr.clone(),
                    params: p_map.clone(),
                };
                step_model(init, dynamics, method, steps, dt)
            } else {
                f64::NAN
            }
        }
    }
}

fn step_model<D: OdeSystem<VecState> + Clone>(
    init: VecState,
    dyn_sys: D,
    method: IntegrationMethod,
    steps: usize,
    dt: f64,
) -> f64 {
    match method {
        IntegrationMethod::RungeKutta4 => {
            let mut model = OdeModel::new(init.clone(), dyn_sys, RungeKutta4::new(&init));
            for _ in 0..steps {
                model.step(dt);
            }
            model.state.0[0]
        }
        IntegrationMethod::Euler => {
            let mut model = OdeModel::new(init.clone(), dyn_sys, Euler::new(&init));
            for _ in 0..steps {
                model.step(dt);
            }
            model.state.0[0]
        }
        IntegrationMethod::Heun => {
            let mut model = OdeModel::new(init.clone(), dyn_sys, Heun::new(&init));
            for _ in 0..steps {
                model.step(dt);
            }
            model.state.0[0]
        }
    }
}

impl scientific_metadata::theory::TheoryDescribable for OdeSolverTool {
    fn theory_description(&self) -> String {
        match self.preset {
            OdePreset::Exponential => "First-order linear ordinary differential equation describing exponential growth or decay proportional to current quantity.".into(),
            OdePreset::HarmonicOscillator => "Second-order ordinary differential equation modeling simple harmonic motion under a Hooke's law restoring force.".into(),
            OdePreset::LogisticGrowth => "Nonlinear ordinary differential equation modeling density-dependent population growth with an environmental carrying capacity.".into(),
            OdePreset::Custom => "Custom user-defined ordinary differential equation: y' = f(t, y)".into(),
        }
    }
    fn phonetic_description(&self) -> String {
        "Ordinary differential equations theory".into()
    }
    fn theory_citation(&self) -> String {
        match self.preset {
            OdePreset::Exponential => {
                "Differential Equations and Dynamical Systems (Perko, 2001)".into()
            }
            OdePreset::HarmonicOscillator => {
                "Classical Mechanics (Goldstein et al., 2002); Perko (2001)".into()
            }
            OdePreset::LogisticGrowth => {
                "Elements of Mathematical Biology (Lotka, 1925); Verhulst (1838)".into()
            }
            OdePreset::Custom => {
                "Differential Equations and Dynamical Systems (Perko, 2001)".into()
            }
        }
    }
    fn theory_latex_equation(&self) -> Option<String> {
        let eq = match self.preset {
            OdePreset::Exponential => r"\frac{dy}{dt} = k y",
            OdePreset::HarmonicOscillator => r"\frac{d^2 y}{dt^2} = -k y",
            OdePreset::LogisticGrowth => r"\frac{dy}{dt} = r y (1 - \frac{y}{K})",
            OdePreset::Custom => return None,
        };
        Some(eq.into())
    }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "Exponential".into(),
            "First-order linear ODE representing exponential growth or decay".into(),
        );
        map.insert(
            "Harmonic Oscillator".into(),
            "Second-order linear ODE representing simple harmonic motion".into(),
        );
        map.insert(
            "Logistic Growth".into(),
            "Nonlinear first-order ODE modeling population growth with carrying capacity".into(),
        );
        map.insert(
            "Custom".into(),
            "Custom user-defined ordinary differential equation".into(),
        );
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
