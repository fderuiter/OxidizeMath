#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! ODE model definitions and zero-allocation `CompiledExpr` evaluation helpers.

use super::OdeSolverTool;
use crate::framework::math_input::{
    CoupledExpressionBank, MathExpressionInputBar, MAX_SYNTHESIZED_PARAMS,
};
use crate::presets::OdePreset;
use math_commons::ast::{CompiledExpr, Instruction};
use math_explorer::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
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

/// Default coupled 2-state ODE bank (`y' = f(t, y, v)`, `v' = g(t, y, v)`).
#[must_use]
pub fn default_coupled_ode_bank() -> CoupledExpressionBank {
    CoupledExpressionBank::new(&["t", "y", "v"], &[("y'", "v"), ("v'", "-k * y - c * v")])
        .with_param_range(-5.0..=5.0, 1.0)
        .with_initial_params(&[("k", 1.0), ("c", 0.2)])
        .with_system_presets(vec![
            ("Damped Harmonic Oscillator", vec!["v", "-k * y - c * v"]),
            (
                "Lotka-Volterra Predator-Prey",
                vec!["a * y - b * y * v", "d * y * v - g * v"],
            ),
            (
                "Van der Pol Oscillator",
                vec!["v", "mu * (1 - y^2) * v - y"],
            ),
            (
                "Duffing Forced Oscillator",
                vec!["v", "-c * v - a * y - b * y^3 + f * cos(w * t)"],
            ),
        ])
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

/// Zero-allocation compiled bytecode ODE specification for scalar `y' = f(t, y)` and
/// coupled `(y', v') = (f(t, y, v), g(t, y, v))` custom systems.
#[derive(Clone, Debug)]
pub enum CustomOdeSpec {
    /// Scalar 1st-order ODE `y' = f(t, y)`.
    Scalar {
        expr: CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
    },
    /// Coupled 2-state ODE `y' = f(t, y, v), v' = g(t, y, v)`.
    Coupled {
        dy: CompiledExpr,
        dv: CompiledExpr,
        params: [f64; MAX_SYNTHESIZED_PARAMS],
        param_len: usize,
    },
}

impl CustomOdeSpec {
    /// Builds a scalar [`CustomOdeSpec`] from a [`MathExpressionInputBar`] with `(t, y)` axes.
    #[must_use]
    pub fn from_scalar_bar(
        bar: &MathExpressionInputBar,
        param_k: f64,
        param_r: f64,
        param_cap_k: f64,
    ) -> Option<Self> {
        let expr = bar.active_compiled()?.clone();
        let names = expr.param_names();
        if names.len() > MAX_SYNTHESIZED_PARAMS {
            return None;
        }
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        let bar_slice = bar.param_slice();
        for (idx, name) in names.iter().enumerate() {
            params[idx] = match name.as_str() {
                "k" => param_k,
                "r" => param_r,
                "cap_k" => param_cap_k,
                _ => bar_slice.get(idx).copied().unwrap_or(1.0),
            };
        }
        Some(Self::Scalar {
            param_len: names.len(),
            expr,
            params,
        })
    }

    /// Builds a coupled 2-state [`CustomOdeSpec`] from a [`CoupledExpressionBank`] with `(t, y, v)` axes.
    #[must_use]
    pub fn from_coupled_bank(bank: &CoupledExpressionBank) -> Option<Self> {
        let compiled = bank.active_compiled();
        let dy = compiled.first()?.clone();
        let dv = compiled.get(1)?.clone();
        let slice = bank.param_slice();
        if slice.len() > MAX_SYNTHESIZED_PARAMS {
            return None;
        }
        let mut params = [0.0; MAX_SYNTHESIZED_PARAMS];
        params[..slice.len()].copy_from_slice(slice);
        Some(Self::Coupled {
            dy,
            dv,
            params,
            param_len: slice.len(),
        })
    }

    /// Returns whether this custom ODE integrates a second state component `v(t)`.
    #[must_use]
    pub const fn is_coupled(&self) -> bool {
        matches!(self, Self::Coupled { .. })
    }

    /// Clones this spec while overriding named parameter slots from `p_map` (for sensitivity sweeps).
    #[must_use]
    pub fn with_param_overrides(&self, p_map: &HashMap<String, f64>) -> Self {
        let mut cloned = self.clone();
        match &mut cloned {
            Self::Scalar {
                expr,
                params,
                param_len,
            } => {
                for (idx, name) in expr.param_names().iter().take(*param_len).enumerate() {
                    if let Some(&val) = p_map.get(name) {
                        params[idx] = val;
                    }
                }
            }
            Self::Coupled {
                dy,
                params,
                param_len,
                ..
            } => {
                for (idx, name) in dy.param_names().iter().take(*param_len).enumerate() {
                    if let Some(&val) = p_map.get(name) {
                        params[idx] = val;
                    }
                }
            }
        }
        cloned
    }
}

impl OdeSystem<VecState> for CustomOdeSpec {
    fn derivative(&self, t: f64, state: &VecState) -> VecState {
        let mut out = VecState(vec![0.0; state.0.len()]);
        self.derivative_in_place(t, state, &mut out);
        out
    }

    fn derivative_in_place(&self, t: f64, state: &VecState, out: &mut VecState) {
        match self {
            Self::Scalar {
                expr,
                params,
                param_len,
            } => {
                let y = state.0[0];
                out.0[0] = expr.eval_2d(t, y, &params[..*param_len]);
            }
            Self::Coupled {
                dy,
                dv,
                params,
                param_len,
            } => {
                let y = state.0[0];
                let v = state.0.get(1).copied().unwrap_or(0.0);
                let p = &params[..*param_len];
                out.0[0] = dy.eval_3d(t, y, v, p);
                if out.0.len() > 1 {
                    out.0[1] = dv.eval_3d(t, y, v, p);
                }
            }
        }
    }
}

pub fn eval_ode_final_state(
    preset: OdePreset,
    method: IntegrationMethod,
    dt: f64,
    total_time: f64,
    p_map: &HashMap<String, f64>,
    custom_spec: Option<&CustomOdeSpec>,
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
            if let Some(spec) = custom_spec {
                let dynamics = spec.with_param_overrides(p_map);
                let init = if dynamics.is_coupled() {
                    VecState(vec![y0, v0])
                } else {
                    VecState(vec![y0])
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
            OdePreset::Custom if self.custom_coupled => "Coupled two-state ordinary differential equation system: y' = f(t, y, v), v' = g(t, y, v)".into(),
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
            "Custom user-defined scalar or coupled ordinary differential equation".into(),
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
