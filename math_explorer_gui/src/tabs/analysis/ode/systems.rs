#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! ODE model definitions and evaluation helpers.

use super::OdeSolverTool;
use crate::presets::{InteractivePreset, OdePreset, OdePresetConfig};
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
};
use std::collections::HashMap;

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
        out.0[0] = self.r * state.0[0] * (1.0 - state.0[0] / self.cap_k);
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
    let y0 = p_map.get("y0").copied().unwrap_or(1.0);
    let v0 = p_map.get("v0").copied().unwrap_or(0.0);
    let k = p_map.get("k").copied().unwrap_or(1.0);
    let r = p_map.get("r").copied().unwrap_or(1.0);
    let cap_k = p_map.get("cap_k").copied().unwrap_or(10.0);
    let steps = (total_time / dt).ceil() as usize;

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
            OdePreset::Exponential => r"Exponential model: $\frac{dy}{dt} = k y$".into(),
            OdePreset::HarmonicOscillator => r"Harmonic oscillator: $\frac{d^2 y}{dt^2} = -k y$".into(),
            OdePreset::LogisticGrowth => r"Logistic growth: $\frac{dy}{dt} = r y (1 - \frac{y}{K})$".into(),
            OdePreset::Custom => r"Custom user-defined ordinary differential equation: $y' = f(t, y)$".into(),
        }
    }
    fn phonetic_description(&self) -> String { "Ordinary differential equations theory".into() }
    fn theory_citation(&self) -> String { "Differential Equations and Dynamical Systems (Perko, 2001)".into() }
    fn available_descriptions(&self) -> std::collections::HashMap<String, String> { std::collections::HashMap::new() }
}

impl InteractivePreset for OdeSolverTool {
    type Config = OdePresetConfig;

    fn domain(&self) -> &'static str { "analysis" }
    fn title(&self) -> String { format!("ODE Solver - {}", self.preset.name()) }

    fn export_config(&self) -> Self::Config {
        OdePresetConfig {
            preset: self.preset, dt: self.dt, total_time: self.total_time,
            param_k: self.param_k, param_r: self.param_r, param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0, ic_v0: self.ic_v0,
            custom_expr: self.math_input.raw_text.clone(),
        }
    }

    fn apply_config(&mut self, config: Self::Config) -> Result<(), String> {
        self.preset = config.preset; self.dt = config.dt; self.total_time = config.total_time;
        self.param_k = config.param_k; self.param_r = config.param_r; self.param_cap_k = config.param_cap_k;
        self.ic_y0 = config.ic_y0; self.ic_v0 = config.ic_v0;
        if !config.custom_expr.is_empty() { self.math_input.set_expression(&config.custom_expr); }
        self.recalculate();
        Ok(())
    }
}
