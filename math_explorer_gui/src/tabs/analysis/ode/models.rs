#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

//! ODE model definitions and evaluation helpers.

use crate::presets::OdePreset;
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{
    OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
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

pub struct SimResults {
    pub time_series: Vec<f64>,
    pub y_series: Vec<f64>,
    pub v_series: Vec<f64>,
    pub diverged: bool,
}

pub struct OdeSimConfig<'a> {
    pub preset: OdePreset,
    pub dt: f64,
    pub total_time: f64,
    pub k: f64,
    pub r: f64,
    pub cap_k: f64,
    pub y0: f64,
    pub v0: f64,
    pub custom_expr: Option<&'a Expr>,
}

pub fn run_simulation(cfg: OdeSimConfig<'_>) -> SimResults {
    let mut res = SimResults {
        time_series: Vec::new(),
        y_series: Vec::new(),
        v_series: Vec::new(),
        diverged: false,
    };

    if cfg.dt <= 0.0 || cfg.total_time <= 0.0 || !cfg.dt.is_finite() || !cfg.total_time.is_finite() {
        return res;
    }

    let num_steps = (cfg.total_time / cfg.dt).ceil() as usize;

    match cfg.preset {
        OdePreset::Exponential => {
            let init = VecState(vec![cfg.y0]);
            let mut model = OdeModel::new(init.clone(), ExponentialOde { k: cfg.k }, RungeKutta4::new(&init));
            step_1d(&mut model, cfg.dt, num_steps, &mut res);
        }
        OdePreset::HarmonicOscillator => {
            let init = VecState(vec![cfg.y0, cfg.v0]);
            let mut model = OdeModel::new(init.clone(), HarmonicOscillatorOde { k: cfg.k }, RungeKutta4::new(&init));
            for i in 0..=num_steps {
                let y = model.get_state().0[0];
                let v = model.get_state().0[1];
                if !y.is_finite() || !v.is_finite() {
                    res.diverged = true;
                    break;
                }
                res.time_series.push(i as f64 * cfg.dt);
                res.y_series.push(y);
                res.v_series.push(v);
                model.step(cfg.dt);
            }
        }
        OdePreset::LogisticGrowth => {
            let init = VecState(vec![cfg.y0]);
            let mut model = OdeModel::new(init.clone(), LogisticGrowthOde { r: cfg.r, cap_k: cfg.cap_k }, RungeKutta4::new(&init));
            step_1d(&mut model, cfg.dt, num_steps, &mut res);
        }
        OdePreset::Custom => {
            if let Some(expr) = cfg.custom_expr {
                let mut p_map = HashMap::new();
                p_map.insert("k".to_string(), cfg.k);
                p_map.insert("r".to_string(), cfg.r);
                p_map.insert("cap_k".to_string(), cfg.cap_k);
                p_map.insert("K".to_string(), cfg.cap_k);
                let init = VecState(vec![cfg.y0]);
                let dynamics = DynamicAstOde { expr: expr.clone(), params: p_map };
                let mut model = OdeModel::new(init.clone(), dynamics, RungeKutta4::new(&init));
                step_1d(&mut model, cfg.dt, num_steps, &mut res);
            } else {
                res.diverged = true;
            }
        }
    }

    res
}

fn step_1d<S: OdeSystem<VecState>>(
    model: &mut OdeModel<VecState, S, RungeKutta4<VecState>>,
    dt: f64,
    steps: usize,
    res: &mut SimResults,
) {
    for i in 0..=steps {
        let y = model.get_state().0[0];
        if !y.is_finite() {
            res.diverged = true;
            break;
        }
        res.time_series.push(i as f64 * dt);
        res.y_series.push(y);
        model.step(dt);
    }
}

pub fn eval_ode_final_state(
    preset: OdePreset,
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
            let mut model = OdeModel::new(init.clone(), ExponentialOde { k }, RungeKutta4::new(&init));
            for _ in 0..steps { model.step(dt); }
            model.state.0[0]
        }
        OdePreset::HarmonicOscillator => {
            let init = VecState(vec![y0, v0]);
            let mut model = OdeModel::new(init.clone(), HarmonicOscillatorOde { k }, RungeKutta4::new(&init));
            for _ in 0..steps { model.step(dt); }
            model.state.0[0]
        }
        OdePreset::LogisticGrowth => {
            let init = VecState(vec![y0]);
            let mut model = OdeModel::new(init.clone(), LogisticGrowthOde { r, cap_k }, RungeKutta4::new(&init));
            for _ in 0..steps { model.step(dt); }
            model.state.0[0]
        }
        OdePreset::Custom => {
            if let Some(expr) = custom_expr {
                let init = VecState(vec![y0]);
                let dynamics = DynamicAstOde { expr: expr.clone(), params: p_map.clone() };
                let mut model = OdeModel::new(init.clone(), dynamics, RungeKutta4::new(&init));
                for _ in 0..steps { model.step(dt); }
                model.state.0[0]
            } else {
                f64::NAN
            }
        }
    }
}
