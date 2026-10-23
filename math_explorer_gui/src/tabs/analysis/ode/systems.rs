#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::{
    OdeModel, OdeSystem, RungeKutta4, TimeStepper, VecState,
};
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub(crate) struct ExponentialOde {
    pub(crate) k: f64,
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
pub(crate) struct HarmonicOscillatorOde {
    pub(crate) k: f64,
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
pub(crate) struct LogisticGrowthOde {
    pub(crate) r: f64,
    pub(crate) cap_k: f64,
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

pub(crate) fn eval_ode_final_state(
    preset: OdePreset,
    dt: f64,
    total_time: f64,
    p_map: &HashMap<String, f64>,
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
            for _ in 0..steps {
                model.step(dt);
            }
            model.state.0[0]
        }
        OdePreset::HarmonicOscillator => {
            let init = VecState(vec![y0, v0]);
            let mut model = OdeModel::new(init.clone(), HarmonicOscillatorOde { k }, RungeKutta4::new(&init));
            for _ in 0..steps {
                model.step(dt);
            }
            model.state.0[0]
        }
        OdePreset::LogisticGrowth => {
            let init = VecState(vec![y0]);
            let mut model = OdeModel::new(
                init.clone(),
                LogisticGrowthOde { r, cap_k },
                RungeKutta4::new(&init),
            );
            for _ in 0..steps {
                model.step(dt);
            }
            model.state.0[0]
        }
    }
}
