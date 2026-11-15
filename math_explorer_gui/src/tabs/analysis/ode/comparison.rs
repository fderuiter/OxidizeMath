#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::systems::{ExponentialOde, HarmonicOscillatorOde, LogisticGrowthOde};
use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, OdeModel, OdeSystem, RungeKutta4, Solver, TimeStepper, VecState,
};
use serde::{Deserialize, Serialize};

/// Trajectory result from an individual ODE numerical solver run.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SolverTrajectory {
    pub method: IntegrationMethod,
    pub time_series: Vec<f64>,
    pub y_series: Vec<f64>,
    pub v_series: Vec<f64>,
    pub diverged: bool,
}

/// Quantitative discrepancy metrics between two numerical solvers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolverDiscrepancy {
    pub max_error: f64,
    pub rms_error: f64,
}

/// Simulates an ODE preset with the specified numerical solver algorithm.
#[allow(clippy::too_many_arguments)]
pub fn simulate_preset(
    preset: OdePreset,
    method: IntegrationMethod,
    dt: f64,
    total_time: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
) -> SolverTrajectory {
    let mut traj = SolverTrajectory {
        method,
        time_series: Vec::new(),
        y_series: Vec::new(),
        v_series: Vec::new(),
        diverged: false,
    };

    if dt <= 0.0 || total_time <= 0.0 || !dt.is_finite() || !total_time.is_finite() {
        return traj;
    }

    if !ic_y0.is_finite() || (preset == OdePreset::HarmonicOscillator && !ic_v0.is_finite()) {
        traj.diverged = true;
        return traj;
    }

    let num_steps = (total_time / dt).ceil() as usize;
    traj.time_series.reserve(num_steps + 1);
    traj.y_series.reserve(num_steps + 1);
    if preset == OdePreset::HarmonicOscillator {
        traj.v_series.reserve(num_steps + 1);
    }

    match preset {
        OdePreset::Exponential => {
            let init_state = VecState(vec![ic_y0]);
            let dynamics = ExponentialOde { k: param_k };
            run_step_loop(&mut traj, init_state, dynamics, method, dt, num_steps);
        }
        OdePreset::HarmonicOscillator => {
            let init_state = VecState(vec![ic_y0, ic_v0]);
            let dynamics = HarmonicOscillatorOde { k: param_k };
            run_step_loop(&mut traj, init_state, dynamics, method, dt, num_steps);
        }
        OdePreset::LogisticGrowth => {
            let init_state = VecState(vec![ic_y0]);
            let dynamics = LogisticGrowthOde {
                r: param_r,
                cap_k: param_cap_k,
            };
            run_step_loop(&mut traj, init_state, dynamics, method, dt, num_steps);
        }
    }

    traj
}

fn run_step_loop<D: OdeSystem<VecState> + Clone>(
    traj: &mut SolverTrajectory,
    init: VecState,
    dynamics: D,
    method: IntegrationMethod,
    dt: f64,
    num_steps: usize,
) {
    match method {
        IntegrationMethod::RungeKutta4 => {
            let solver = RungeKutta4::new(&init);
            let mut model = OdeModel::new(init, dynamics, solver);
            execute_steps(traj, &mut model, dt, num_steps);
        }
        IntegrationMethod::Euler => {
            let solver = Euler::new(&init);
            let mut model = OdeModel::new(init, dynamics, solver);
            execute_steps(traj, &mut model, dt, num_steps);
        }
        IntegrationMethod::Heun => {
            let solver = Heun::new(&init);
            let mut model = OdeModel::new(init, dynamics, solver);
            execute_steps(traj, &mut model, dt, num_steps);
        }
    }
}

fn execute_steps<D, S>(
    traj: &mut SolverTrajectory,
    model: &mut OdeModel<VecState, D, S>,
    dt: f64,
    num_steps: usize,
) where
    D: OdeSystem<VecState>,
    S: Solver<VecState>,
{
    let has_v = model.state.0.len() > 1;
    for i in 0..=num_steps {
        let y = model.get_state().0[0];
        let v_finite = !has_v || model.get_state().0[1].is_finite();
        if !y.is_finite() || !v_finite {
            traj.diverged = true;
            break;
        }
        traj.time_series.push(i as f64 * dt);
        traj.y_series.push(y);
        if has_v {
            traj.v_series.push(model.get_state().0[1]);
        }
        model.step(dt);
    }
}

/// Computes maximum discrepancy and root-mean-square discrepancy between two solver trajectories.
pub fn compute_discrepancy(a: &SolverTrajectory, b: &SolverTrajectory) -> SolverDiscrepancy {
    let mut max_err: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let mut count: usize = 0;

    for (&ya, &yb) in a.y_series.iter().zip(b.y_series.iter()) {
        if ya.is_finite() && yb.is_finite() {
            let err = (ya - yb).abs();
            if err > max_err {
                max_err = err;
            }
            sum_sq_err += err * err;
            count += 1;
        }
    }

    let rms_error = if count > 0 {
        (sum_sq_err / count as f64).sqrt()
    } else {
        0.0
    };

    SolverDiscrepancy {
        max_error: max_err,
        rms_error,
    }
}
