#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::systems::{DynamicAstOde, ExponentialOde, HarmonicOscillatorOde, LogisticGrowthOde};
use super::OdeSolverTool;
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use math_commons::ast::Expr;
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
    pub divergence_info: Option<(f64, usize, String)>,
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
    custom_expr: Option<&Expr>,
) -> SolverTrajectory {
    let mut traj = SolverTrajectory {
        method,
        time_series: Vec::new(),
        y_series: Vec::new(),
        v_series: Vec::new(),
        diverged: false,
        divergence_info: None,
    };

    let Some(num_steps) = validate_initial_state(&mut traj, preset, dt, total_time, ic_y0, ic_v0)
    else {
        return traj;
    };

    let Some(point_cap) = num_steps.checked_add(1) else {
        return traj;
    };
    if traj.time_series.try_reserve(point_cap).is_err()
        || traj.y_series.try_reserve(point_cap).is_err()
        || (preset == OdePreset::HarmonicOscillator && traj.v_series.try_reserve(point_cap).is_err())
    {
        return traj;
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
        OdePreset::Custom => {
            run_custom_ode(
                &mut traj,
                custom_expr,
                ic_y0,
                param_k,
                param_r,
                param_cap_k,
                method,
                dt,
                num_steps,
            );
        }
    }

    traj
}

fn validate_initial_state(
    traj: &mut SolverTrajectory,
    preset: OdePreset,
    dt: f64,
    total_time: f64,
    ic_y0: f64,
    ic_v0: f64,
) -> Option<usize> {
    let num_steps = crate::presets::checked_ode_step_count(dt, total_time)?;
    if !ic_y0.is_finite() {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, "y(t)".to_string()));
        return None;
    }
    if preset == OdePreset::HarmonicOscillator && !ic_v0.is_finite() {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, "y'(t)".to_string()));
        return None;
    }
    Some(num_steps)
}

#[allow(clippy::too_many_arguments)]
fn run_custom_ode(
    traj: &mut SolverTrajectory,
    expr: Option<&Expr>,
    ic_y0: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    method: IntegrationMethod,
    dt: f64,
    num_steps: usize,
) {
    if let Some(expr) = expr {
        let mut p_map = std::collections::HashMap::new();
        p_map.insert("k".to_string(), param_k);
        p_map.insert("r".to_string(), param_r);
        p_map.insert("cap_k".to_string(), param_cap_k);
        p_map.insert("K".to_string(), param_cap_k);
        let init_state = VecState(vec![ic_y0]);
        let dynamics = DynamicAstOde {
            expr: expr.clone(),
            params: p_map,
        };
        run_step_loop(traj, init_state, dynamics, method, dt, num_steps);
    } else {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, "y(t)".to_string()));
    }
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
        if !y.is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((i as f64 * dt, i, "y(t)".to_string()));
            break;
        }
        if has_v && !model.get_state().0[1].is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((i as f64 * dt, i, "y'(t)".to_string()));
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

impl OdeSolverTool {
    pub(super) fn recalculate_comparison(&mut self) {
        self.comparison_manager.clear();
        if !self.comparison_mode {
            return;
        }

        let custom_expr = self.math_input.expr();
        let rk4 = simulate_preset(
            self.preset,
            IntegrationMethod::RungeKutta4,
            self.dt,
            self.total_time,
            self.param_k,
            self.param_r,
            self.param_cap_k,
            self.ic_y0,
            self.ic_v0,
            custom_expr,
        );
        let euler = simulate_preset(
            self.preset,
            IntegrationMethod::Euler,
            self.dt,
            self.total_time,
            self.param_k,
            self.param_r,
            self.param_cap_k,
            self.ic_y0,
            self.ic_v0,
            custom_expr,
        );
        let heun = simulate_preset(
            self.preset,
            IntegrationMethod::Heun,
            self.dt,
            self.total_time,
            self.param_k,
            self.param_r,
            self.param_cap_k,
            self.ic_y0,
            self.ic_v0,
            custom_expr,
        );

        if rk4.diverged || euler.diverged || heun.diverged {
            self.diverged = true;
        }

        self.discrepancy_euler_rk4 = compute_discrepancy(&euler, &rk4);
        self.discrepancy_heun_rk4 = compute_discrepancy(&heun, &rk4);
        self.discrepancy_euler_heun = compute_discrepancy(&euler, &heun);

        self.populate_comparison_snapshots(&[rk4, euler, heun]);
    }

    pub(super) fn populate_comparison_snapshots(&mut self, trajs: &[SolverTrajectory]) {
        for traj in trajs {
            let pts_y: Vec<[f64; 2]> = traj
                .time_series
                .iter()
                .zip(&traj.y_series)
                .map(|(&t, &y)| [t, y])
                .collect();
            let mut series = vec![TraceSeries::new("y(t)", pts_y)];
            if self.preset == OdePreset::HarmonicOscillator && !traj.v_series.is_empty() {
                let pts_v: Vec<[f64; 2]> = traj
                    .time_series
                    .iter()
                    .zip(&traj.v_series)
                    .map(|(&t, &v)| [t, v])
                    .collect();
                series.push(TraceSeries::new("y'(t)", pts_v));
            }
            self.comparison_manager
                .add_snapshot(traj.method.name(), "", series);
        }
    }
}
