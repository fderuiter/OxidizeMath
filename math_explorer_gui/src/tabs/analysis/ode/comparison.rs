#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(test)]
use super::poincare_map::DEFAULT_MATRIX_A;
use super::systems::{
    build_ode_param_map, DynamicAstOde, ExponentialOde, HarmonicOscillatorOde, Linear2DOde,
    LogisticGrowthOde, PlanarAstOde,
};
use super::OdeSolverTool;
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{
    ArrayState, Euler, Heun, IntegrationMethod, OdeSystem, RungeKutta4, Solver,
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

/// Complete specification for simulating a 1D scalar or 2D planar ODE run.
pub struct OdeSimSpec<'a> {
    pub preset: OdePreset,
    pub method: IntegrationMethod,
    pub dt: f64,
    pub total_time: f64,
    pub param_k: f64,
    pub param_r: f64,
    pub param_cap_k: f64,
    pub ic_y0: f64,
    pub ic_v0: f64,
    pub matrix_a: [[f64; 2]; 2],
    pub custom_expr: Option<&'a Expr>,
    pub planar_x_expr: Option<&'a Expr>,
    pub planar_y_expr: Option<&'a Expr>,
}

/// Simulates an ODE preset with the specified numerical solver algorithm.
#[cfg(test)]
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
    simulate_ode(&OdeSimSpec {
        preset,
        method,
        dt,
        total_time,
        param_k,
        param_r,
        param_cap_k,
        ic_y0,
        ic_v0,
        matrix_a: DEFAULT_MATRIX_A,
        custom_expr,
        planar_x_expr: None,
        planar_y_expr: None,
    })
}

/// Simulates a 1D or 2D planar [`OdeSimSpec`] using stack-allocated `ArrayState` buffers
/// and explicit step time $t_i = i \cdot \Delta t$.
#[must_use]
pub fn simulate_ode(spec: &OdeSimSpec<'_>) -> SolverTrajectory {
    let mut traj = SolverTrajectory {
        method: spec.method,
        ..Default::default()
    };

    let Some(num_steps) = validate_initial_state(&mut traj, spec) else {
        return traj;
    };
    let Some(point_cap) = num_steps.checked_add(1) else {
        return traj;
    };
    if traj.time_series.try_reserve(point_cap).is_err()
        || traj.y_series.try_reserve(point_cap).is_err()
        || (spec.preset.is_2d() && traj.v_series.try_reserve(point_cap).is_err())
    {
        return traj;
    }

    dispatch_sim_preset(&mut traj, spec, num_steps);
    traj
}

fn dispatch_sim_preset(traj: &mut SolverTrajectory, spec: &OdeSimSpec<'_>, num_steps: usize) {
    if spec.preset.is_2d() {
        dispatch_sim_2d(traj, spec, num_steps);
        return;
    }
    let init_1d = ArrayState([spec.ic_y0]);
    match spec.preset {
        OdePreset::Exponential => {
            let dyn_sys = ExponentialOde { k: spec.param_k };
            run_loop_1d(traj, init_1d, &dyn_sys, spec.method, spec.dt, num_steps);
        }
        OdePreset::LogisticGrowth => {
            let dyn_sys = LogisticGrowthOde {
                r: spec.param_r,
                cap_k: spec.param_cap_k,
            };
            run_loop_1d(traj, init_1d, &dyn_sys, spec.method, spec.dt, num_steps);
        }
        OdePreset::Custom => {
            if let Some(expr) = spec.custom_expr {
                let dyn_sys = DynamicAstOde {
                    expr,
                    params: build_ode_param_map(spec.param_k, spec.param_r, spec.param_cap_k),
                };
                run_loop_1d(traj, init_1d, &dyn_sys, spec.method, spec.dt, num_steps);
            }
        }
        OdePreset::HarmonicOscillator | OdePreset::LinearSystem2D | OdePreset::PlanarCustom => {}
    }
}

fn dispatch_sim_2d(traj: &mut SolverTrajectory, spec: &OdeSimSpec<'_>, num_steps: usize) {
    let init_2d = ArrayState([spec.ic_y0, spec.ic_v0]);
    match spec.preset {
        OdePreset::HarmonicOscillator => {
            let dyn_sys = HarmonicOscillatorOde { k: spec.param_k };
            run_loop_2d(
                traj,
                init_2d,
                &dyn_sys,
                spec.method,
                spec.dt,
                num_steps,
                ("y(t)", "y'(t)"),
            );
        }
        OdePreset::LinearSystem2D => {
            let dyn_sys = Linear2DOde {
                matrix_a: spec.matrix_a,
            };
            run_loop_2d(
                traj,
                init_2d,
                &dyn_sys,
                spec.method,
                spec.dt,
                num_steps,
                ("x(t)", "y(t)"),
            );
        }
        OdePreset::PlanarCustom => {
            if let (Some(x_expr), Some(y_expr)) = (spec.planar_x_expr, spec.planar_y_expr) {
                let dyn_sys = PlanarAstOde {
                    x_expr,
                    y_expr,
                    params: build_ode_param_map(spec.param_k, spec.param_r, spec.param_cap_k),
                };
                run_loop_2d(
                    traj,
                    init_2d,
                    &dyn_sys,
                    spec.method,
                    spec.dt,
                    num_steps,
                    ("x(t)", "y(t)"),
                );
            }
        }
        OdePreset::Exponential | OdePreset::LogisticGrowth | OdePreset::Custom => {}
    }
}

fn validate_initial_state(traj: &mut SolverTrajectory, spec: &OdeSimSpec<'_>) -> Option<usize> {
    let num_steps = crate::presets::checked_ode_step_count(spec.dt, spec.total_time)?;
    let (v0_name, v1_name, _, _) = spec.preset.state_labels();
    if !spec.ic_y0.is_finite() {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, v0_name.to_string()));
        return None;
    }
    if spec.preset.is_2d() && !spec.ic_v0.is_finite() {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, v1_name.to_string()));
        return None;
    }
    if spec.preset == OdePreset::LinearSystem2D
        && !spec.matrix_a.iter().flatten().all(|v| v.is_finite())
    {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, "A".to_string()));
        return None;
    }
    Some(num_steps)
}

fn run_loop_1d<D: OdeSystem<ArrayState<1>>>(
    traj: &mut SolverTrajectory,
    init: ArrayState<1>,
    dynamics: &D,
    method: IntegrationMethod,
    dt: f64,
    num_steps: usize,
) {
    match method {
        IntegrationMethod::RungeKutta4 => {
            execute_steps_1d(
                traj,
                init,
                dynamics,
                &mut RungeKutta4::new(&init),
                dt,
                num_steps,
            );
        }
        IntegrationMethod::Euler => {
            execute_steps_1d(traj, init, dynamics, &mut Euler::new(&init), dt, num_steps);
        }
        IntegrationMethod::Heun => {
            execute_steps_1d(traj, init, dynamics, &mut Heun::new(&init), dt, num_steps);
        }
    }
}

fn execute_steps_1d<D: OdeSystem<ArrayState<1>>, S: Solver<ArrayState<1>>>(
    traj: &mut SolverTrajectory,
    mut state: ArrayState<1>,
    dynamics: &D,
    solver: &mut S,
    dt: f64,
    num_steps: usize,
) {
    for i in 0..=num_steps {
        let t = i as f64 * dt;
        let y = state.0[0];
        if !y.is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((t, i, "y(t)".to_string()));
            break;
        }
        traj.time_series.push(t);
        traj.y_series.push(y);
        solver.step(dynamics, t, &mut state, dt);
    }
}

fn run_loop_2d<D: OdeSystem<ArrayState<2>>>(
    traj: &mut SolverTrajectory,
    init: ArrayState<2>,
    dynamics: &D,
    method: IntegrationMethod,
    dt: f64,
    num_steps: usize,
    names: (&str, &str),
) {
    match method {
        IntegrationMethod::RungeKutta4 => {
            execute_steps_2d(
                traj,
                init,
                dynamics,
                &mut RungeKutta4::new(&init),
                dt,
                num_steps,
                names,
            );
        }
        IntegrationMethod::Euler => {
            execute_steps_2d(
                traj,
                init,
                dynamics,
                &mut Euler::new(&init),
                dt,
                num_steps,
                names,
            );
        }
        IntegrationMethod::Heun => {
            execute_steps_2d(
                traj,
                init,
                dynamics,
                &mut Heun::new(&init),
                dt,
                num_steps,
                names,
            );
        }
    }
}

fn execute_steps_2d<D: OdeSystem<ArrayState<2>>, S: Solver<ArrayState<2>>>(
    traj: &mut SolverTrajectory,
    mut state: ArrayState<2>,
    dynamics: &D,
    solver: &mut S,
    dt: f64,
    num_steps: usize,
    names: (&str, &str),
) {
    for i in 0..=num_steps {
        let t = i as f64 * dt;
        let [x, y] = state.0;
        if !x.is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((t, i, names.0.to_string()));
            break;
        }
        if !y.is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((t, i, names.1.to_string()));
            break;
        }
        traj.time_series.push(t);
        traj.y_series.push(x);
        traj.v_series.push(y);
        solver.step(dynamics, t, &mut state, dt);
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
    pub(super) fn sim_spec(&self, method: IntegrationMethod) -> OdeSimSpec<'_> {
        OdeSimSpec {
            preset: self.preset,
            method,
            dt: self.dt,
            total_time: self.total_time,
            param_k: self.param_k,
            param_r: self.param_r,
            param_cap_k: self.param_cap_k,
            ic_y0: self.ic_y0,
            ic_v0: self.ic_v0,
            matrix_a: self.matrix_a,
            custom_expr: self.math_input.cached_expr(),
            planar_x_expr: self.planar_x_input.cached_expr(),
            planar_y_expr: self.planar_y_input.cached_expr(),
        }
    }

    pub(super) fn recalculate_comparison(&mut self) {
        self.comparison_manager.clear();
        if !self.comparison_mode {
            return;
        }

        let rk4 = simulate_ode(&self.sim_spec(IntegrationMethod::RungeKutta4));
        let euler = simulate_ode(&self.sim_spec(IntegrationMethod::Euler));
        let heun = simulate_ode(&self.sim_spec(IntegrationMethod::Heun));

        if rk4.diverged || euler.diverged || heun.diverged {
            self.diverged = true;
        }

        self.discrepancy_euler_rk4 = compute_discrepancy(&euler, &rk4);
        self.discrepancy_heun_rk4 = compute_discrepancy(&heun, &rk4);
        self.discrepancy_euler_heun = compute_discrepancy(&euler, &heun);

        self.populate_comparison_snapshots(&[rk4, euler, heun]);
    }

    pub(super) fn populate_comparison_snapshots(&mut self, trajs: &[SolverTrajectory]) {
        let (t1_lbl, t2_lbl, px_lbl, py_lbl) = self.preset.state_labels();
        let phase_active = self.is_phase_plane_active();
        for traj in trajs {
            let series = if phase_active {
                let pts_phase: Vec<[f64; 2]> = traj
                    .y_series
                    .iter()
                    .zip(&traj.v_series)
                    .map(|(&x, &y)| [x, y])
                    .collect();
                vec![TraceSeries::new(format!("({px_lbl}, {py_lbl})"), pts_phase)]
            } else {
                let pts_y: Vec<[f64; 2]> = traj
                    .time_series
                    .iter()
                    .zip(&traj.y_series)
                    .map(|(&t, &y)| [t, y])
                    .collect();
                let mut out = vec![TraceSeries::new(t1_lbl, pts_y)];
                if self.preset.is_2d() && !traj.v_series.is_empty() {
                    let pts_v: Vec<[f64; 2]> = traj
                        .time_series
                        .iter()
                        .zip(&traj.v_series)
                        .map(|(&t, &v)| [t, v])
                        .collect();
                    out.push(TraceSeries::new(t2_lbl, pts_v));
                }
                out
            };
            self.comparison_manager
                .add_snapshot(traj.method.name(), "", series);
        }
    }
}
