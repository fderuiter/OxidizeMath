#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(test)]
use super::poincare_map::DEFAULT_MATRIX_A;
use super::systems::{
    build_ode_param_map, CustomOdeSpec, ExponentialOde, HarmonicOscillatorOde, Linear2DOde,
    LogisticGrowthOde, PlanarAstOde,
};
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

/// Quantitative discrepancy metrics between two numerical solvers across active state variables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SolverDiscrepancy {
    pub max_error: f64,
    pub rms_error: f64,
}

/// Complete specification for simulating a 1D scalar or 2D coupled/planar ODE run.
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
    pub custom_spec: Option<&'a CustomOdeSpec>,
    pub planar_x_expr: Option<&'a Expr>,
    pub planar_y_expr: Option<&'a Expr>,
}

impl OdeSimSpec<'_> {
    #[must_use]
    pub fn has_second_state(&self) -> bool {
        self.preset.is_2d()
            || (self.preset == OdePreset::Custom
                && self.custom_spec.is_some_and(CustomOdeSpec::is_coupled))
    }

    #[must_use]
    pub fn state_labels(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        if self.preset == OdePreset::Custom
            && self.custom_spec.is_some_and(CustomOdeSpec::is_coupled)
        {
            ("y(t)", "v(t)", "y", "v")
        } else {
            self.preset.state_labels()
        }
    }
}

/// Simulates an ODE preset or custom compiled specification with the specified solver algorithm.
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
    custom_spec: Option<&CustomOdeSpec>,
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
        custom_spec,
        planar_x_expr: None,
        planar_y_expr: None,
    })
}

/// Simulates a 1D or 2D planar/coupled [`OdeSimSpec`] using stack-allocated `ArrayState` buffers
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
    let has_v = spec.has_second_state();
    if traj.time_series.try_reserve(point_cap).is_err()
        || traj.y_series.try_reserve(point_cap).is_err()
        || (has_v && traj.v_series.try_reserve(point_cap).is_err())
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
            if let Some(cspec) = spec.custom_spec {
                if cspec.is_coupled() {
                    run_loop_2d(
                        traj,
                        ArrayState([spec.ic_y0, spec.ic_v0]),
                        cspec,
                        spec.method,
                        spec.dt,
                        num_steps,
                        ("y(t)", "v(t)"),
                    );
                } else {
                    run_loop_1d(traj, init_1d, cspec, spec.method, spec.dt, num_steps);
                }
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
    let (v0_name, v1_name, _, _) = spec.state_labels();
    if !spec.ic_y0.is_finite() {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, v0_name.to_string()));
        return None;
    }
    if spec.has_second_state() && !spec.ic_v0.is_finite() {
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

/// Computes maximum discrepancy and root-mean-square discrepancy between two solver trajectories
/// across `y(t)` and (when present for coupled 2-state systems) `v(t)`.
pub fn compute_discrepancy(a: &SolverTrajectory, b: &SolverTrajectory) -> SolverDiscrepancy {
    let mut max_err: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let mut count: usize = 0;

    accumulate_series_discrepancy(
        &a.y_series,
        &b.y_series,
        &mut max_err,
        &mut sum_sq_err,
        &mut count,
    );
    if !a.v_series.is_empty() && !b.v_series.is_empty() {
        accumulate_series_discrepancy(
            &a.v_series,
            &b.v_series,
            &mut max_err,
            &mut sum_sq_err,
            &mut count,
        );
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

fn accumulate_series_discrepancy(
    sa: &[f64],
    sb: &[f64],
    max_err: &mut f64,
    sum_sq_err: &mut f64,
    count: &mut usize,
) {
    for (&va, &vb) in sa.iter().zip(sb.iter()) {
        if va.is_finite() && vb.is_finite() {
            let err = (va - vb).abs();
            if err > *max_err {
                *max_err = err;
            }
            *sum_sq_err += err * err;
            *count += 1;
        }
    }
}
