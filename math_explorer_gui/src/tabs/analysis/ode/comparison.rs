#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

#[cfg(test)]
use super::poincare_map::DEFAULT_MATRIX_A;
use super::systems::{
    build_ode_param_map, CustomOdeSpec, ExponentialOde, HarmonicOscillatorOde, Linear2DOde,
    LogisticGrowthOde, PlanarAstOde,
};
use super::{DEFAULT_ODE_ADAPTIVE_TOL, MAX_ODE_ADAPTIVE_TOL, MIN_ODE_ADAPTIVE_TOL};
use crate::presets::OdePreset;
use math_commons::ast::Expr;
use math_explorer::pure_math::analysis::ode::{
    ArrayState, BackwardEuler, DormandPrince45, Euler, Heun, ImplicitTrapezoidal, IndexedState,
    IntegrationMethod, OdeSystem, Rosenbrock23, RungeKutta4, Solver,
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
    pub adaptive_tol: f64,
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

pub(super) fn sanitize_adaptive_tol(adaptive_tol: f64) -> f64 {
    if adaptive_tol.is_finite() {
        adaptive_tol.clamp(MIN_ODE_ADAPTIVE_TOL, MAX_ODE_ADAPTIVE_TOL)
    } else {
        DEFAULT_ODE_ADAPTIVE_TOL
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
        adaptive_tol: DEFAULT_ODE_ADAPTIVE_TOL,
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
    let names_1d = ("y(t)", "v(t)");
    match spec.preset {
        OdePreset::Exponential => {
            let dyn_sys = ExponentialOde { k: spec.param_k };
            run_loop_nd(traj, init_1d, &dyn_sys, spec, num_steps, names_1d);
        }
        OdePreset::LogisticGrowth => {
            let dyn_sys = LogisticGrowthOde {
                r: spec.param_r,
                cap_k: spec.param_cap_k,
            };
            run_loop_nd(traj, init_1d, &dyn_sys, spec, num_steps, names_1d);
        }
        OdePreset::Custom => {
            if let Some(cspec) = spec.custom_spec {
                if cspec.is_coupled() {
                    let init_2d = ArrayState([spec.ic_y0, spec.ic_v0]);
                    run_loop_nd(traj, init_2d, cspec, spec, num_steps, ("y(t)", "v(t)"));
                } else {
                    run_loop_nd(traj, init_1d, cspec, spec, num_steps, names_1d);
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
            run_loop_nd(traj, init_2d, &dyn_sys, spec, num_steps, ("y(t)", "y'(t)"));
        }
        OdePreset::LinearSystem2D => {
            let dyn_sys = Linear2DOde {
                matrix_a: spec.matrix_a,
            };
            run_loop_nd(traj, init_2d, &dyn_sys, spec, num_steps, ("x(t)", "y(t)"));
        }
        OdePreset::PlanarCustom => {
            if let (Some(x_expr), Some(y_expr)) = (spec.planar_x_expr, spec.planar_y_expr) {
                let dyn_sys = PlanarAstOde {
                    x_expr,
                    y_expr,
                    params: build_ode_param_map(spec.param_k, spec.param_r, spec.param_cap_k),
                };
                run_loop_nd(traj, init_2d, &dyn_sys, spec, num_steps, ("x(t)", "y(t)"));
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

fn run_loop_nd<const N: usize, D>(
    traj: &mut SolverTrajectory,
    init: ArrayState<N>,
    dynamics: &D,
    spec: &OdeSimSpec<'_>,
    num_steps: usize,
    names: (&str, &str),
) where
    D: OdeSystem<ArrayState<N>>,
    ArrayState<N>: IndexedState,
{
    let rtol = sanitize_adaptive_tol(spec.adaptive_tol);
    let atol = (rtol * 0.01).max(1e-14);
    let dt = spec.dt;
    match spec.method {
        IntegrationMethod::RungeKutta4 => {
            let mut s = RungeKutta4::new(&init);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
        IntegrationMethod::Euler => {
            let mut s = Euler::new(&init);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
        IntegrationMethod::Heun => {
            let mut s = Heun::new(&init);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
        IntegrationMethod::DormandPrince45 => {
            let mut s = DormandPrince45::new(&init).with_tolerances(rtol, atol);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
        IntegrationMethod::BackwardEuler => {
            let mut s = BackwardEuler::new(&init);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
        IntegrationMethod::ImplicitTrapezoidal => {
            let mut s = ImplicitTrapezoidal::new(&init);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
        IntegrationMethod::Rosenbrock23 => {
            let mut s = Rosenbrock23::new(&init).with_tolerances(rtol, atol);
            execute_steps_nd(traj, init, dynamics, &mut s, dt, num_steps, names);
        }
    }
}

fn execute_steps_nd<const N: usize, D: OdeSystem<ArrayState<N>>, S: Solver<ArrayState<N>>>(
    traj: &mut SolverTrajectory,
    mut state: ArrayState<N>,
    dynamics: &D,
    solver: &mut S,
    dt: f64,
    num_steps: usize,
    names: (&str, &str),
) {
    for i in 0..=num_steps {
        let t = i as f64 * dt;
        let x = state.0[0];
        if !x.is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((t, i, names.0.to_string()));
            break;
        }
        if N >= 2 && !state.0[1].is_finite() {
            traj.diverged = true;
            traj.divergence_info = Some((t, i, names.1.to_string()));
            break;
        }
        traj.time_series.push(t);
        traj.y_series.push(x);
        if N >= 2 {
            traj.v_series.push(state.0[1]);
        }
        solver.step(dynamics, t, &mut state, dt);
    }
}

/// Computes maximum discrepancy and root-mean-square discrepancy between two solver trajectories
/// across `y(t)` and (when present for coupled 2-state systems) `v(t)`.
#[cfg(test)]
#[must_use]
pub fn compute_discrepancy(a: &SolverTrajectory, b: &SolverTrajectory) -> SolverDiscrepancy {
    compute_series_discrepancy(&a.y_series, &a.v_series, &b.y_series, &b.v_series)
}

/// Computes maximum discrepancy and root-mean-square discrepancy between slice-backed `(y, v)` series.
#[must_use]
pub fn compute_series_discrepancy(
    a_y: &[f64],
    a_v: &[f64],
    b_y: &[f64],
    b_v: &[f64],
) -> SolverDiscrepancy {
    let mut acc = (0.0_f64, 0.0_f64, 0_usize);
    accumulate_series_discrepancy(a_y, b_y, &mut acc);
    if !a_v.is_empty() && !b_v.is_empty() {
        accumulate_series_discrepancy(a_v, b_v, &mut acc);
    }
    SolverDiscrepancy {
        max_error: acc.0,
        rms_error: if acc.2 > 0 {
            (acc.1 / acc.2 as f64).sqrt()
        } else {
            0.0
        },
    }
}

fn accumulate_series_discrepancy(sa: &[f64], sb: &[f64], acc: &mut (f64, f64, usize)) {
    for (&va, &vb) in sa.iter().zip(sb.iter()) {
        if va.is_finite() && vb.is_finite() {
            let err = (va - vb).abs();
            acc.0 = acc.0.max(err);
            acc.1 += err * err;
            acc.2 += 1;
        }
    }
}
pub(super) struct TrajectoryView<'a> {
    pub(super) method: IntegrationMethod,
    pub(super) time_series: &'a [f64],
    pub(super) y_series: &'a [f64],
    pub(super) v_series: &'a [f64],
    pub(super) diverged: bool,
    pub(super) is_active_reused: bool,
}

impl TrajectoryView<'_> {
    pub(super) fn discrepancy(&self, other: &Self) -> SolverDiscrepancy {
        compute_series_discrepancy(self.y_series, self.v_series, other.y_series, other.v_series)
    }
}

pub(super) fn build_view_trace_series(
    view: &TrajectoryView<'_>,
    labels: (&'static str, &'static str, &'static str, &'static str),
    has_v: bool,
    phase_active: bool,
) -> Vec<crate::framework::TraceSeries> {
    use crate::framework::TraceSeries;

    let (t1_lbl, t2_lbl, px_lbl, py_lbl) = labels;
    if phase_active {
        let pts_phase: Vec<[f64; 2]> = view
            .y_series
            .iter()
            .zip(view.v_series)
            .map(|(&x, &y)| [x, y])
            .collect();
        return vec![TraceSeries::new(format!("({px_lbl}, {py_lbl})"), pts_phase)];
    }
    let pts_y: Vec<[f64; 2]> = view
        .time_series
        .iter()
        .zip(view.y_series)
        .map(|(&t, &y)| [t, y])
        .collect();
    let mut series = vec![TraceSeries::new(t1_lbl, pts_y)];
    if has_v && !view.v_series.is_empty() {
        let pts_v: Vec<[f64; 2]> = view
            .time_series
            .iter()
            .zip(view.v_series)
            .map(|(&t, &v)| [t, v])
            .collect();
        series.push(TraceSeries::new(t2_lbl, pts_v));
    }
    series
}
