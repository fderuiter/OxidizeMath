#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::systems::{CustomOdeSpec, ExponentialOde, HarmonicOscillatorOde, LogisticGrowthOde};
use super::{OdeSolverTool, DEFAULT_ODE_ADAPTIVE_TOL, MAX_ODE_ADAPTIVE_TOL, MIN_ODE_ADAPTIVE_TOL};
use crate::framework::TraceSeries;
use crate::presets::OdePreset;
use math_explorer::pure_math::analysis::ode::{
    BackwardEuler, DormandPrince45, Euler, Heun, ImplicitTrapezoidal, IntegrationMethod, OdeModel,
    OdeSystem, Rosenbrock23, RungeKutta4, Solver, TimeStepper, VecState,
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

#[derive(Clone, Copy)]
struct StepSchedule {
    method: IntegrationMethod,
    dt: f64,
    num_steps: usize,
    adaptive_tol: f64,
    v_label: &'static str,
}

/// Simulates an ODE preset or custom compiled specification with default adaptive tolerances.
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
    simulate_preset_with_tol(
        preset,
        method,
        dt,
        total_time,
        DEFAULT_ODE_ADAPTIVE_TOL,
        param_k,
        param_r,
        param_cap_k,
        ic_y0,
        ic_v0,
        custom_spec,
    )
}

/// Simulates an ODE preset or custom compiled specification with explicit adaptive tolerance `adaptive_tol`.
#[allow(clippy::too_many_arguments)]
pub fn simulate_preset_with_tol(
    preset: OdePreset,
    method: IntegrationMethod,
    dt: f64,
    total_time: f64,
    adaptive_tol: f64,
    param_k: f64,
    param_r: f64,
    param_cap_k: f64,
    ic_y0: f64,
    ic_v0: f64,
    custom_spec: Option<&CustomOdeSpec>,
) -> SolverTrajectory {
    let mut traj = SolverTrajectory {
        method,
        ..Default::default()
    };
    let has_second_state = preset == OdePreset::HarmonicOscillator
        || (preset == OdePreset::Custom && custom_spec.is_some_and(CustomOdeSpec::is_coupled));
    let v_label = if preset == OdePreset::HarmonicOscillator {
        "y'(t)"
    } else {
        "v(t)"
    };
    let Some(num_steps) = validate_initial_state(
        &mut traj,
        has_second_state,
        v_label,
        dt,
        total_time,
        ic_y0,
        ic_v0,
    ) else {
        return traj;
    };
    if !reserve_trajectory_capacity(&mut traj, num_steps, has_second_state) {
        return traj;
    }
    let sched = StepSchedule {
        method,
        dt,
        num_steps,
        adaptive_tol: sanitize_adaptive_tol(adaptive_tol),
        v_label,
    };
    dispatch_preset_simulation(
        &mut traj,
        preset,
        [param_k, param_r, param_cap_k],
        [ic_y0, ic_v0],
        custom_spec,
        sched,
    );
    traj
}

pub(super) fn sanitize_adaptive_tol(adaptive_tol: f64) -> f64 {
    if adaptive_tol.is_finite() {
        adaptive_tol.clamp(MIN_ODE_ADAPTIVE_TOL, MAX_ODE_ADAPTIVE_TOL)
    } else {
        DEFAULT_ODE_ADAPTIVE_TOL
    }
}

fn dispatch_preset_simulation(
    traj: &mut SolverTrajectory,
    preset: OdePreset,
    [k, r, cap_k]: [f64; 3],
    [y0, v0]: [f64; 2],
    custom_spec: Option<&CustomOdeSpec>,
    sched: StepSchedule,
) {
    match preset {
        OdePreset::Exponential => {
            run_step_loop(traj, VecState(vec![y0]), ExponentialOde { k }, sched)
        }
        OdePreset::HarmonicOscillator => run_step_loop(
            traj,
            VecState(vec![y0, v0]),
            HarmonicOscillatorOde { k },
            sched,
        ),
        OdePreset::LogisticGrowth => run_step_loop(
            traj,
            VecState(vec![y0]),
            LogisticGrowthOde { r, cap_k },
            sched,
        ),
        OdePreset::Custom => {
            if let Some(spec) = custom_spec {
                let init = if spec.is_coupled() {
                    vec![y0, v0]
                } else {
                    vec![y0]
                };
                run_step_loop(traj, VecState(init), spec.clone(), sched);
            }
        }
    }
}

fn reserve_trajectory_capacity(
    traj: &mut SolverTrajectory,
    num_steps: usize,
    has_second_state: bool,
) -> bool {
    let Some(point_cap) = num_steps.checked_add(1) else {
        return false;
    };
    traj.time_series.try_reserve(point_cap).is_ok()
        && traj.y_series.try_reserve(point_cap).is_ok()
        && (!has_second_state || traj.v_series.try_reserve(point_cap).is_ok())
}

fn validate_initial_state(
    traj: &mut SolverTrajectory,
    has_second_state: bool,
    v_label: &str,
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
    if has_second_state && !ic_v0.is_finite() {
        traj.diverged = true;
        traj.divergence_info = Some((0.0, 0, v_label.to_string()));
        return None;
    }
    Some(num_steps)
}

fn run_with_solver<D: OdeSystem<VecState>, S: Solver<VecState>>(
    traj: &mut SolverTrajectory,
    init: VecState,
    dynamics: D,
    solver: S,
    sched: StepSchedule,
) {
    let mut model = OdeModel::new(init, dynamics, solver);
    execute_steps(traj, &mut model, sched.dt, sched.num_steps, sched.v_label);
}

fn run_step_loop<D: OdeSystem<VecState> + Clone>(
    traj: &mut SolverTrajectory,
    init: VecState,
    dynamics: D,
    sched: StepSchedule,
) {
    let rtol = sched.adaptive_tol;
    let atol = (rtol * 0.01).max(1e-14);
    match sched.method {
        IntegrationMethod::RungeKutta4 => {
            let s = RungeKutta4::new(&init);
            run_with_solver(traj, init, dynamics, s, sched);
        }
        IntegrationMethod::Euler => {
            let s = Euler::new(&init);
            run_with_solver(traj, init, dynamics, s, sched);
        }
        IntegrationMethod::Heun => {
            let s = Heun::new(&init);
            run_with_solver(traj, init, dynamics, s, sched);
        }
        IntegrationMethod::DormandPrince45 => {
            let s = DormandPrince45::new(&init).with_tolerances(rtol, atol);
            run_with_solver(traj, init, dynamics, s, sched);
        }
        IntegrationMethod::BackwardEuler => {
            let s = BackwardEuler::new(&init);
            run_with_solver(traj, init, dynamics, s, sched);
        }
        IntegrationMethod::ImplicitTrapezoidal => {
            let s = ImplicitTrapezoidal::new(&init);
            run_with_solver(traj, init, dynamics, s, sched);
        }
        IntegrationMethod::Rosenbrock23 => {
            let s = Rosenbrock23::new(&init).with_tolerances(rtol, atol);
            run_with_solver(traj, init, dynamics, s, sched);
        }
    }
}

fn execute_steps<D, S>(
    traj: &mut SolverTrajectory,
    model: &mut OdeModel<VecState, D, S>,
    dt: f64,
    num_steps: usize,
    v_label: &str,
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
            traj.divergence_info = Some((i as f64 * dt, i, v_label.to_string()));
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

struct TrajectoryView<'a> {
    method: IntegrationMethod,
    time_series: &'a [f64],
    y_series: &'a [f64],
    v_series: &'a [f64],
    diverged: bool,
    is_active_reused: bool,
}

impl TrajectoryView<'_> {
    fn discrepancy(&self, other: &Self) -> SolverDiscrepancy {
        compute_series_discrepancy(self.y_series, self.v_series, other.y_series, other.v_series)
    }
}

fn build_view_trace_series(
    view: &TrajectoryView<'_>,
    has_v: bool,
    v_label: &str,
) -> Vec<TraceSeries> {
    let zip = |vals: &[f64]| -> Vec<[f64; 2]> {
        view.time_series
            .iter()
            .zip(vals)
            .map(|(&t, &v)| [t, v])
            .collect()
    };
    let mut series = vec![TraceSeries::new("y(t)", zip(view.y_series))];
    if has_v && !view.v_series.is_empty() {
        series.push(TraceSeries::new(v_label, zip(view.v_series)));
    }
    series
}

impl OdeSolverTool {
    fn simulate_if_inactive(
        &self,
        target: IntegrationMethod,
        reuse_active: bool,
        custom_spec: Option<&CustomOdeSpec>,
    ) -> Option<SolverTrajectory> {
        (!reuse_active || self.method != target).then(|| {
            simulate_preset_with_tol(
                self.preset,
                target,
                self.dt,
                self.total_time,
                self.adaptive_tol,
                self.param_k,
                self.param_r,
                self.param_cap_k,
                self.ic_y0,
                self.ic_v0,
                custom_spec,
            )
        })
    }

    fn trajectory_view<'a>(
        &'a self,
        target: IntegrationMethod,
        simulated: &'a Option<SolverTrajectory>,
    ) -> TrajectoryView<'a> {
        if let Some(traj) = simulated {
            TrajectoryView {
                method: traj.method,
                time_series: &traj.time_series,
                y_series: &traj.y_series,
                v_series: &traj.v_series,
                diverged: traj.diverged,
                is_active_reused: false,
            }
        } else {
            TrajectoryView {
                method: target,
                time_series: &self.time_series,
                y_series: &self.y_series,
                v_series: &self.v_series,
                diverged: self.diverged,
                is_active_reused: true,
            }
        }
    }

    fn build_comparison_snapshot_series(
        &self,
        views: &[TrajectoryView<'_>],
    ) -> Vec<(&'static str, Vec<TraceSeries>)> {
        let has_v = self.has_second_state();
        let v_label = self.second_state_label();
        let cache_matches =
            self.series_cache
                .matches_series(&self.time_series, &self.y_series, &self.v_series);
        views
            .iter()
            .map(|view| {
                let series = if view.is_active_reused && cache_matches {
                    let (py, pv) = self.series_cache.shared_points();
                    let mut s = vec![TraceSeries::new("y(t)", py)];
                    if has_v && !pv.is_empty() {
                        s.push(TraceSeries::new(v_label, pv));
                    }
                    s
                } else {
                    build_view_trace_series(view, has_v, v_label)
                };
                (view.method.name(), series)
            })
            .collect()
    }

    pub(super) fn recalculate_comparison(&mut self) {
        self.comparison_manager.clear();
        if !self.comparison_mode {
            return;
        }

        let custom_spec = self.build_custom_ode_spec();
        let reuse_active =
            !self.has_custom_input_error() || !self.time_series.is_empty() || self.diverged;
        let sims = IntegrationMethod::ALL
            .map(|m| self.simulate_if_inactive(m, reuse_active, custom_spec.as_ref()));

        let (any_diverged, euler_rk4, heun_rk4, euler_heun, rk4_dp45, be_ros23, snapshots) = {
            let views: [TrajectoryView<'_>; 7] = std::array::from_fn(|idx| {
                self.trajectory_view(IntegrationMethod::ALL[idx], &sims[idx])
            });
            // Order matches IntegrationMethod::ALL: [RK4, Euler, Heun, DP45, BE, Trap, Ros23]
            let snapshots = self.build_comparison_snapshot_series(&views);
            (
                views.iter().any(|v| v.diverged),
                views[1].discrepancy(&views[0]),
                views[2].discrepancy(&views[0]),
                views[1].discrepancy(&views[2]),
                views[0].discrepancy(&views[3]),
                views[4].discrepancy(&views[6]),
                snapshots,
            )
        };

        if any_diverged {
            self.diverged = true;
        }
        self.discrepancy_euler_rk4 = euler_rk4;
        self.discrepancy_heun_rk4 = heun_rk4;
        self.discrepancy_euler_heun = euler_heun;
        self.discrepancy_rk4_dp45 = rk4_dp45;
        self.discrepancy_be_ros23 = be_ros23;
        for (label, series) in snapshots {
            self.comparison_manager.add_snapshot(label, "", series);
        }
    }
}
