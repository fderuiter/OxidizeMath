#![cfg_attr(any(), verified(opt_out = "gui_tool"))]

use super::systems::{CustomOdeSpec, ExponentialOde, HarmonicOscillatorOde, LogisticGrowthOde};
use super::OdeSolverTool;
use crate::framework::TraceSeries;
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
    v_label: &'static str,
}

/// Simulates an ODE preset or custom compiled specification with the specified solver algorithm.
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
        v_label,
    };
    match preset {
        OdePreset::Exponential => {
            run_step_loop(
                &mut traj,
                VecState(vec![ic_y0]),
                ExponentialOde { k: param_k },
                sched,
            );
        }
        OdePreset::HarmonicOscillator => {
            let init = VecState(vec![ic_y0, ic_v0]);
            run_step_loop(&mut traj, init, HarmonicOscillatorOde { k: param_k }, sched);
        }
        OdePreset::LogisticGrowth => {
            let dyn_log = LogisticGrowthOde {
                r: param_r,
                cap_k: param_cap_k,
            };
            run_step_loop(&mut traj, VecState(vec![ic_y0]), dyn_log, sched);
        }
        OdePreset::Custom => run_custom_ode(&mut traj, custom_spec, [ic_y0, ic_v0], sched),
    }
    traj
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

fn run_custom_ode(
    traj: &mut SolverTrajectory,
    spec: Option<&CustomOdeSpec>,
    ic: [f64; 2],
    sched: StepSchedule,
) {
    if let Some(spec) = spec {
        let init_state = if spec.is_coupled() {
            VecState(vec![ic[0], ic[1]])
        } else {
            VecState(vec![ic[0]])
        };
        run_step_loop(traj, init_state, spec.clone(), sched);
    }
}

fn run_step_loop<D: OdeSystem<VecState> + Clone>(
    traj: &mut SolverTrajectory,
    init: VecState,
    dynamics: D,
    sched: StepSchedule,
) {
    match sched.method {
        IntegrationMethod::RungeKutta4 => {
            let solver = RungeKutta4::new(&init);
            let mut model = OdeModel::new(init, dynamics, solver);
            execute_steps(traj, &mut model, sched.dt, sched.num_steps, sched.v_label);
        }
        IntegrationMethod::Euler => {
            let solver = Euler::new(&init);
            let mut model = OdeModel::new(init, dynamics, solver);
            execute_steps(traj, &mut model, sched.dt, sched.num_steps, sched.v_label);
        }
        IntegrationMethod::Heun => {
            let solver = Heun::new(&init);
            let mut model = OdeModel::new(init, dynamics, solver);
            execute_steps(traj, &mut model, sched.dt, sched.num_steps, sched.v_label);
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

impl OdeSolverTool {
    pub(super) fn recalculate_comparison(&mut self) {
        self.comparison_manager.clear();
        if !self.comparison_mode {
            return;
        }

        let custom_spec = self.build_custom_ode_spec();
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
            custom_spec.as_ref(),
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
            custom_spec.as_ref(),
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
            custom_spec.as_ref(),
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
        let has_v = self.has_second_state();
        let v_label = self.second_state_label();
        for traj in trajs {
            let pts_y: Vec<[f64; 2]> = traj
                .time_series
                .iter()
                .zip(&traj.y_series)
                .map(|(&t, &y)| [t, y])
                .collect();
            let mut series = vec![TraceSeries::new("y(t)", pts_y)];
            if has_v && !traj.v_series.is_empty() {
                let pts_v: Vec<[f64; 2]> = traj
                    .time_series
                    .iter()
                    .zip(&traj.v_series)
                    .map(|(&t, &v)| [t, v])
                    .collect();
                series.push(TraceSeries::new(v_label, pts_v));
            }
            self.comparison_manager
                .add_snapshot(traj.method.name(), "", series);
        }
    }
}
