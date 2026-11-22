//! Pre-allocated Runge-Kutta 4 ODE trajectory shooting objective for nonlinear
//! least-squares parameter calibration.

use super::OptimizationError;
use super::least_squares::{LeastSquaresObjective, MAX_CALIBRATION_OBSERVATIONS};
use crate::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, VectorOperations};

/// Default maximum RK4 integration step used between empirical observation timestamps.
pub const DEFAULT_SHOOTING_MAX_DT: f64 = 0.02;

/// Upper bound on RK4 sub-steps per observation interval to prevent unbounded loops.
pub const MAX_SUBSTEPS_PER_INTERVAL: usize = 4_096;

/// Pre-allocated single-shooting ODE residual objective built on [`RungeKutta4`].
///
/// Evaluates $r_k(\theta) = y_{\text{obs}}(t_k) - h(\vec{x}(t_k; \theta))$ across
/// monotonic observation timestamps $t_0 \le t_1 \le \dots \le t_{N-1}$ with zero
/// heap allocations during [`LeastSquaresObjective::eval_residuals`].
pub struct Rk4ShootingObjective<'a, Sys, State, ConfigureFn, ObserveFn, FiniteFn>
where
    State: VectorOperations,
{
    system: Sys,
    solver: RungeKutta4<State>,
    work_state: State,
    times: &'a [f64],
    observed: &'a [f64],
    max_dt: f64,
    configure: ConfigureFn,
    observe: ObserveFn,
    is_finite_state: FiniteFn,
}

impl<'a, Sys, State, ConfigureFn, ObserveFn, FiniteFn>
    Rk4ShootingObjective<'a, Sys, State, ConfigureFn, ObserveFn, FiniteFn>
where
    Sys: OdeSystem<State>,
    State: VectorOperations,
    ConfigureFn: FnMut(&mut Sys, &[f64], f64, &mut State) -> bool,
    ObserveFn: Fn(&State) -> f64,
    FiniteFn: Fn(&State) -> bool,
{
    /// Constructs a pre-allocated RK4 shooting objective over monotonic `times` and `observed`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        system: Sys,
        example_state: &State,
        times: &'a [f64],
        observed: &'a [f64],
        max_dt: f64,
        configure: ConfigureFn,
        observe: ObserveFn,
        is_finite_state: FiniteFn,
    ) -> Result<Self, OptimizationError> {
        validate_shooting_series(times, observed, max_dt)?;
        Ok(Self {
            system,
            solver: RungeKutta4::new(example_state),
            work_state: example_state.clone(),
            times,
            observed,
            max_dt,
            configure,
            observe,
            is_finite_state,
        })
    }

    /// Evaluates the simulated scalar observable $\hat{y}(t_k; \theta)$ at each observation timestamp.
    #[verified_engine::verified]
    pub fn eval_predictions(&mut self, params: &[f64], predictions_out: &mut [f64]) -> bool {
        let n = self.times.len();
        if predictions_out.len() < n || params.iter().any(|p| !p.is_finite()) {
            return false;
        }
        let t0 = self.times[0];
        if !(self.configure)(&mut self.system, params, t0, &mut self.work_state)
            || !(self.is_finite_state)(&self.work_state)
        {
            return false;
        }
        let y0 = (self.observe)(&self.work_state);
        if !y0.is_finite() {
            return false;
        }
        predictions_out[0] = y0;

        for (k, pred_out) in predictions_out.iter_mut().enumerate().take(n).skip(1) {
            let t_prev = self.times[k - 1];
            let t_next = self.times[k];
            if !self.advance_interval(t_prev, t_next) {
                return false;
            }
            let y_hat = (self.observe)(&self.work_state);
            if !y_hat.is_finite() {
                return false;
            }
            *pred_out = y_hat;
        }
        true
    }

    /// Immutable reference to the underlying ODE system.
    pub fn system(&self) -> &Sys {
        &self.system
    }

    fn advance_interval(&mut self, t_prev: f64, t_next: f64) -> bool {
        let span = t_next - t_prev;
        if span <= 0.0 {
            return true;
        }
        let raw_steps = (span / self.max_dt).ceil();
        if !raw_steps.is_finite() {
            return false;
        }
        let substeps = (raw_steps as usize).clamp(1, MAX_SUBSTEPS_PER_INTERVAL);
        let dt = span / (substeps as f64);
        let mut t_cur = t_prev;
        for _ in 0..substeps {
            self.solver
                .step(&self.system, t_cur, &mut self.work_state, dt);
            if !(self.is_finite_state)(&self.work_state) {
                return false;
            }
            t_cur += dt;
        }
        true
    }
}

#[verified_engine::verified]
fn validate_shooting_series(
    times: &[f64],
    observed: &[f64],
    max_dt: f64,
) -> Result<(), OptimizationError> {
    let n = times.len();
    if n != observed.len() || !(2..=MAX_CALIBRATION_OBSERVATIONS).contains(&n) {
        return Err(OptimizationError::InvalidObservationCount {
            count: n,
            min: 2,
            max: MAX_CALIBRATION_OBSERVATIONS,
        });
    }
    if !max_dt.is_finite() || max_dt <= 0.0 {
        return Err(OptimizationError::NumericalInstability);
    }
    for i in 0..n {
        if !times[i].is_finite() || !observed[i].is_finite() {
            return Err(OptimizationError::NumericalInstability);
        }
        if i > 0 && times[i] < times[i - 1] {
            return Err(OptimizationError::NumericalInstability);
        }
    }
    Ok(())
}

impl<Sys, State, ConfigureFn, ObserveFn, FiniteFn> LeastSquaresObjective
    for Rk4ShootingObjective<'_, Sys, State, ConfigureFn, ObserveFn, FiniteFn>
where
    Sys: OdeSystem<State>,
    State: VectorOperations,
    ConfigureFn: FnMut(&mut Sys, &[f64], f64, &mut State) -> bool,
    ObserveFn: Fn(&State) -> f64,
    FiniteFn: Fn(&State) -> bool,
{
    fn observation_count(&self) -> usize {
        self.observed.len()
    }

    fn copy_observed(&self, observed_out: &mut [f64]) {
        let n = self.observed.len().min(observed_out.len());
        observed_out[..n].copy_from_slice(&self.observed[..n]);
    }

    fn eval_residuals(&mut self, params: &[f64], residuals_out: &mut [f64]) -> bool {
        let n = self.observed.len();
        if !self.eval_predictions(params, residuals_out) {
            return false;
        }
        for (&obs_y, res_slot) in self.observed.iter().zip(residuals_out.iter_mut()).take(n) {
            let res = obs_y - *res_slot;
            if !res.is_finite() {
                return false;
            }
            *res_slot = res;
        }
        true
    }
}

#[cfg(test)]
#[path = "ode_shooting_tests.rs"]
mod tests;
