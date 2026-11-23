//! Variable-step two-step `Bdf2` stiff ODE solver bootstrapped with L-stable second-order TR-BDF2.
// [cite:engine_diffeq]

use super::stiff_common::{
    DEFAULT_STIFF_MAT_CAP, DEFAULT_STIFF_MAX_DIM, StatePair, StiffScratch,
    build_and_factor_stage_matrix,
};
use super::traits::{OdeState, OdeSystem, Solver};
use crate::error::DiffeqError;
use crate::workspace::{SolverWorkspace, lu_solve_in_place};

/// TR-BDF2 stage parameter $\gamma = 2 - \sqrt{2}$.
const TRBDF2_GAMMA: f64 = 0.585_786_437_626_904_9;
/// TR-BDF2 weight $w_1 = \frac{1}{\gamma(2 - \gamma)} = \frac{1 + \sqrt{2}}{2}$.
const TRBDF2_W1: f64 = 1.207_106_781_186_547_5;
/// TR-BDF2 weight $w_0 = -\frac{(1 - \gamma)^2}{\gamma(2 - \gamma)} = 1 - w_1$.
const TRBDF2_W0: f64 = -0.207_106_781_186_547_5;

const MAX_NEWTON_ITERS: usize = 12;

struct ImplicitTarget<'a, State> {
    t_next: f64,
    coeff_c: f64,
    anchor: &'a State,
}

struct NewtonStates<'a, State> {
    f_eval: &'a mut State,
    cand: &'a mut State,
    work_state: &'a mut State,
    work_f: &'a mut State,
}

/// Two-step Backward Differentiation Formula (`Bdf2`) stiff solver using damped Newton-LU iterations.
///
/// Step 1 (or any step following [`Self::reset`] or a non-contiguous time jump) executes an L-stable
/// second-order TR-BDF2 composite step so quadratic trajectories and stiff decays are both integrated
/// accurately from $t_0$ without order loss.
#[derive(Debug, Clone)]
pub struct Bdf2<State> {
    prev_state: State,
    anchor: State,
    f_eval: State,
    cand_state: State,
    work_state: State,
    work_f: State,
    w_lu: [f64; DEFAULT_STIFF_MAT_CAP],
    vecs: [f64; 4 * DEFAULT_STIFF_MAX_DIM],
    pivots: [usize; DEFAULT_STIFF_MAX_DIM],
    prev_dt: f64,
    expected_t: f64,
    has_history: bool,
}

impl<State: OdeState> Bdf2<State> {
    /// Creates a new `Bdf2` solver pre-initializing history and Newton scratch buffers from `template`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            prev_state: template.clone(),
            anchor: template.clone(),
            f_eval: template.clone(),
            cand_state: template.clone(),
            work_state: template.clone(),
            work_f: template.clone(),
            w_lu: [0.0; DEFAULT_STIFF_MAT_CAP],
            vecs: [0.0; 4 * DEFAULT_STIFF_MAX_DIM],
            pivots: [0; DEFAULT_STIFF_MAX_DIM],
            prev_dt: 0.0,
            expected_t: 0.0,
            has_history: false,
        }
    }

    /// Clears stored multi-step history so the next call to `step` re-bootstraps via TR-BDF2.
    #[verified_engine::verified]
    pub fn reset(&mut self) {
        self.has_history = false;
        self.prev_dt = 0.0;
    }

    /// Advances `state` by `dt` using an explicit caller-owned [`SolverWorkspace`] and `pivots` slice.
    #[verified_engine::verified]
    pub fn step_in_workspace<S: OdeSystem<State> + ?Sized>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut State,
        dt: f64,
        workspace: &mut SolverWorkspace<'_>,
        pivots: &mut [usize],
    ) -> Result<(), DiffeqError> {
        if !dt.is_finite() || dt == 0.0 {
            return Err(DiffeqError::InvalidStepSize { dt });
        }
        let n = state.state_dim();
        if pivots.len() < n {
            return Err(DiffeqError::InsufficientWorkspace {
                required: n,
                available: pivots.len(),
            });
        }
        let buf = workspace.try_slice_mut(SolverWorkspace::stiff_scratch_len(n))?;
        let (w_lu, vec_area) = buf.split_at_mut(n * n);
        let mut scratch = StiffScratch {
            w_lu,
            vec_area,
            pivots,
            dim: n,
        };
        self.execute_bdf2_step(system, t, state, dt, &mut scratch)
    }

    #[verified_engine::verified]
    fn execute_bdf2_step<S: OdeSystem<State> + ?Sized>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut State,
        dt: f64,
        scratch: &mut StiffScratch<'_>,
    ) -> Result<(), DiffeqError> {
        let n = scratch.dim;
        let contiguous = self.has_history
            && self.prev_dt.abs() > 1e-30
            && (dt > 0.0) == (self.prev_dt > 0.0)
            && (t - self.expected_t).abs() <= 1e-7 * dt.abs().max(1.0);

        let y_curr = state.clone();
        if !contiguous {
            self.bootstrap_trbdf2(system, t, state, dt, scratch)?;
        } else {
            let omega = dt / self.prev_dt;
            let denom = 1.0 + 2.0 * omega;
            let alpha1 = ((1.0 + omega) * (1.0 + omega)) / denom;
            let alpha0 = -(omega * omega) / denom;
            let beta = (1.0 + omega) / denom;
            for i in 0..n {
                let hist = alpha1 * y_curr.component(i) + alpha0 * self.prev_state.component(i);
                self.anchor.set_component(i, hist);
            }
            let target = ImplicitTarget {
                t_next: t + dt,
                coeff_c: beta * dt,
                anchor: &self.anchor,
            };
            let mut ns = NewtonStates {
                f_eval: &mut self.f_eval,
                cand: &mut self.cand_state,
                work_state: &mut self.work_state,
                work_f: &mut self.work_f,
            };
            solve_implicit_newton(system, state, &target, &mut ns, scratch)?;
        }

        self.prev_state.copy_from(&y_curr);
        self.prev_dt = dt;
        self.expected_t = t + dt;
        self.has_history = true;
        Ok(())
    }

    #[verified_engine::verified]
    fn bootstrap_trbdf2<S: OdeSystem<State> + ?Sized>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut State,
        dt: f64,
        scratch: &mut StiffScratch<'_>,
    ) -> Result<(), DiffeqError> {
        let n = scratch.dim;
        let y0 = state.clone();
        let half_gamma_h = 0.5 * TRBDF2_GAMMA * dt;
        system.derivative_in_place(t, &y0, &mut self.f_eval);
        for i in 0..n {
            self.anchor
                .set_component(i, y0.component(i) + half_gamma_h * self.f_eval.component(i));
        }

        let mut y_gamma = y0.clone();
        {
            let target1 = ImplicitTarget {
                t_next: t + TRBDF2_GAMMA * dt,
                coeff_c: half_gamma_h,
                anchor: &self.anchor,
            };
            let mut ns = NewtonStates {
                f_eval: &mut self.f_eval,
                cand: &mut self.cand_state,
                work_state: &mut self.work_state,
                work_f: &mut self.work_f,
            };
            solve_implicit_newton(system, &mut y_gamma, &target1, &mut ns, scratch)?;
        }

        for i in 0..n {
            let rhs_hist = TRBDF2_W1 * y_gamma.component(i) + TRBDF2_W0 * y0.component(i);
            self.anchor.set_component(i, rhs_hist);
        }
        state.copy_from(&y_gamma);
        let target2 = ImplicitTarget {
            t_next: t + dt,
            coeff_c: half_gamma_h,
            anchor: &self.anchor,
        };
        let mut ns = NewtonStates {
            f_eval: &mut self.f_eval,
            cand: &mut self.cand_state,
            work_state: &mut self.work_state,
            work_f: &mut self.work_f,
        };
        solve_implicit_newton(system, state, &target2, &mut ns, scratch)
    }
}

#[verified_engine::verified]
fn eval_residual_inf<State: OdeState>(
    y: &State,
    f_val: &State,
    anchor: &State,
    coeff_c: f64,
    res_out: &mut [f64],
    n: usize,
) -> (f64, f64) {
    let mut max_res = 0.0_f64;
    let mut max_y = 1.0_f64;
    for (i, slot) in res_out.iter_mut().enumerate().take(n) {
        let yi = y.component(i);
        let ri = yi - anchor.component(i) - coeff_c * f_val.component(i);
        *slot = ri;
        if !yi.is_finite() || !ri.is_finite() {
            return (f64::INFINITY, max_y);
        }
        if ri.abs() > max_res {
            max_res = ri.abs();
        }
        if yi.abs() > max_y {
            max_y = yi.abs();
        }
    }
    (max_res, max_y)
}

#[verified_engine::verified]
fn solve_implicit_newton<State: OdeState, S: OdeSystem<State> + ?Sized>(
    system: &S,
    y: &mut State,
    target: &ImplicitTarget<'_, State>,
    ns: &mut NewtonStates<'_, State>,
    scratch: &mut StiffScratch<'_>,
) -> Result<(), DiffeqError> {
    let n = scratch.dim;
    for _ in 0..MAX_NEWTON_ITERS {
        system.derivative_in_place(target.t_next, y, ns.f_eval);
        let (res_norm, y_scale) = {
            let (res, _) = scratch.vec_area.split_at_mut(n);
            eval_residual_inf(y, ns.f_eval, target.anchor, target.coeff_c, res, n)
        };
        if res_norm <= 1e-12 * y_scale {
            return Ok(());
        }
        {
            let mut work = StatePair {
                state: ns.work_state,
                deriv: ns.work_f,
            };
            build_and_factor_stage_matrix(
                system,
                target.t_next,
                y,
                ns.f_eval,
                target.coeff_c,
                &mut work,
                scratch,
            )?;
        }

        let (res, rest) = scratch.vec_area.split_at_mut(n);
        let (delta, rest) = rest.split_at_mut(n);
        let (trial_res, _) = rest.split_at_mut(n);
        lu_solve_in_place(scratch.w_lu, scratch.pivots, res, delta, n);

        let mut accepted = false;
        let mut damp = 1.0_f64;
        for _ in 0..6 {
            for (i, &d_i) in delta.iter().enumerate().take(n) {
                ns.cand.set_component(i, y.component(i) - damp * d_i);
            }
            system.derivative_in_place(target.t_next, ns.cand, ns.work_f);
            let (cand_norm, _) = eval_residual_inf(
                ns.cand,
                ns.work_f,
                target.anchor,
                target.coeff_c,
                trial_res,
                n,
            );
            if cand_norm.is_finite() && (cand_norm < res_norm || damp <= 0.04) {
                y.copy_from(ns.cand);
                accepted = true;
                break;
            }
            damp *= 0.5;
        }
        if !accepted {
            return Err(DiffeqError::NonFiniteEvaluation);
        }
        let mut step_norm = 0.0_f64;
        for d in delta.iter().take(n) {
            if d.abs() > step_norm {
                step_norm = d.abs();
            }
        }
        if step_norm * damp <= 1e-13 * y_scale {
            return Ok(());
        }
    }
    Ok(())
}

impl<State: OdeState> Solver<State> for Bdf2<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        let n = state.state_dim();
        if n == 0 || n > DEFAULT_STIFF_MAX_DIM || !dt.is_finite() || dt.abs() <= 1e-30 {
            return;
        }
        let mut w_lu = self.w_lu;
        let mut vecs = self.vecs;
        let mut pivots = self.pivots;
        let mut scratch = StiffScratch {
            w_lu: &mut w_lu,
            vec_area: &mut vecs,
            pivots: &mut pivots,
            dim: n,
        };
        if self
            .execute_bdf2_step(system, t, state, dt, &mut scratch)
            .is_ok()
        {
            self.w_lu = w_lu;
            self.vecs = vecs;
            self.pivots = pivots;
        }
    }
}
