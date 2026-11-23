//! L-stable second/third-order Rosenbrock-Wanner (`ode23s`) solver for stiff and non-autonomous ODEs.
// [cite:engine_diffeq]

use super::stiff_common::{
    DEFAULT_STIFF_MAT_CAP, DEFAULT_STIFF_MAX_DIM, StatePair, StiffScratch,
    build_and_factor_stage_matrix, compute_time_partial,
};
use super::traits::{OdeState, OdeSystem, Solver};
use crate::error::DiffeqError;
use crate::workspace::{SolverWorkspace, lu_solve_in_place};

/// Shampine & Reichelt `ode23s` L-stability parameter $d = \frac{1}{2 + \sqrt{2}} = 1 - \frac{\sqrt{2}}{2}$.
const ROS23_D: f64 = 0.292_893_218_813_452_5;
/// `ode23s` stage-3 error coefficient $e_{32} = 6 + \sqrt{2}$.
const ROS23_E32: f64 = 7.414_213_562_373_095;

/// L-stable Rosenbrock-Wanner 2(3) stiff ODE solver (`ode23s`) using bounded LU scratch buffers.
#[derive(Debug, Clone)]
pub struct Rosenbrock23<State> {
    f0: State,
    f1: State,
    f2: State,
    work_state: State,
    work_f: State,
    w_lu: [f64; DEFAULT_STIFF_MAT_CAP],
    vecs: [f64; 6 * DEFAULT_STIFF_MAX_DIM],
    pivots: [usize; DEFAULT_STIFF_MAX_DIM],
    last_error: f64,
}

impl<State: OdeState> Rosenbrock23<State> {
    /// Creates a new `Rosenbrock23` solver pre-initializing state scratch buffers from `template`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            f0: template.clone(),
            f1: template.clone(),
            f2: template.clone(),
            work_state: template.clone(),
            work_f: template.clone(),
            w_lu: [0.0; DEFAULT_STIFF_MAT_CAP],
            vecs: [0.0; 6 * DEFAULT_STIFF_MAX_DIM],
            pivots: [0; DEFAULT_STIFF_MAX_DIM],
            last_error: 0.0,
        }
    }

    /// Returns the embedded 3rd-order local truncation error estimate ($L_\infty$ norm) from the last step.
    #[must_use]
    #[verified_engine::verified]
    pub const fn last_error(&self) -> f64 {
        self.last_error
    }

    /// Advances `state` by `dt` using an explicit caller-owned [`SolverWorkspace`] and `pivots` slice,
    /// supporting arbitrary state dimensions `n` (`workspace.capacity() >= n * n + 8 * n`).
    #[verified_engine::verified]
    pub fn step_in_workspace<S: OdeSystem<State> + ?Sized>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut State,
        dt: f64,
        workspace: &mut SolverWorkspace<'_>,
        pivots: &mut [usize],
    ) -> Result<f64, DiffeqError> {
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
        let err = self.execute_ros23_step(system, t, state, dt, &mut scratch)?;
        self.last_error = err;
        Ok(err)
    }

    #[verified_engine::verified]
    fn execute_ros23_step<S: OdeSystem<State> + ?Sized>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut State,
        h: f64,
        scratch: &mut StiffScratch<'_>,
    ) -> Result<f64, DiffeqError> {
        let n = scratch.dim;
        system.derivative_in_place(t, state, &mut self.f0);
        {
            let mut work = StatePair {
                state: &mut self.work_state,
                deriv: &mut self.work_f,
            };
            build_and_factor_stage_matrix(
                system,
                t,
                state,
                &self.f0,
                h * ROS23_D,
                &mut work,
                scratch,
            )?;
        }

        let (df_dt, rest) = scratch.vec_area.split_at_mut(n);
        let (rhs, rest) = rest.split_at_mut(n);
        let (k1, rest) = rest.split_at_mut(n);
        let (k2, rest) = rest.split_at_mut(n);
        let (k3, _) = rest.split_at_mut(n);

        {
            let mut work = StatePair {
                state: &mut self.work_state,
                deriv: &mut self.work_f,
            };
            compute_time_partial(system, t, state, h, &mut work, df_dt);
        }

        for (i, slot) in rhs.iter_mut().enumerate().take(n) {
            *slot = self.f0.component(i) + h * ROS23_D * df_dt[i];
        }
        lu_solve_in_place(scratch.w_lu, scratch.pivots, rhs, k1, n);

        for (i, &k1_i) in k1.iter().enumerate().take(n) {
            self.work_state
                .set_component(i, state.component(i) + 0.5 * h * k1_i);
        }
        system.derivative_in_place(t + 0.5 * h, &self.work_state, &mut self.f1);

        for (i, slot) in rhs.iter_mut().enumerate().take(n) {
            *slot = self.f1.component(i) - k1[i];
        }
        lu_solve_in_place(scratch.w_lu, scratch.pivots, rhs, k2, n);
        for i in 0..n {
            k2[i] += k1[i];
            self.work_state
                .set_component(i, state.component(i) + h * k2[i]);
        }

        system.derivative_in_place(t + h, &self.work_state, &mut self.f2);
        for (i, slot) in rhs.iter_mut().enumerate().take(n) {
            let f0_i = self.f0.component(i);
            let f1_i = self.f1.component(i);
            let f2_i = self.f2.component(i);
            *slot =
                f2_i - ROS23_E32 * (k2[i] - f1_i) - 2.0 * (k1[i] - f0_i) + h * ROS23_D * df_dt[i];
        }
        lu_solve_in_place(scratch.w_lu, scratch.pivots, rhs, k3, n);

        let mut max_err = 0.0_f64;
        for i in 0..n {
            state.set_component(i, self.work_state.component(i));
            let e_i = (h / 6.0) * (k1[i] - 2.0 * k2[i] + k3[i]).abs();
            if e_i > max_err {
                max_err = e_i;
            }
        }
        Ok(max_err)
    }
}

impl<State: OdeState> Solver<State> for Rosenbrock23<State> {
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
        if let Ok(err) = self.execute_ros23_step(system, t, state, dt, &mut scratch) {
            self.w_lu = w_lu;
            self.vecs = vecs;
            self.pivots = pivots;
            self.last_error = err;
        }
    }
}
