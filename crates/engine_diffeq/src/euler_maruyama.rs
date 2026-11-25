// [cite:engine_diffeq]
use crate::error::DiffeqError;
use crate::system::{SdeSystem, StochasticCalculus};
use math_commons::math_kernel::types::{
    Scalar, SolverWorkspace, fill_standard_normal, flatten_2d_index,
};
use rand_core::RngCore;

/// Multivariate Euler-Maruyama (Itô) and Euler-Heun (Stratonovich) stochastic integrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EulerMaruyamaSolver {
    calculus: StochasticCalculus,
}

impl EulerMaruyamaSolver {
    /// Creates an [`EulerMaruyamaSolver`] configured for `calculus`.
    #[inline]
    pub const fn new(calculus: StochasticCalculus) -> Self {
        Self { calculus }
    }

    /// Returns the configured [`StochasticCalculus`].
    #[inline]
    pub const fn calculus(&self) -> StochasticCalculus {
        self.calculus
    }

    /// Returns the minimum [`SolverWorkspace`] length required for a system of `(state_dim, wiener_dim)`.
    #[inline]
    pub const fn scratch_len(&self, state_dim: usize, wiener_dim: usize) -> usize {
        match self.calculus {
            StochasticCalculus::Ito => state_dim + state_dim * wiener_dim + wiener_dim,
            StochasticCalculus::Stratonovich => {
                3 * state_dim + 2 * state_dim * wiener_dim + wiener_dim
            }
        }
    }

    /// Fallibly advances `state` by one step `dt` using freshly sampled Wiener increments from `rng`.
    ///
    /// # Errors
    /// Returns [`DiffeqError`] if `dt < 0` or non-finite, `state` is shorter than `system.state_dim()`,
    /// or `workspace` is smaller than [`Self::scratch_len`].
    #[verified_engine::verified]
    pub fn try_step<T: Scalar, S: SdeSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        state: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
        rng: &mut R,
    ) -> Result<(), DiffeqError> {
        let d = system.state_dim();
        let m = system.wiener_dim();
        if !(dt >= T::ZERO && dt.is_finite()) {
            return Err(DiffeqError::InvalidStepSize);
        }
        if state.len() < d {
            return Err(DiffeqError::BufferTooSmall {
                buffer: "state",
                required: d,
                actual: state.len(),
            });
        }

        let mut ws = workspace.reborrow();
        let dw = ws.try_take_slice(m)?;
        fill_standard_normal(dw, rng);
        let sqrt_dt = dt.sqrt();
        for w in dw.iter_mut() {
            *w *= sqrt_dt;
        }
        debug_assert!(sqrt_dt >= T::ZERO);
        self.try_step_with_increments(system, t, dt, dw, state, &mut ws)
    }

    /// Advances `state` by one step `dt` using freshly sampled Wiener increments from `rng`.
    ///
    /// # Panics
    /// Panics if `dt < 0`, `state` is undersized, or `workspace` is too small; prefer [`Self::try_step`].
    #[verified_engine::verified]
    pub fn step<T: Scalar, S: SdeSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        state: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
        rng: &mut R,
    ) {
        self.try_step(system, t, dt, state, workspace, rng)
            .expect("EulerMaruyamaSolver::step preconditions violated");
    }

    /// Fallibly advances `state` by one step `dt` given pre-sampled Wiener increments `dw`.
    ///
    /// # Errors
    /// Returns [`DiffeqError`] if `dt < 0` or non-finite, `state` or `dw` is undersized,
    /// or `workspace` lacks sufficient scratch capacity.
    #[verified_engine::verified]
    pub fn try_step_with_increments<T: Scalar, S: SdeSystem<T> + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        dw: &[T],
        state: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
    ) -> Result<(), DiffeqError> {
        let d = system.state_dim();
        let m = system.wiener_dim();
        if !(dt >= T::ZERO && dt.is_finite()) {
            return Err(DiffeqError::InvalidStepSize);
        }
        if state.len() < d {
            return Err(DiffeqError::BufferTooSmall {
                buffer: "state",
                required: d,
                actual: state.len(),
            });
        }
        if dw.len() < m {
            return Err(DiffeqError::BufferTooSmall {
                buffer: "dw",
                required: m,
                actual: dw.len(),
            });
        }

        let dm = d * m;
        let mut ws = workspace.reborrow();
        let f0 = ws.try_take_slice(d)?;
        let g0 = ws.try_take_slice(dm)?;
        let euler_heun_slices = if self.calculus == StochasticCalculus::Stratonovich {
            let xp = ws.try_take_slice(d)?;
            let f1 = ws.try_take_slice(d)?;
            let g1 = ws.try_take_slice(dm)?;
            Some((xp, f1, g1))
        } else {
            None
        };

        system.drift(t, state, f0);
        system.diffusion(t, state, g0);
        apply_euler_update(system, t, dt, dw, state, f0, g0, euler_heun_slices);
        debug_assert!(state.len() >= d);
        Ok(())
    }

    /// Advances `state` by one step `dt` given pre-sampled Wiener increments `dw` ($\Delta W \sim \mathcal{N}(0, \Delta t I_m)$).
    ///
    /// # Panics
    /// Panics if `dt < 0`, `state`/`dw` are undersized, or `workspace` is too small; prefer [`Self::try_step_with_increments`].
    #[verified_engine::verified]
    pub fn step_with_increments<T: Scalar, S: SdeSystem<T> + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        dw: &[T],
        state: &mut [T],
        workspace: &mut SolverWorkspace<'_, T>,
    ) {
        self.try_step_with_increments(system, t, dt, dw, state, workspace)
            .expect("EulerMaruyamaSolver::step_with_increments preconditions violated");
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn apply_euler_update<T: Scalar, S: SdeSystem<T> + ?Sized>(
    system: &S,
    t: T,
    dt: T,
    dw: &[T],
    state: &mut [T],
    f0: &[T],
    g0: &[T],
    stratonovich_scratch: Option<(&mut [T], &mut [T], &mut [T])>,
) {
    let d = system.state_dim();
    let m = system.wiener_dim();
    debug_assert!(state.len() >= d && dw.len() >= m);
    match stratonovich_scratch {
        None => {
            for i in 0..d {
                let mut diff_sum = T::ZERO;
                let row = &g0[i * m..(i + 1) * m];
                for j in 0..m {
                    diff_sum += row[j] * dw[j];
                }
                state[i] += f0[i] * dt + diff_sum;
            }
        }
        Some((x_pred, f1, g1)) => {
            for i in 0..d {
                let mut diff_sum = T::ZERO;
                let row = &g0[i * m..(i + 1) * m];
                for j in 0..m {
                    diff_sum += row[j] * dw[j];
                }
                x_pred[i] = state[i] + f0[i] * dt + diff_sum;
            }
            system.drift(t + dt, x_pred, f1);
            system.diffusion(t + dt, x_pred, g1);
            for i in 0..d {
                let mut diff_avg = T::ZERO;
                for (j, &dw_j) in dw.iter().enumerate().take(m) {
                    let idx = flatten_2d_index(j, i, m);
                    diff_avg += T::HALF * (g0[idx] + g1[idx]) * dw_j;
                }
                state[i] += T::HALF * (f0[i] + f1[i]) * dt + diff_avg;
            }
        }
    }
}
