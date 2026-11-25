// [cite:engine_diffeq]
use crate::system::{SdeSystem, StochasticCalculus};
use math_commons::math_kernel::types::{
    Scalar, SolverWorkspace, fill_standard_normal, flatten_2d_index,
};
use rand_core::RngCore;

/// Multivariate Euler-Maruyama (Itô) and Euler-Heun (Stratonovich) stochastic integrator for
/// $dX_t = f(t, X_t)\,dt + G(t, X_t)\,dW_t$.
///
/// Reference: Kloeden & Platen (1992), *Numerical Solution of Stochastic Differential Equations*,
/// DOI: 10.1007/978-3-662-12616-5; [`docs/math_specs/engine_diffeq.md`](../../../docs/math_specs/engine_diffeq.md).
///
/// # Stability
/// Achieves strong convergence order $0.5$ (weak order $1.0$) for general non-commutative noise;
/// mean-square linear stability requires $|1 + \lambda \Delta t|^2 + |\mu|^2 \Delta t < 1$.
///
/// # Domain of Validity
/// Requires Lipschitz drift and diffusion coefficients, `dt >= 0`, `state.len() >= state_dim`,
/// and `SolverWorkspace` capacity at least [`Self::scratch_len`]`(state_dim, wiener_dim)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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

    /// Advances `state` by one step `dt` using freshly sampled Wiener increments from `rng`.
    #[verified_engine::verified]
    pub fn step<T: Scalar, S: SdeSystem<T> + ?Sized, R: RngCore + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        state: &mut [T],
        workspace: &mut impl SolverWorkspace,
        rng: &mut R,
    ) {
        let d = system.state_dim();
        let m = system.wiener_dim();
        assert!(dt >= T::ZERO, "time step dt must be non-negative");
        assert!(state.len() >= d, "state slice shorter than state_dim");

        let mut ws = workspace.reborrow();
        let dw = ws.take_slice::<T>(m);
        fill_standard_normal(dw, rng);
        let sqrt_dt = dt.sqrt();
        for w in dw.iter_mut() {
            *w *= sqrt_dt;
        }
        self.step_with_increments(system, t, dt, dw, state, &mut ws);
    }

    /// Advances `state` by one step `dt` given pre-sampled Wiener increments `dw` ($\Delta W \sim \mathcal{N}(0, \Delta t I_m)$).
    #[verified_engine::verified]
    pub fn step_with_increments<T: Scalar, S: SdeSystem<T> + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        dw: &[T],
        state: &mut [T],
        workspace: &mut impl SolverWorkspace,
    ) {
        let d = system.state_dim();
        let m = system.wiener_dim();
        assert!(state.len() >= d && dw.len() >= m);

        let dm = d * m;
        let mut ws = workspace.reborrow();
        let f0 = ws.take_slice::<T>(d);
        let g0 = ws.take_slice::<T>(dm);
        system.drift(t, state, f0);
        system.diffusion(t, state, g0);

        match self.calculus {
            StochasticCalculus::Ito => {
                for i in 0..d {
                    let mut diff_sum = T::ZERO;
                    let row = &g0[i * m..(i + 1) * m];
                    for j in 0..m {
                        diff_sum += row[j] * dw[j];
                    }
                    state[i] += f0[i] * dt + diff_sum;
                }
            }
            StochasticCalculus::Stratonovich => {
                let x_pred = ws.take_slice::<T>(d);
                let f1 = ws.take_slice::<T>(d);
                let g1 = ws.take_slice::<T>(dm);
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
}
