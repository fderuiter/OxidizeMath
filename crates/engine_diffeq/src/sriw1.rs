// [cite:engine_diffeq]
use crate::system::{
    SdeSystem, StochasticCalculus, eval_ito_effective_drift, ito_drift_scratch_len,
};
use math_commons::math_kernel::types::{
    Scalar, SolverWorkspace, flatten_2d_index, sample_standard_normal_pair,
};
use rand_core::RngCore;

/// Correlated Gaussian increments $(\Delta W_j, \Delta Z_j)$ for strong-order-1.5 Stochastic Runge-Kutta,
/// where $\Delta W_j = \int_{t_n}^{t_{n+1}} dW_s^j$ and $\Delta Z_j = \int_{t_n}^{t_{n+1}} (W_s^j - W_{t_n}^j)\,ds$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CorrelatedWienerIncrement<T: Scalar> {
    /// First-order Wiener increment $\Delta W_j \sim \mathcal{N}(0, \Delta t)$.
    pub dw: T,
    /// Coupled time-Wiener integral $\Delta Z_j$ with $\mathbb{E}[\Delta W_j \Delta Z_j] = \frac{1}{2}\Delta t^2$
    /// and $\mathbb{E}[\Delta Z_j^2] = \frac{1}{3}\Delta t^3$.
    pub dz: T,
}

impl<T: Scalar> CorrelatedWienerIncrement<T> {
    /// Samples a correlated pair $(\Delta W, \Delta Z)$ for step size `dt` from `rng`.
    #[inline]
    #[verified_engine::verified]
    pub fn sample<R: RngCore + ?Sized>(dt: T, rng: &mut R) -> Self {
        debug_assert!(dt >= T::ZERO);
        let (z1, z2) = sample_standard_normal_pair::<T, R>(rng);
        let sqrt_dt = dt.sqrt();
        let inv_sqrt_three = T::ONE / T::from_f64(3.0).sqrt();
        let dw = sqrt_dt * z1;
        let dz = T::HALF * dt * sqrt_dt * (z1 + inv_sqrt_three * z2);
        Self { dw, dz }
    }
}

/// Derivative-free strong-order-1.5 Rössler-type Stochastic Runge-Kutta (`SRIW1`) solver for
/// multivariate SDEs $dX_t = f(t, X_t)\,dt + G(t, X_t)\,dW_t$ under Itô or Stratonovich calculus.
///
/// Reference: Rössler (2010), *SIAM J. Numer. Anal.* 48(3), DOI: 10.1137/09076636X;
/// [`docs/math_specs/engine_diffeq.md`](../../../docs/math_specs/engine_diffeq.md).
///
/// # Stability
/// Supporting-value Runge-Kutta stages avoid explicit Jacobian evaluations while attaining strong
/// order $1.5$ for scalar/diagonal/commutative noise under explicit RK mean-square step bounds.
///
/// # Domain of Validity
/// Requires smooth ($C^2$/$C^3$) drift and commutative/diagonal diffusion for order $1.5$ (reduces to
/// strong order $0.5$ on non-commutative multi-channel noise without Lévy areas), `dt >= 0`,
/// `state.len() >= state_dim`, and `SolverWorkspace` capacity [`Self::scratch_len`]`(state_dim, wiener_dim)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Sriw1Solver {
    calculus: StochasticCalculus,
}

impl Sriw1Solver {
    /// Creates an [`Sriw1Solver`] configured for `calculus`.
    #[inline]
    pub const fn new(calculus: StochasticCalculus) -> Self {
        Self { calculus }
    }

    /// Returns the configured [`StochasticCalculus`].
    #[inline]
    pub const fn calculus(&self) -> StochasticCalculus {
        self.calculus
    }

    /// Returns the minimum [`SolverWorkspace`] length required for `(state_dim, wiener_dim)`.
    #[inline]
    pub const fn scratch_len(&self, state_dim: usize, wiener_dim: usize) -> usize {
        let dm = state_dim * wiener_dim;
        let drift_sub = ito_drift_scratch_len(state_dim, wiener_dim);
        2 * wiener_dim + 7 * state_dim + 4 * dm + drift_sub
    }

    /// Advances `state` by one step `dt` drawing correlated $(\Delta W, \Delta Z)$ pairs from `rng`.
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
        assert!(dt >= T::ZERO && state.len() >= d);
        if dt <= T::EPSILON {
            return;
        }

        let mut ws = workspace.reborrow();
        let dw = ws.take_slice::<T>(m);
        let dz = ws.take_slice::<T>(m);
        for j in 0..m {
            let inc = CorrelatedWienerIncrement::sample(dt, rng);
            dw[j] = inc.dw;
            dz[j] = inc.dz;
        }
        self.step_with_correlated_increments(system, t, dt, dw, dz, state, &mut ws);
    }

    /// Advances `state` by one step `dt` using caller-supplied correlated slices `dw` ($\Delta W$) and `dz` ($\Delta Z$).
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    pub fn step_with_correlated_increments<T: Scalar, S: SdeSystem<T> + ?Sized>(
        &self,
        system: &S,
        t: T,
        dt: T,
        dw: &[T],
        dz: &[T],
        state: &mut [T],
        workspace: &mut impl SolverWorkspace,
    ) {
        let d = system.state_dim();
        let m = system.wiener_dim();
        assert!(state.len() >= d && dw.len() >= m && dz.len() >= m);
        if dt <= T::EPSILON {
            return;
        }

        let dm = d * m;
        let mut ws = workspace.reborrow();
        let a1 = ws.take_slice::<T>(d);
        let a2 = ws.take_slice::<T>(d);
        let h0 = ws.take_slice::<T>(d);
        let y_plus = ws.take_slice::<T>(d);
        let y_minus = ws.take_slice::<T>(d);
        let psi_plus = ws.take_slice::<T>(d);
        let delta_x = ws.take_slice::<T>(d);
        let g1 = ws.take_slice::<T>(dm);
        let g_yp = ws.take_slice::<T>(dm);
        let g_ym = ws.take_slice::<T>(dm);
        let g_phi = ws.take_slice::<T>(dm);

        eval_ito_effective_drift(system, self.calculus, t, state, a1, &mut ws);
        system.diffusion(t, state, g1);

        let three_quarters = T::from_f64(0.75);
        let three_halves_inv_dt = T::from_f64(1.5) / dt;
        for i in 0..d {
            let mut b_dz = T::ZERO;
            for j in 0..m {
                b_dz += g1[flatten_2d_index(j, i, m)] * dz[j];
            }
            h0[i] = state[i] + three_quarters * a1[i] * dt + three_halves_inv_dt * b_dz;
        }
        eval_ito_effective_drift(
            system,
            self.calculus,
            t + three_quarters * dt,
            h0,
            a2,
            &mut ws,
        );

        let one_third = T::ONE / T::from_f64(3.0);
        let two_thirds = T::TWO * one_third;
        for i in 0..d {
            delta_x[i] = (one_third * a1[i] + two_thirds * a2[i]) * dt;
        }

        accumulate_sriw1_diffusion(
            system, t, dt, dw, dz, state, a1, g1, y_plus, y_minus, psi_plus, g_yp, g_ym, g_phi,
            delta_x,
        );
        for i in 0..d {
            state[i] += delta_x[i];
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn accumulate_sriw1_diffusion<T: Scalar, S: SdeSystem<T> + ?Sized>(
    system: &S,
    t: T,
    dt: T,
    dw: &[T],
    dz: &[T],
    state: &[T],
    a1: &[T],
    g1: &[T],
    y_plus: &mut [T],
    y_minus: &mut [T],
    psi_plus: &mut [T],
    g_yp: &mut [T],
    g_ym: &mut [T],
    g_phi: &mut [T],
    delta_x: &mut [T],
) {
    let d = system.state_dim();
    let m = system.wiener_dim();
    debug_assert!(state.len() >= d && delta_x.len() >= d);
    let sqrt_dt = dt.sqrt();
    let inv_two_sqrt_dt = T::HALF / sqrt_dt;
    let inv_two_dt = T::HALF / dt;
    let inv_dt = T::ONE / dt;
    let one_sixth = T::ONE / T::from_f64(6.0);

    for j in 0..m {
        let dw_j = dw[j];
        let i_11 = T::HALF * (dw_j * dw_j - dt);
        let i_01 = dw_j * dt - dz[j];
        let i_111 = one_sixth * (dw_j * dw_j - T::from_f64(3.0) * dt) * dw_j;

        for i in 0..d {
            let b_ij = g1[flatten_2d_index(j, i, m)];
            y_plus[i] = state[i] + a1[i] * dt + b_ij * sqrt_dt;
            y_minus[i] = state[i] + a1[i] * dt - b_ij * sqrt_dt;
        }
        system.diffusion(t + dt, y_plus, g_yp);
        system.diffusion(t + dt, y_minus, g_ym);

        for i in 0..d {
            let b_ij = g1[flatten_2d_index(j, i, m)];
            y_plus[i] = state[i] + b_ij * sqrt_dt;
            y_minus[i] = state[i] - b_ij * sqrt_dt;
        }
        system.diffusion(t, y_plus, g_phi);
        for i in 0..d {
            psi_plus[i] = state[i] + g_phi[flatten_2d_index(j, i, m)] * sqrt_dt;
        }
        // Reuse g_phi for b(t, psi_plus) and g_yp/g_ym before overwriting
        for (i, dx_i) in delta_x.iter_mut().enumerate().take(d) {
            let idx = flatten_2d_index(j, i, m);
            let b0 = g1[idx];
            let bp = g_yp[idx];
            let bm = g_ym[idx];
            *dx_i += b0 * dw_j
                + (bp - bm) * inv_two_sqrt_dt * i_11
                + (bp + bm - T::TWO * b0) * inv_two_dt * i_01;
        }

        // Cross-channel commutative Milstein terms for j2 > j using g_phi = b(t, state + b_j sqrt(dt))
        system.diffusion(t, y_minus, g_ym);
        for (j2, &dw_j2) in dw.iter().enumerate().take(m).skip(j + 1) {
            let cross_w = dw_j * dw_j2;
            for (i, dx_i) in delta_x.iter_mut().enumerate().take(d) {
                let idx2 = flatten_2d_index(j2, i, m);
                *dx_i += (g_phi[idx2] - g_ym[idx2]) * inv_two_sqrt_dt * cross_w;
            }
        }

        system.diffusion(t, psi_plus, g_yp);
        for (i, dx_i) in delta_x.iter_mut().enumerate().take(d) {
            let idx = flatten_2d_index(j, i, m);
            let triple = (g_yp[idx] + g_ym[idx] - T::TWO * g1[idx]) * inv_dt;
            *dx_i += triple * i_111;
        }
    }
}
