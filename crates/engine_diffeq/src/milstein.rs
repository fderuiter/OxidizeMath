// [cite:engine_diffeq]
use crate::error::DiffeqError;
use crate::system::{SdeSystem, StochasticCalculus};
use math_commons::math_kernel::types::{
    Scalar, SolverWorkspace, fill_standard_normal, flatten_2d_index, sample_standard_normal_pair,
};
use rand_core::RngCore;

/// Multivariate strong-order-1.0 Milstein SDE solver supporting both Itô and Stratonovich calculi.
///
/// For diagonal or commutative diffusion matrices ($D g_{j_1} g_{j_2} = D g_{j_2} g_{j_1}$), set
/// `levy_truncation = 0` for exact zero-area overhead. For general non-commutative multi-dimensional
/// noise, `levy_truncation > 0` approximates the Lévy area $A_{j_1, j_2}$ via a `p`-mode Karhunen-Loève
/// Fourier expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MilsteinSolver {
    calculus: StochasticCalculus,
    levy_truncation: usize,
}

impl Default for MilsteinSolver {
    fn default() -> Self {
        Self::new(StochasticCalculus::Ito)
    }
}

impl MilsteinSolver {
    /// Creates a [`MilsteinSolver`] for diagonal or commutative noise (`levy_truncation = 0`).
    #[inline]
    pub const fn new(calculus: StochasticCalculus) -> Self {
        Self {
            calculus,
            levy_truncation: 0,
        }
    }

    /// Configures `p` Karhunen-Loève modes for non-commutative multi-channel Lévy area sampling.
    #[inline]
    pub const fn with_levy_truncation(mut self, modes: usize) -> Self {
        self.levy_truncation = modes;
        self
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
        let levy = if self.levy_truncation > 0 && wiener_dim > 1 {
            wiener_dim * wiener_dim + 2 * wiener_dim
        } else {
            0
        };
        5 * state_dim + 3 * dm + wiener_dim + levy
    }

    /// Fallibly advances `state` by one step `dt` drawing increments (and optional Lévy areas) from `rng`.
    ///
    /// # Errors
    /// Returns [`DiffeqError`] if `dt < 0` or non-finite, `state` is undersized, or `workspace`
    /// lacks [`Self::scratch_len`] elements.
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
        let levy_slices = if self.levy_truncation > 0 && m > 1 {
            let area = ws.try_take_slice(m * m)?;
            let xi = ws.try_take_slice(m)?;
            let zeta = ws.try_take_slice(m)?;
            Some((area, xi, zeta))
        } else {
            None
        };
        let (f0, g0, g_plus, g_minus, col_vec, dg_a, dg_b, x_tmp) =
            take_milstein_slices(&mut ws, d, d * m)?;

        fill_standard_normal(dw, rng);
        let sqrt_dt = dt.sqrt();
        for w in dw.iter_mut() {
            *w *= sqrt_dt;
        }
        debug_assert!(sqrt_dt >= T::ZERO);

        let area_ref = if let Some((area, xi, zeta)) = levy_slices {
            sample_levy_area(dt, dw, self.levy_truncation, area, xi, zeta, rng);
            Some(&*area)
        } else {
            None
        };

        system.drift(t, state, f0);
        system.diffusion(t, state, g0);
        accumulate_milstein_update(
            system,
            self.calculus,
            t,
            dt,
            dw,
            area_ref,
            state,
            f0,
            g0,
            g_plus,
            g_minus,
            col_vec,
            dg_a,
            dg_b,
            x_tmp,
        );
        Ok(())
    }

    /// Advances `state` by one step `dt` drawing increments (and optional Lévy areas) from `rng`.
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
            .expect("MilsteinSolver::step preconditions violated");
    }

    /// Fallibly advances `state` by one step `dt` given pre-sampled Wiener increments `dw`.
    ///
    /// # Errors
    /// Returns [`DiffeqError`] if `dt < 0` or non-finite, `state`/`dw` are undersized,
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
        let mut ws = workspace.reborrow();
        let (f0, g0, g_plus, g_minus, col_vec, dg_a, dg_b, x_tmp) =
            take_milstein_slices(&mut ws, d, d * m)?;
        system.drift(t, state, f0);
        system.diffusion(t, state, g0);
        accumulate_milstein_update(
            system,
            self.calculus,
            t,
            dt,
            dw,
            None,
            state,
            f0,
            g0,
            g_plus,
            g_minus,
            col_vec,
            dg_a,
            dg_b,
            x_tmp,
        );
        debug_assert!(state.len() >= d);
        Ok(())
    }

    /// Advances `state` by one step `dt` given pre-sampled Wiener increments `dw` (commutative/diagonal mode).
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
            .expect("MilsteinSolver::step_with_increments preconditions violated");
    }
}

type MilsteinScratch<'a, T> = (
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
    &'a mut [T],
);

#[verified_engine::verified]
fn take_milstein_slices<'a, T: Scalar>(
    ws: &mut SolverWorkspace<'a, T>,
    d: usize,
    dm: usize,
) -> Result<MilsteinScratch<'a, T>, DiffeqError> {
    let f0 = ws.try_take_slice(d)?;
    let g0 = ws.try_take_slice(dm)?;
    let g_plus = ws.try_take_slice(dm)?;
    let g_minus = ws.try_take_slice(dm)?;
    let col_vec = ws.try_take_slice(d)?;
    let dg_a = ws.try_take_slice(d)?;
    let dg_b = ws.try_take_slice(d)?;
    let x_tmp = ws.try_take_slice(d)?;
    debug_assert_eq!(x_tmp.len(), d);
    Ok((f0, g0, g_plus, g_minus, col_vec, dg_a, dg_b, x_tmp))
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn accumulate_milstein_update<T: Scalar, S: SdeSystem<T> + ?Sized>(
    system: &S,
    calculus: StochasticCalculus,
    t: T,
    dt: T,
    dw: &[T],
    levy_area: Option<&[T]>,
    state: &mut [T],
    f0: &[T],
    g0: &[T],
    g_plus: &mut [T],
    g_minus: &mut [T],
    col_vec: &mut [T],
    dg_a: &mut [T],
    dg_b: &mut [T],
    x_tmp: &mut [T],
) {
    let d = system.state_dim();
    let m = system.wiener_dim();
    debug_assert!(state.len() >= d && f0.len() >= d);
    let ito_shift = match calculus {
        StochasticCalculus::Ito => dt,
        StochasticCalculus::Stratonovich => T::ZERO,
    };

    for i in 0..d {
        let mut linear_diff = T::ZERO;
        for j in 0..m {
            linear_diff += g0[flatten_2d_index(j, i, m)] * dw[j];
        }
        dg_b[i] = f0[i] * dt + linear_diff;
    }

    for j1 in 0..m {
        for i in 0..d {
            col_vec[i] = g0[flatten_2d_index(j1, i, m)];
        }
        system.diffusion_jacobian_action(t, state, j1, col_vec, dg_a, x_tmp, g_plus, g_minus);
        let w_diag = T::HALF * (dw[j1] * dw[j1] - ito_shift);
        for i in 0..d {
            dg_b[i] += w_diag * dg_a[i];
        }

        for j2 in (j1 + 1)..m {
            let area = levy_area.map_or(T::ZERO, |a| a[flatten_2d_index(j2, j1, m)]);
            system.diffusion_jacobian_action(t, state, j2, col_vec, dg_a, x_tmp, g_plus, g_minus);
            let w_12 = T::HALF * dw[j1] * dw[j2] + area;
            for i in 0..d {
                dg_b[i] += w_12 * dg_a[i];
            }
            for i in 0..d {
                col_vec[i] = g0[flatten_2d_index(j2, i, m)];
            }
            system.diffusion_jacobian_action(t, state, j1, col_vec, dg_a, x_tmp, g_plus, g_minus);
            let w_21 = T::HALF * dw[j1] * dw[j2] - area;
            for i in 0..d {
                dg_b[i] += w_21 * dg_a[i];
                col_vec[i] = g0[flatten_2d_index(j1, i, m)];
            }
        }
    }

    for i in 0..d {
        state[i] += dg_b[i];
    }
}

/// Samples the anti-symmetric Lévy area matrix $A_{j_1, j_2} = \frac{1}{2}(I_{(j_1, j_2)} - I_{(j_2, j_1)})$
/// via Kloeden-Platen truncated Fourier series with `modes` harmonics.
#[verified_engine::verified]
fn sample_levy_area<T: Scalar, R: RngCore + ?Sized>(
    dt: T,
    dw: &[T],
    modes: usize,
    area_out: &mut [T],
    xi: &mut [T],
    zeta: &mut [T],
    rng: &mut R,
) {
    let m = dw.len();
    debug_assert!(area_out.len() >= m * m);
    area_out[..m * m].fill(T::ZERO);
    let sqrt_dt = dt.sqrt();
    let scale_outer = dt / T::TAU;

    for r in 1..=modes {
        let inv_r = T::ONE / T::from_usize(r);
        for j in 0..m {
            let (z0, z1) = sample_standard_normal_pair::<T, R>(rng);
            xi[j] = z0;
            zeta[j] = z1;
        }
        let sqrt_two = T::TWO.sqrt();
        for j1 in 0..m {
            let u1 = sqrt_two * dw[j1] / sqrt_dt.max(T::EPSILON);
            for j2 in (j1 + 1)..m {
                let u2 = sqrt_two * dw[j2] / sqrt_dt.max(T::EPSILON);
                let term = xi[j1] * (zeta[j2] + u2) - xi[j2] * (zeta[j1] + u1);
                let delta = scale_outer * inv_r * term;
                area_out[flatten_2d_index(j2, j1, m)] += delta;
                area_out[flatten_2d_index(j1, j2, m)] -= delta;
            }
        }
    }
}
