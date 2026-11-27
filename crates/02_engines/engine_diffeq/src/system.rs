// [cite:engine_diffeq]
use crate::error::{DiffeqError, carve_scalars};
use math_commons::math_kernel::types::{Scalar, flatten_2d_index};
use math_memory::SolverWorkspace;

/// Stochastic integral interpretation governing drift-diffusion coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StochasticCalculus {
    /// Non-anticipating left-point Itô integral $\int G(t, X_t) \, dW_t$.
    #[default]
    Ito,
    /// Midpoint Stratonovich integral $\int G(t, X_t) \circ dW_t$ obeying the classical chain rule.
    Stratonovich,
}

/// Multivariate stochastic differential equation $dX_t = f(t, X_t)\,dt + G(t, X_t)\,dW_t$
/// (or $\circ dW_t$ under [`StochasticCalculus::Stratonovich`]).
pub trait SdeSystem<T: Scalar> {
    /// Dimension $d$ of the state vector $X_t \in \mathbb{R}^d$.
    fn state_dim(&self) -> usize;

    /// Dimension $m$ of the driving Wiener process $W_t \in \mathbb{R}^m$.
    fn wiener_dim(&self) -> usize;

    /// Evaluates the drift vector $f(t, x) \in \mathbb{R}^d$ into `out_drift` (`len == state_dim()`).
    fn drift(&self, t: T, x: &[T], out_drift: &mut [T]);

    /// Evaluates the row-major $d \times m$ diffusion matrix $G(t, x)$ into `out_diffusion`
    /// (`out_diffusion[i * wiener_dim() + j] = G_{i, j}(t, x)`).
    fn diffusion(&self, t: T, x: &[T], out_diffusion: &mut [T]);

    /// Evaluates the directional derivative $(D_x G_{\cdot, \text{col}})(t, x) \cdot \text{direction} \in \mathbb{R}^d$
    /// of column `col` of $G(t, x)$ along `direction` into `out_dg`.
    ///
    /// By default, uses a second-order central finite-difference quotient with caller-provided scratch
    /// slices (`scratch_x` of length $d$, `scratch_g_plus` and `scratch_g_minus` of length $d \times m$).
    /// Systems with known analytical diffusion derivatives may override this method.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    fn diffusion_jacobian_action(
        &self,
        t: T,
        x: &[T],
        col: usize,
        direction: &[T],
        out_dg: &mut [T],
        scratch_x: &mut [T],
        scratch_g_plus: &mut [T],
        scratch_g_minus: &mut [T],
    ) {
        let d = self.state_dim();
        let m = self.wiener_dim();
        debug_assert!(x.len() >= d && direction.len() >= d && out_dg.len() >= d);
        debug_assert!(col < m);

        let mut norm_sq = T::ZERO;
        for &v in &direction[..d] {
            norm_sq += v * v;
        }
        if norm_sq <= T::ZERO {
            out_dg[..d].fill(T::ZERO);
            return;
        }

        let eps_base = T::from_f64(1e-6);
        let inv_two_eps = T::HALF / eps_base;
        for i in 0..d {
            scratch_x[i] = x[i] + eps_base * direction[i];
        }
        self.diffusion(t, &scratch_x[..d], scratch_g_plus);

        for i in 0..d {
            scratch_x[i] = x[i] - eps_base * direction[i];
        }
        self.diffusion(t, &scratch_x[..d], scratch_g_minus);

        for (i, out_i) in out_dg.iter_mut().enumerate().take(d) {
            let idx = flatten_2d_index(col, i, m);
            *out_i = (scratch_g_plus[idx] - scratch_g_minus[idx]) * inv_two_eps;
        }
    }
}

/// Potential energy gradient contract $\nabla V(q)$ for multivariate underdamped Langevin dynamics.
pub trait LangevinSystem<T: Scalar> {
    /// Configuration dimension $d$ ($q, p \in \mathbb{R}^d$).
    fn dim(&self) -> usize;

    /// Evaluates $\nabla V(q) \in \mathbb{R}^d$ into `out_grad` (`len >= dim()`).
    fn grad_potential(&self, q: &[T], out_grad: &mut [T]);
}

/// Returns the scratch capacity (`3 * d * m + 3 * d`) required by [`eval_ito_effective_drift`].
#[inline]
pub const fn ito_drift_scratch_len(state_dim: usize, wiener_dim: usize) -> usize {
    3 * state_dim * wiener_dim + 3 * state_dim
}

/// Fallibly evaluates the equivalent Itô drift $f_{\text{Itô}}(t, x)$ into `out_drift`.
///
/// # Errors
/// Returns [`DiffeqError::BufferTooSmall`] if `x` or `out_drift` is shorter than `system.state_dim()`,
/// or [`DiffeqError::Math`] if `workspace` lacks [`ito_drift_scratch_len`] scalars under Stratonovich calculus.
#[verified_engine::verified]
pub fn try_eval_ito_effective_drift<T: Scalar, S: SdeSystem<T> + ?Sized>(
    system: &S,
    calculus: StochasticCalculus,
    t: T,
    x: &[T],
    out_drift: &mut [T],
    workspace: &mut (impl SolverWorkspace + ?Sized),
) -> Result<(), DiffeqError> {
    let d = system.state_dim();
    let m = system.wiener_dim();
    if x.len() < d {
        return Err(DiffeqError::BufferTooSmall {
            buffer: "x",
            required: d,
            actual: x.len(),
        });
    }
    if out_drift.len() < d {
        return Err(DiffeqError::BufferTooSmall {
            buffer: "out_drift",
            required: d,
            actual: out_drift.len(),
        });
    }
    system.drift(t, x, out_drift);
    if calculus == StochasticCalculus::Ito || m == 0 {
        debug_assert!(out_drift.len() >= d);
        return Ok(());
    }

    let dm = d * m;
    let mut ws = workspace.reborrow();
    let mut cur = ws.cursor();
    let g_base = carve_scalars::<T>(&mut cur, dm)?;
    let g_plus = carve_scalars::<T>(&mut cur, dm)?;
    let g_minus = carve_scalars::<T>(&mut cur, dm)?;
    let col_vec = carve_scalars::<T>(&mut cur, d)?;
    let dg_vec = carve_scalars::<T>(&mut cur, d)?;
    let x_tmp = carve_scalars::<T>(&mut cur, d)?;

    system.diffusion(t, x, g_base);
    for j in 0..m {
        for i in 0..d {
            col_vec[i] = g_base[flatten_2d_index(j, i, m)];
        }
        system.diffusion_jacobian_action(t, x, j, col_vec, dg_vec, x_tmp, g_plus, g_minus);
        for i in 0..d {
            out_drift[i] += T::HALF * dg_vec[i];
        }
    }
    debug_assert!(out_drift.len() >= d);
    Ok(())
}

/// Evaluates the equivalent Itô drift $f_{\text{Itô}}(t, x)$ into `out_drift`.
///
/// Under [`StochasticCalculus::Ito`], evaluates `system.drift(t, x, out_drift)` directly.
/// Under [`StochasticCalculus::Stratonovich`], adds the Wong-Zakai correction
/// $\frac{1}{2} \sum_{j=0}^{m-1} (D_x G_{\cdot, j}(t, x)) G_{\cdot, j}(t, x)$.
///
/// # Panics
/// Panics if `x`, `out_drift`, or `workspace` is undersized; prefer [`try_eval_ito_effective_drift`].
#[verified_engine::verified]
pub fn eval_ito_effective_drift<T: Scalar, S: SdeSystem<T> + ?Sized>(
    system: &S,
    calculus: StochasticCalculus,
    t: T,
    x: &[T],
    out_drift: &mut [T],
    workspace: &mut (impl SolverWorkspace + ?Sized),
) {
    try_eval_ito_effective_drift(system, calculus, t, x, out_drift, workspace)
        .expect("eval_ito_effective_drift preconditions violated");
}
