// [cite:engine_diffeq]
use crate::error::DiffeqError;
use math_commons::math_kernel::types::{SolverWorkspace, flatten_2d_index};
use math_scalars::{Dual, RealScalar, Scalar};

const MAX_STACK_SDE_DIM: usize = 32;
const MAX_STACK_SDE_DIFFUSION: usize = 128;

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
///
/// Implementing `SdeSystem<S>` generically over `S: math_scalars::Scalar` automatically enables
/// exact zero-allocation `Dual<T>` directional diffusion derivatives in [`MilsteinSolver`](crate::MilsteinSolver)
/// and Stratonovich [`Sriw1Solver`](crate::Sriw1Solver) without finite-difference step underflow.
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

    /// Fallibly evaluates the exact directional derivative $(D_x G_{\cdot, \text{col}})(t, x) \cdot \text{direction} \in \mathbb{R}^d$
    /// of column `col` of $G(t, x)$ along `direction` into `out_dg` via forward-mode [`Dual<T>`].
    ///
    /// # Errors
    /// Returns [`DiffeqError::BufferTooSmall`] if `x`, `direction`, or `out_dg` has length `< state_dim()`,
    /// [`DiffeqError::InvalidParameter`] if `col >= wiener_dim()`, or [`DiffeqError::WorkspaceAllocationFailed`]
    /// if dimensions exceed the `#![no_std]` stack budget (`d > 32` or `d * m > 128`) without `alloc`.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    fn try_diffusion_jacobian_action(
        &self,
        t: T,
        x: &[T],
        col: usize,
        direction: &[T],
        out_dg: &mut [T],
        _scratch_x: &mut [T],
        _scratch_g_plus: &mut [T],
        _scratch_g_minus: &mut [T],
    ) -> Result<(), DiffeqError>
    where
        Self: SdeSystem<Dual<T>>,
    {
        let d = <Self as SdeSystem<T>>::state_dim(self);
        let m = <Self as SdeSystem<T>>::wiener_dim(self);
        validate_jacobian_slices(d, m, col, x, direction, out_dg)?;
        eval_dual_diffusion_action(self, d, m, t, x, col, direction, out_dg)
    }

    /// Evaluates the exact directional derivative $(D_x G_{\cdot, \text{col}})(t, x) \cdot \text{direction} \in \mathbb{R}^d$
    /// of column `col` of $G(t, x)$ along `direction` into `out_dg` via forward-mode [`Dual<T>`].
    ///
    /// # Panics
    /// Panics if slices are undersized, `col >= wiener_dim()`, or dimensions exceed `#![no_std]` stack
    /// capacity without `alloc`; prefer [`Self::try_diffusion_jacobian_action`].
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
    ) where
        Self: SdeSystem<Dual<T>>,
    {
        self.try_diffusion_jacobian_action(
            t,
            x,
            col,
            direction,
            out_dg,
            scratch_x,
            scratch_g_plus,
            scratch_g_minus,
        )
        .expect("SdeSystem::diffusion_jacobian_action preconditions violated");
    }
}

#[verified_engine::verified]
fn validate_jacobian_slices<T: Scalar>(
    d: usize,
    m: usize,
    col: usize,
    x: &[T],
    direction: &[T],
    out_dg: &[T],
) -> Result<(), DiffeqError> {
    if x.len() < d {
        return Err(DiffeqError::BufferTooSmall {
            buffer: "x",
            required: d,
            actual: x.len(),
        });
    }
    if direction.len() < d {
        return Err(DiffeqError::BufferTooSmall {
            buffer: "buffer",
            required: d,
            actual: direction.len(),
        });
    }
    if out_dg.len() < d {
        return Err(DiffeqError::BufferTooSmall {
            buffer: "buffer",
            required: d,
            actual: out_dg.len(),
        });
    }
    if col >= m {
        return Err(DiffeqError::InvalidParameter {
            parameter: "parameter",
        });
    }
    debug_assert!(x.len() >= d && col < m);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[verified_engine::verified]
fn eval_dual_diffusion_action<T, S>(
    system: &S,
    d: usize,
    m: usize,
    t: T,
    x: &[T],
    col: usize,
    direction: &[T],
    out_dg: &mut [T],
) -> Result<(), DiffeqError>
where
    T: Scalar,
    S: SdeSystem<T> + SdeSystem<Dual<T>> + ?Sized,
{
    let dm = d * m;
    debug_assert!(out_dg.len() >= d && col < m);
    if d <= MAX_STACK_SDE_DIM && dm <= MAX_STACK_SDE_DIFFUSION {
        let mut dual_x = [Dual::constant(T::ZERO); MAX_STACK_SDE_DIM];
        let mut dual_g = [Dual::constant(T::ZERO); MAX_STACK_SDE_DIFFUSION];
        for i in 0..d {
            dual_x[i] = Dual::new(x[i], direction[i]);
        }
        <S as SdeSystem<Dual<T>>>::diffusion(
            system,
            Dual::constant(t),
            &dual_x[..d],
            &mut dual_g[..dm],
        );
        for (i, out_i) in out_dg.iter_mut().enumerate().take(d) {
            *out_i = dual_g[flatten_2d_index(col, i, m)].eps;
        }
        return Ok(());
    }

    #[cfg(feature = "alloc")]
    {
        let mut dual_x = alloc::vec::Vec::new();
        dual_x
            .try_reserve_exact(d)
            .map_err(|_| DiffeqError::WorkspaceAllocationFailed { capacity: d })?;
        dual_x.resize(d, Dual::constant(T::ZERO));
        let mut dual_g = alloc::vec::Vec::new();
        dual_g
            .try_reserve_exact(dm)
            .map_err(|_| DiffeqError::WorkspaceAllocationFailed { capacity: dm })?;
        dual_g.resize(dm, Dual::constant(T::ZERO));
        for i in 0..d {
            dual_x[i] = Dual::new(x[i], direction[i]);
        }
        <S as SdeSystem<Dual<T>>>::diffusion(system, Dual::constant(t), &dual_x, &mut dual_g);
        for (i, out_i) in out_dg.iter_mut().enumerate().take(d) {
            *out_i = dual_g[flatten_2d_index(col, i, m)].eps;
        }
        Ok(())
    }
    #[cfg(not(feature = "alloc"))]
    {
        Err(DiffeqError::WorkspaceAllocationFailed { capacity: dm })
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
pub fn try_eval_ito_effective_drift<T, S>(
    system: &S,
    calculus: StochasticCalculus,
    t: T,
    x: &[T],
    out_drift: &mut [T],
    workspace: &mut SolverWorkspace<'_, T>,
) -> Result<(), DiffeqError>
where
    T: RealScalar,
    S: SdeSystem<T> + SdeSystem<Dual<T>> + ?Sized,
{
    let d = <S as SdeSystem<T>>::state_dim(system);
    let m = <S as SdeSystem<T>>::wiener_dim(system);
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
    let g_base = ws.try_take_slice(dm)?;
    let g_plus = ws.try_take_slice(dm)?;
    let g_minus = ws.try_take_slice(dm)?;
    let col_vec = ws.try_take_slice(d)?;
    let dg_vec = ws.try_take_slice(d)?;
    let x_tmp = ws.try_take_slice(d)?;

    system.diffusion(t, x, g_base);
    for j in 0..m {
        for i in 0..d {
            col_vec[i] = g_base[flatten_2d_index(j, i, m)];
        }
        system.try_diffusion_jacobian_action(t, x, j, col_vec, dg_vec, x_tmp, g_plus, g_minus)?;
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
/// Under [`StochasticCalculus::Stratonovich`], adds the exact forward-AD Wong-Zakai correction
/// $\frac{1}{2} \sum_{j=0}^{m-1} (D_x G_{\cdot, j}(t, x)) G_{\cdot, j}(t, x)$.
///
/// # Panics
/// Panics if `x`, `out_drift`, or `workspace` is undersized; prefer [`try_eval_ito_effective_drift`].
#[verified_engine::verified]
pub fn eval_ito_effective_drift<T, S>(
    system: &S,
    calculus: StochasticCalculus,
    t: T,
    x: &[T],
    out_drift: &mut [T],
    workspace: &mut SolverWorkspace<'_, T>,
) where
    T: RealScalar,
    S: SdeSystem<T> + SdeSystem<Dual<T>> + ?Sized,
{
    try_eval_ito_effective_drift(system, calculus, t, x, out_drift, workspace)
        .expect("eval_ito_effective_drift preconditions violated");
}
