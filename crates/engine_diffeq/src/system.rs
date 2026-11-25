// [cite:engine_diffeq]
use math_commons::math_kernel::types::{SolverWorkspace, flatten_2d_index};
use math_scalars::{Dual, RealScalar, Scalar};

const MAX_STACK_SDE_DIM: usize = 32;
const MAX_STACK_SDE_DIFFUSION: usize = 128;

/// Stochastic integral interpretation governing drift-diffusion coupling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
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

    /// Evaluates the exact directional derivative $(D_x G_{\cdot, \text{col}})(t, x) \cdot \text{direction} \in \mathbb{R}^d$
    /// of column `col` of $G(t, x)$ along `direction` into `out_dg` via forward-mode [`Dual<T>`].
    ///
    /// Propagates tangents in a single stack-allocated forward pass (`d <= 32`, `d * m <= 128`, with
    /// optional `alloc` fallback for larger state dimensions), eliminating finite-difference step
    /// underflow in `f32` above `|x| > 32`.
    #[allow(clippy::too_many_arguments)]
    #[verified_engine::verified]
    fn diffusion_jacobian_action(
        &self,
        t: T,
        x: &[T],
        col: usize,
        direction: &[T],
        out_dg: &mut [T],
        _scratch_x: &mut [T],
        _scratch_g_plus: &mut [T],
        _scratch_g_minus: &mut [T],
    ) where
        Self: SdeSystem<Dual<T>>,
    {
        let d = <Self as SdeSystem<T>>::state_dim(self);
        let m = <Self as SdeSystem<T>>::wiener_dim(self);
        let dm = d * m;
        assert!(x.len() >= d && direction.len() >= d && out_dg.len() >= d);
        assert!(col < m, "diffusion column index out of bounds");

        if d <= MAX_STACK_SDE_DIM && dm <= MAX_STACK_SDE_DIFFUSION {
            let mut dual_x = [Dual::constant(T::ZERO); MAX_STACK_SDE_DIM];
            let mut dual_g = [Dual::constant(T::ZERO); MAX_STACK_SDE_DIFFUSION];
            for i in 0..d {
                dual_x[i] = Dual::new(x[i], direction[i]);
            }
            <Self as SdeSystem<Dual<T>>>::diffusion(
                self,
                Dual::constant(t),
                &dual_x[..d],
                &mut dual_g[..dm],
            );
            for (i, out_i) in out_dg.iter_mut().enumerate().take(d) {
                *out_i = dual_g[flatten_2d_index(col, i, m)].eps;
            }
            return;
        }

        #[cfg(feature = "alloc")]
        {
            let mut dual_x = alloc::vec![Dual::constant(T::ZERO); d];
            let mut dual_g = alloc::vec![Dual::constant(T::ZERO); dm];
            for i in 0..d {
                dual_x[i] = Dual::new(x[i], direction[i]);
            }
            <Self as SdeSystem<Dual<T>>>::diffusion(self, Dual::constant(t), &dual_x, &mut dual_g);
            for (i, out_i) in out_dg.iter_mut().enumerate().take(d) {
                *out_i = dual_g[flatten_2d_index(col, i, m)].eps;
            }
        }
        #[cfg(not(feature = "alloc"))]
        {
            panic!(
                "SdeSystem::diffusion_jacobian_action exceeded no_std stack capacity (d={d}, d*m={dm})"
            );
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

/// Evaluates the equivalent Itô drift $f_{\text{Itô}}(t, x)$ into `out_drift`.
///
/// Under [`StochasticCalculus::Ito`], evaluates `system.drift(t, x, out_drift)` directly.
/// Under [`StochasticCalculus::Stratonovich`], adds the exact forward-AD Wong-Zakai correction
/// $\frac{1}{2} \sum_{j=0}^{m-1} (D_x G_{\cdot, j}(t, x)) G_{\cdot, j}(t, x)$.
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
    let d = <S as SdeSystem<T>>::state_dim(system);
    let m = <S as SdeSystem<T>>::wiener_dim(system);
    assert!(x.len() >= d && out_drift.len() >= d);
    system.drift(t, x, out_drift);
    if calculus == StochasticCalculus::Ito || m == 0 {
        return;
    }

    let dm = d * m;
    let mut ws = workspace.reborrow();
    let g_base = ws.take_slice(dm);
    let g_plus = ws.take_slice(dm);
    let g_minus = ws.take_slice(dm);
    let col_vec = ws.take_slice(d);
    let dg_vec = ws.take_slice(d);
    let x_tmp = ws.take_slice(d);

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
}
