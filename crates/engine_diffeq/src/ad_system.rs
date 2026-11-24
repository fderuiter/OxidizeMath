// [cite:engine_diffeq]
// [cite:math_scalars]
//! Zero-allocation forward-mode [`math_scalars::Dual`] directional diffusion derivative adapter [`AdSde`].

use math_commons::math_kernel::types::flatten_2d_index;
use math_scalars::{Dual, Scalar as DiffScalar};

use crate::system::SdeSystem;

/// Maximum state dimension evaluated on inline stack `Dual<T>` buffers in [`AdSde`].
pub const MAX_INLINE_SDE_DIM: usize = 32;

/// Maximum flattened $d \times m$ diffusion entries evaluated on inline stack `Dual<T>` buffers.
pub const MAX_INLINE_SDE_DIFFUSION: usize = 256;

/// Generic differentiable SDE specification evaluated once over any [`DiffScalar`].
///
/// Wrapping an implementation in [`AdSde`] automatically implements [`SdeSystem<T>`] for both
/// `f32` and `f64` and computes exact `Dual<T>` directional diffusion derivatives in
/// [`SdeSystem::diffusion_jacobian_action`], [`crate::MilsteinSolver`], and
/// [`crate::system::eval_ito_effective_drift`] without finite-difference step underflow.
pub trait DifferentiableSde {
    /// Dimension $d$ of the state vector $X_t \in \mathbb{R}^d$.
    fn state_dim(&self) -> usize;

    /// Dimension $m$ of the driving Wiener process $W_t \in \mathbb{R}^m$.
    fn wiener_dim(&self) -> usize;

    /// Evaluates drift $f(t, x)$ into `out_drift` over any differentiable scalar `S`.
    fn drift_scalar<S: DiffScalar>(&self, t: S, x: &[S], out_drift: &mut [S]);

    /// Evaluates row-major $d \times m$ diffusion $G(t, x)$ into `out_diffusion` over scalar `S`.
    fn diffusion_scalar<S: DiffScalar>(&self, t: S, x: &[S], out_diffusion: &mut [S]);
}

/// Zero-cost adapter equipping any [`DifferentiableSde`] with exact `Dual<T>` diffusion Jacobians.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AdSde<S>(
    /// Underlying generic scalar SDE specification.
    pub S,
);

/// Evaluates $(D_x G_{\cdot, \text{col}})(t, x) \cdot \text{direction}$ exactly via `Dual<T>`
/// on stack buffers for systems with $d \le 32$ and $d \times m \le 256$.
#[verified_engine::verified]
pub fn eval_diffusion_jacobian_dual<T: DiffScalar, D: DifferentiableSde + ?Sized>(
    system: &D,
    t: T,
    x: &[T],
    col: usize,
    direction: &[T],
    out_dg: &mut [T],
) {
    let d = system.state_dim();
    let m = system.wiener_dim();
    assert!(d <= MAX_INLINE_SDE_DIM && d * m <= MAX_INLINE_SDE_DIFFUSION);
    assert!(x.len() >= d && direction.len() >= d && out_dg.len() >= d && col < m);

    let mut x_dual = [Dual::<T>::ZERO; MAX_INLINE_SDE_DIM];
    for i in 0..d {
        x_dual[i] = Dual::new(x[i], direction[i]);
    }
    let mut g_dual = [Dual::<T>::ZERO; MAX_INLINE_SDE_DIFFUSION];
    system.diffusion_scalar(Dual::constant(t), &x_dual[..d], &mut g_dual[..d * m]);
    for (i, out_i) in out_dg.iter_mut().enumerate().take(d) {
        *out_i = g_dual[flatten_2d_index(col, i, m)].eps;
    }
}

impl<T, D> SdeSystem<T> for AdSde<D>
where
    T: math_commons::math_kernel::types::Scalar + math_scalars::RealScalar,
    D: DifferentiableSde,
{
    #[inline]
    fn state_dim(&self) -> usize {
        self.0.state_dim()
    }

    #[inline]
    fn wiener_dim(&self) -> usize {
        self.0.wiener_dim()
    }

    #[inline]
    fn drift(&self, t: T, x: &[T], out_drift: &mut [T]) {
        self.0.drift_scalar(t, x, out_drift);
    }

    #[inline]
    fn diffusion(&self, t: T, x: &[T], out_diffusion: &mut [T]) {
        self.0.diffusion_scalar(t, x, out_diffusion);
    }

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
    ) {
        debug_assert!(self.0.state_dim() <= MAX_INLINE_SDE_DIM);
        debug_assert!(self.0.state_dim() * self.0.wiener_dim() <= MAX_INLINE_SDE_DIFFUSION);
        eval_diffusion_jacobian_dual(&self.0, t, x, col, direction, out_dg);
    }
}
