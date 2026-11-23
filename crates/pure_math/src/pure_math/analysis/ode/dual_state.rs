//! Stack-allocated forward-mode [`engine_symbolic::Dual`] ODE state vector and exact RK4 state-transition
//! Jacobian propagation for zero-allocation Extended Kalman Filtering and sensitivity analysis.
// [cite:graph_parameters_rust]

use std::ops::{Add, AddAssign, Mul, MulAssign};

use engine_symbolic::Dual;

use super::solvers::RungeKutta4;
use super::traits::{OdeSystem, Solver, VectorOperations};

/// Zero-allocation fixed-size forward-mode [`Dual<f64>`] ODE state vector implementing
/// [`VectorOperations`] so standard integrators ([`RungeKutta4`], [`super::Heun`], [`super::Euler`])
/// propagate exact tangent sensitivities alongside primal trajectories.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualArrayState<const N: usize>(pub [Dual<f64>; N]);

impl<const N: usize> Default for DualArrayState<N> {
    #[verified_engine::verified]
    fn default() -> Self {
        Self([Dual::ZERO; N])
    }
}

impl<const N: usize> DualArrayState<N> {
    /// Constructs a zero-initialized `DualArrayState<N>`.
    #[must_use]
    #[verified_engine::verified]
    pub const fn zeros() -> Self {
        Self([Dual::ZERO; N])
    }

    /// Seeds `state` with unit tangent along `active_col` (`dx_i / dx_{active_col} = \delta_{i, active_col}`).
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_column(state: &[f64; N], active_col: usize) -> Self {
        let mut out = [Dual::ZERO; N];
        let mut i = 0;
        while i < N {
            out[i] = Dual::new(state[i], if i == active_col { 1.0 } else { 0.0 });
            i += 1;
        }
        Self(out)
    }
}

impl<const N: usize> Add for DualArrayState<N> {
    type Output = Self;

    #[verified_engine::verified]
    fn add(self, rhs: Self) -> Self {
        let mut arr = [Dual::ZERO; N];
        for (i, slot) in arr.iter_mut().enumerate() {
            *slot = self.0[i] + rhs.0[i];
        }
        Self(arr)
    }
}

impl<const N: usize> AddAssign for DualArrayState<N> {
    #[verified_engine::verified]
    fn add_assign(&mut self, rhs: Self) {
        for (a, b) in self.0.iter_mut().zip(rhs.0.iter()) {
            *a += *b;
        }
    }
}

impl<const N: usize> Mul<f64> for DualArrayState<N> {
    type Output = Self;

    #[verified_engine::verified]
    fn mul(self, scalar: f64) -> Self {
        let mut arr = [Dual::ZERO; N];
        for (i, slot) in arr.iter_mut().enumerate() {
            *slot = self.0[i] * scalar;
        }
        Self(arr)
    }
}

impl<const N: usize> MulAssign<f64> for DualArrayState<N> {
    #[verified_engine::verified]
    fn mul_assign(&mut self, scalar: f64) {
        for slot in &mut self.0 {
            *slot = *slot * scalar;
        }
    }
}

impl<const N: usize> VectorOperations for DualArrayState<N> {
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64) {
        for (a, b) in self.0.iter_mut().zip(other.0.iter()) {
            *a += *b * scale;
        }
    }

    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self) {
        self.0 = other.0;
    }

    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        for ((dst, src), oth) in self.0.iter_mut().zip(source.0.iter()).zip(other.0.iter()) {
            *dst = *src + *oth * scale;
        }
    }
}

/// Evaluates the continuous-time ODE right-hand side $f(t, x)$ and its exact Jacobian
/// $J_{ij} = \frac{\partial f_i}{\partial x_j}(t, x)$ on stack buffers with zero heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn ode_rhs_dual_jacobian<const N: usize, S>(
    system: &S,
    t: f64,
    state: &[f64; N],
) -> ([f64; N], [[f64; N]; N])
where
    S: OdeSystem<DualArrayState<N>> + ?Sized,
{
    let mut rhs = [0.0_f64; N];
    let mut jac = [[0.0_f64; N]; N];
    let mut deriv = DualArrayState::<N>::zeros();

    let mut col = 0;
    while col < N {
        let seeded = DualArrayState::seed_column(state, col);
        system.derivative_in_place(t, &seeded, &mut deriv);
        let mut row = 0;
        while row < N {
            if col == 0 {
                rhs[row] = deriv.0[row].re;
            }
            jac[row][col] = deriv.0[row].eps;
            row += 1;
        }
        col += 1;
    }
    (rhs, jac)
}

/// Propagates one 4th-order Runge-Kutta step $x_{k+1} = \Phi_{\text{RK4}}(t, x_k, \Delta t)$ and
/// computes its exact discrete state-transition Jacobian $F_{ij} = \frac{\partial x_{k+1,i}}{\partial x_{k,j}}$
/// via forward-mode [`Dual<f64>`] propagation through [`RungeKutta4`] with zero heap allocations.
#[must_use]
#[verified_engine::verified]
pub fn rk4_step_dual_jacobian<const N: usize, S>(
    system: &S,
    t: f64,
    state: &[f64; N],
    dt: f64,
) -> ([f64; N], [[f64; N]; N])
where
    S: OdeSystem<DualArrayState<N>> + ?Sized,
{
    if N == 0 {
        return ([0.0; N], [[0.0; N]; N]);
    }
    let mut next_state = [0.0_f64; N];
    let mut f_jac = [[0.0_f64; N]; N];
    let mut rk4 = RungeKutta4::new(&DualArrayState::<N>::zeros());

    let mut col = 0;
    while col < N {
        let mut work = DualArrayState::seed_column(state, col);
        rk4.step(system, t, &mut work, dt);
        let mut row = 0;
        while row < N {
            if col == 0 {
                next_state[row] = work.0[row].re;
            }
            f_jac[row][col] = work.0[row].eps;
            row += 1;
        }
        col += 1;
    }
    (next_state, f_jac)
}
