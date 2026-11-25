//! Pre-allocated runtime-configurable ODE solver workspace dispatching [`IntegrationMethod`].

use super::IntegrationMethod;
use super::implicit_midpoint::ImplicitMidpoint;
use super::solvers::{Euler, Heun, RungeKutta4};
use super::symplectic::GaussLegendre4;
use super::traits::{OdeSystem, Solver, VectorOperations};

/// Pre-allocated multi-method ODE solver workspace $\mathbf{y}_{n+1} = \Phi_{\Delta t}^{(m)}(t_n, \mathbf{y}_n)$
/// dispatching [`IntegrationMethod`] without post-initialization heap allocations.
///
/// Reference: Hairer, Norsett & Wanner (1993), *Solving Ordinary Differential Equations I*,
/// DOI: 10.1007/978-3-540-78862-1; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Inherits the exact stability polynomial $R_m(\lambda \Delta t)$ of the selected [`IntegrationMethod`]
/// (`Euler`, `Heun`, `RungeKutta4`, A-stable `ImplicitMidpoint`, or A-stable `GaussLegendre4`).
///
/// # Domain of Validity
/// Valid for locally Lipschitz ODE systems $\dot{\mathbf{y}} = \mathbf{f}(t, \mathbf{y})$ over finite step sizes
/// $\Delta t$ within the selected method's stability or fixed-point contraction region.
#[derive(Debug, Clone, PartialEq)]
pub struct PreallocatedOdeSolver<State> {
    euler: Euler<State>,
    heun: Heun<State>,
    rk4: RungeKutta4<State>,
    implicit_midpoint: ImplicitMidpoint<State>,
    gauss_legendre4: GaussLegendre4<State>,
}

impl<State: Clone> PreallocatedOdeSolver<State> {
    /// Pre-allocates scratch buffers for all supported [`IntegrationMethod`] variants from `example_state`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(example_state: &State) -> Self {
        Self {
            euler: Euler::new(example_state),
            heun: Heun::new(example_state),
            rk4: RungeKutta4::new(example_state),
            implicit_midpoint: ImplicitMidpoint::new(example_state),
            gauss_legendre4: GaussLegendre4::new(example_state),
        }
    }
}

impl<State: VectorOperations> PreallocatedOdeSolver<State> {
    /// Advances `state` in place by `dt` at time `t` using the specified `method` and pre-allocated buffers.
    #[verified_engine::verified]
    pub fn step_method<S>(
        &mut self,
        method: IntegrationMethod,
        system: &S,
        t: f64,
        state: &mut State,
        dt: f64,
    ) where
        S: OdeSystem<State> + ?Sized,
    {
        match method {
            IntegrationMethod::Euler => self.euler.step(system, t, state, dt),
            IntegrationMethod::Heun => self.heun.step(system, t, state, dt),
            IntegrationMethod::RungeKutta4 => self.rk4.step(system, t, state, dt),
            IntegrationMethod::ImplicitMidpoint => {
                self.implicit_midpoint.step(system, t, state, dt);
            }
            IntegrationMethod::GaussLegendre4 => {
                self.gauss_legendre4.step(system, t, state, dt);
            }
        }
    }
}
