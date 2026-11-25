//! Second-order symplectic Implicit Midpoint (1-stage Gauss-Legendre) ODE solver.
//!
//! Preserves the canonical symplectic 2-form $dq \wedge dp$ on general (including
//! non-separable) Hamiltonian systems via bounded fixed-point iteration capped at
//! [`MAX_FIXED_POINT_ITERS`] (`12`) using pre-allocated state scratch buffers.

// [cite:advanced_linear_algebra]

use super::state::ArrayState;
use super::traits::{OdeSystem, Solver, VectorOperations};

/// Hard ceiling on implicit fixed-point stage iterations per step (`12`).
pub const MAX_FIXED_POINT_ITERS: usize = 12;

/// Second-order symplectic 1-stage Gauss-Legendre Implicit Midpoint integrator.
///
/// Solves the implicit midpoint stage equation
/// $$k_1 = f\!\left(t + \tfrac{1}{2}\Delta t,\; y_n + \tfrac{1}{2}\Delta t\, k_1\right), \qquad y_{n+1} = y_n + \Delta t\, k_1$$
/// via bounded fixed-point iteration in pre-allocated buffers with zero heap allocations.
///
/// Reference: Hairer, Lubich & Wanner (2006), *Geometric Numerical Integration*, DOI: 10.1007/3-540-30666-8; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Unconditionally A-stable, algebraically B-stable, symmetric, and canonical symplectic ($dq \wedge dp$-preserving) on general Hamiltonian systems; fixed-point stage iteration contracts when $\frac{1}{2} L |\Delta t| < 1$.
///
/// # Domain of Validity
/// Valid for smooth stiff or Hamiltonian ODEs where $f$ is locally Lipschitz with $\frac{1}{2} L |\Delta t| < 1$ so stage fixed-point iteration converges within `max_iters` (`<= 12`).
///
/// # Example
///
/// ```
/// # use pure_math::__doctest_facade as math_explorer;
/// use math_explorer::pure_math::analysis::ode::{
///     ArrayState, ImplicitMidpoint, OdeSystem, Solver,
/// };
///
/// struct Harmonic;
/// impl OdeSystem<ArrayState<2>> for Harmonic {
///     fn derivative(&self, _t: f64, s: &ArrayState<2>) -> ArrayState<2> {
///         ArrayState([s.0[1], -s.0[0]])
///     }
/// }
///
/// let mut state = ArrayState([1.0, 0.0]);
/// let mut solver = ImplicitMidpoint::new(&state);
/// for _ in 0..400 {
///     solver.step(&Harmonic, 0.0, &mut state, 0.05);
/// }
/// let amp_sq = state.0[0] * state.0[0] + state.0[1] * state.0[1];
/// assert!((amp_sq - 1.0).abs() < 1e-10);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ImplicitMidpoint<State> {
    initial_state: State,
    stage: State,
    k: State,
    k_next: State,
    max_iters: usize,
}

impl<State: Clone> ImplicitMidpoint<State> {
    /// Default fixed-point stage iterations per step (`10`, within [`MAX_FIXED_POINT_ITERS`]).
    pub const DEFAULT_MAX_ITERS: usize = 10;

    /// Upper cap on fixed-point stage iterations (`12`).
    pub const MAX_ALLOWED_ITERS: usize = MAX_FIXED_POINT_ITERS;

    /// Constructs a new `ImplicitMidpoint` solver with pre-allocated scratch buffers.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(example_state: &State) -> Self {
        Self::with_max_iters(example_state, Self::DEFAULT_MAX_ITERS)
    }

    /// Constructs a new `ImplicitMidpoint` solver with a clamped fixed-point iteration budget (`1..=12`).
    #[must_use]
    #[verified_engine::verified]
    pub fn with_max_iters(example_state: &State, max_iters: usize) -> Self {
        Self {
            initial_state: example_state.clone(),
            stage: example_state.clone(),
            k: example_state.clone(),
            k_next: example_state.clone(),
            max_iters: max_iters.clamp(1, Self::MAX_ALLOWED_ITERS),
        }
    }

    /// Returns the configured fixed-point iteration cap (`1..=12`).
    #[must_use]
    #[verified_engine::verified]
    pub const fn max_iters(&self) -> usize {
        self.max_iters
    }
}

impl<const N: usize> ImplicitMidpoint<ArrayState<N>> {
    /// Advances `state` by `dt`, rolling back to `y_n` and returning `false` if any component becomes non-finite.
    #[verified_engine::verified]
    pub fn try_step_array<S>(
        &mut self,
        system: &S,
        t: f64,
        state: &mut ArrayState<N>,
        dt: f64,
    ) -> bool
    where
        S: OdeSystem<ArrayState<N>> + ?Sized,
    {
        if !dt.is_finite() || !state.is_finite() {
            return false;
        }
        self.step(system, t, state, dt);
        if state.is_finite() {
            true
        } else {
            state.copy_from(&self.initial_state);
            false
        }
    }
}

impl<State: VectorOperations> Solver<State> for ImplicitMidpoint<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        self.initial_state.copy_from(state);
        system.derivative_in_place(t, &self.initial_state, &mut self.k);

        let half_dt = 0.5 * dt;
        let t_mid = t + half_dt;
        let iters = self.max_iters.clamp(1, MAX_FIXED_POINT_ITERS);

        for _ in 0..iters {
            self.stage
                .copy_from_scaled(&self.initial_state, &self.k, half_dt);
            system.derivative_in_place(t_mid, &self.stage, &mut self.k_next);
            self.k.copy_from(&self.k_next);
        }

        state.copy_from_scaled(&self.initial_state, &self.k, dt);
    }
}
