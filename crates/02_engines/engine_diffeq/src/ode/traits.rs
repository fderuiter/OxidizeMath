//! Core `#![no_std]` traits for ODE vector spaces, dynamical systems, and solvers.
// [cite:engine_diffeq]

use core::ops::{Add, AddAssign, Mul, MulAssign};

/// Vector-space arithmetic required by explicit and implicit numerical integrators.
pub trait VectorOperations:
    Sized + Clone + Add<Output = Self> + Mul<f64, Output = Self> + AddAssign + MulAssign<f64>
{
    /// Adds `other` scaled by `scale` to `self`: `self += other * scale`.
    #[verified_engine::verified]
    fn scale_add(&mut self, other: &Self, scale: f64);

    /// Copies the contents of `other` into `self` in-place.
    #[verified_engine::verified]
    fn copy_from(&mut self, other: &Self);

    /// Fused operation `self = source + other * scale` avoiding intermediate writes.
    #[verified_engine::verified]
    fn copy_from_scaled(&mut self, source: &Self, other: &Self, scale: f64) {
        self.copy_from(source);
        self.scale_add(other, scale);
    }
}

/// Component-indexable state vector supporting finite-difference Jacobians, norms, and LU updates.
pub trait OdeState: VectorOperations {
    /// Returns the number of scalar components in `self`.
    #[verified_engine::verified]
    fn state_dim(&self) -> usize;

    /// Reads scalar component `idx` (`0 <= idx < state_dim()`).
    #[verified_engine::verified]
    fn component(&self, idx: usize) -> f64;

    /// Writes `value` into scalar component `idx` (`0 <= idx < state_dim()`).
    #[verified_engine::verified]
    fn set_component(&mut self, idx: usize, value: f64);
}

/// System of Ordinary Differential Equations $\frac{d\vec{y}}{dt} = f(t, \vec{y})$.
pub trait OdeSystem<State: VectorOperations> {
    /// Computes the time derivative $\frac{d\vec{y}}{dt}$ at `(t, state)`.
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &State) -> State;

    /// Computes the time derivative $\frac{d\vec{y}}{dt}$ in-place into `out`.
    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &State, out: &mut State) {
        *out = self.derivative(t, state);
    }
}

impl<State: VectorOperations, T: OdeSystem<State> + ?Sized> OdeSystem<State> for &T {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &State) -> State {
        (**self).derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &State, out: &mut State) {
        (**self).derivative_in_place(t, state, out);
    }
}

/// Slice-native ODE system $\frac{d\vec{y}}{dt} = f(t, \vec{y})$ operating directly on caller slices.
pub trait SliceOdeSystem {
    /// Evaluates $\frac{d\vec{y}}{dt}$ from `state` directly into `out` (both of length `dim`).
    #[verified_engine::verified]
    fn derivative_slice(&self, t: f64, state: &[f64], out: &mut [f64]);
}

/// Numerical ODE solver strategy advancing `State` in-place.
pub trait Solver<State: VectorOperations> {
    /// Advances `state` from `t` to `t + dt` in-place.
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized;
}

/// Extension trait providing functional out-of-place stepping on top of [`Solver`].
pub trait SolverExt<State: VectorOperations>: Solver<State> {
    /// Clones `state`, advances it by `dt` at time `t`, and returns the updated state.
    #[verified_engine::verified]
    fn solve<S>(&mut self, system: &S, t: f64, state: &State, dt: f64) -> State
    where
        S: OdeSystem<State> + ?Sized;
}

impl<State: VectorOperations, T: Solver<State> + ?Sized> SolverExt<State> for T {
    #[verified_engine::verified]
    fn solve<S>(&mut self, system: &S, t: f64, state: &State, dt: f64) -> State
    where
        S: OdeSystem<State> + ?Sized,
    {
        let mut new_state = state.clone();
        self.step(system, t, &mut new_state, dt);
        new_state
    }
}
