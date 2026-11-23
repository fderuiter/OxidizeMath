//! Explicit fixed-step ODE solvers (`Euler`, `Heun`, `RungeKutta4`) and `SolverWorkspace` slice kernels.
// [cite:engine_diffeq]

use super::traits::{OdeSystem, SliceOdeSystem, Solver, VectorOperations};
use crate::error::DiffeqError;
use crate::workspace::SolverWorkspace;

/// Forward Euler first-order explicit ODE solver.
#[derive(Debug, Clone)]
pub struct Euler<State> {
    k1: State,
}

impl<State: VectorOperations> Euler<State> {
    /// Creates a new `Euler` solver pre-initializing scratch state from `template`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            k1: template.clone(),
        }
    }
}

impl<State: VectorOperations> Solver<State> for Euler<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        system.derivative_in_place(t, state, &mut self.k1);
        state.scale_add(&self.k1, dt);
    }
}

/// Heun's method (explicit trapezoidal / second-order Runge-Kutta) ODE solver.
#[derive(Debug, Clone)]
pub struct Heun<State> {
    k1: State,
    k2: State,
    temp: State,
}

impl<State: VectorOperations> Heun<State> {
    /// Creates a new `Heun` solver pre-initializing scratch states from `template`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            k1: template.clone(),
            k2: template.clone(),
            temp: template.clone(),
        }
    }
}

impl<State: VectorOperations> Solver<State> for Heun<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        system.derivative_in_place(t, state, &mut self.k1);
        self.temp.copy_from_scaled(state, &self.k1, dt);
        system.derivative_in_place(t + dt, &self.temp, &mut self.k2);
        state.scale_add(&self.k1, 0.5 * dt);
        state.scale_add(&self.k2, 0.5 * dt);
    }
}

/// Classic fourth-order Runge-Kutta (`RK4`) explicit ODE solver.
#[derive(Debug, Clone)]
pub struct RungeKutta4<State> {
    k1: State,
    k2: State,
    k3: State,
    k4: State,
    temp: State,
}

impl<State: VectorOperations> RungeKutta4<State> {
    /// Creates a new `RungeKutta4` solver pre-initializing stage buffers from `template`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(template: &State) -> Self {
        Self {
            k1: template.clone(),
            k2: template.clone(),
            k3: template.clone(),
            k4: template.clone(),
            temp: template.clone(),
        }
    }
}

impl<State: VectorOperations> Solver<State> for RungeKutta4<State> {
    #[verified_engine::verified]
    fn step<S>(&mut self, system: &S, t: f64, state: &mut State, dt: f64)
    where
        S: OdeSystem<State> + ?Sized,
    {
        let half_dt = 0.5 * dt;
        system.derivative_in_place(t, state, &mut self.k1);

        self.temp.copy_from_scaled(state, &self.k1, half_dt);
        system.derivative_in_place(t + half_dt, &self.temp, &mut self.k2);

        self.temp.copy_from_scaled(state, &self.k2, half_dt);
        system.derivative_in_place(t + half_dt, &self.temp, &mut self.k3);

        self.temp.copy_from_scaled(state, &self.k3, dt);
        system.derivative_in_place(t + dt, &self.temp, &mut self.k4);

        let dt_6 = dt / 6.0;
        let dt_3 = dt / 3.0;
        state.scale_add(&self.k1, dt_6);
        state.scale_add(&self.k2, dt_3);
        state.scale_add(&self.k3, dt_3);
        state.scale_add(&self.k4, dt_6);
    }
}

/// Advances a caller-owned slice `state` by `dt` via classic `RK4` using `workspace` (`>= 5 * state.len()`).
#[verified_engine::verified]
pub fn step_slice_rk4<S: SliceOdeSystem + ?Sized>(
    system: &S,
    t: f64,
    state: &mut [f64],
    dt: f64,
    workspace: &mut SolverWorkspace<'_>,
) -> Result<(), DiffeqError> {
    let n = state.len();
    let scratch = workspace.try_slice_mut(5 * n)?;
    let (k1, rest) = scratch.split_at_mut(n);
    let (k2, rest) = rest.split_at_mut(n);
    let (k3, rest) = rest.split_at_mut(n);
    let (k4, tmp) = rest.split_at_mut(n);

    let half_dt = 0.5 * dt;
    system.derivative_slice(t, state, k1);
    for i in 0..n {
        tmp[i] = state[i] + half_dt * k1[i];
    }
    system.derivative_slice(t + half_dt, tmp, k2);
    for i in 0..n {
        tmp[i] = state[i] + half_dt * k2[i];
    }
    system.derivative_slice(t + half_dt, tmp, k3);
    for i in 0..n {
        tmp[i] = state[i] + dt * k3[i];
    }
    system.derivative_slice(t + dt, tmp, k4);

    let dt_6 = dt / 6.0;
    for i in 0..n {
        state[i] += dt_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
    Ok(())
}
