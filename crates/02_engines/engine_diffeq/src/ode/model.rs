//! Generic `OdeModel` container coupling state, governing dynamics, solver, and elapsed time `t`.
// [cite:engine_diffeq]

use super::stepper::TimeStepper;
use super::traits::{OdeSystem, Solver, VectorOperations};
#[cfg(feature = "std")]
use std::prelude::rust_2024::*;

/// Generic container for an ODE simulation combining `State`, `Dynamics`, a `Solver`, and elapsed time `t`.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "std",
    derive(verified_engine::Theory),
    theory(
        description = "An Ordinary Differential Equation (ODE) model encapsulates a dynamical system's state, its governing differential equations, and a numerical solver to simulate its evolution over time.",
        citation = "Numerical Recipes: The Art of Scientific Computing (Press et al., 2007)"
    )
)]
pub struct OdeModel<State, Dynamics, S> {
    /// Current state vector of the system.
    pub state: State,
    /// Governing ODE dynamics definition.
    pub dynamics: Dynamics,
    /// Numerical integration strategy.
    pub solver: S,
    /// Current elapsed simulation time `t`.
    t: f64,
    /// Epoch time `t0` for drift-free uniform-step time tracking.
    t0: f64,
    /// Active uniform step size since `t0`.
    step_dt: f64,
    /// Consecutive steps taken at `step_dt` since `t0`.
    step_count: u64,
}

impl<State, Dynamics, S> OdeModel<State, Dynamics, S> {
    /// Constructs a new [`OdeModel`] initialized at `t = 0.0`.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(state: State, dynamics: Dynamics, solver: S) -> Self {
        Self::with_time(state, dynamics, solver, 0.0)
    }

    /// Constructs a new [`OdeModel`] initialized at explicit starting time `t0`.
    #[must_use]
    #[verified_engine::verified]
    pub fn with_time(state: State, dynamics: Dynamics, solver: S, t0: f64) -> Self {
        Self {
            state,
            dynamics,
            solver,
            t: t0,
            t0,
            step_dt: 0.0,
            step_count: 0,
        }
    }

    /// Returns the current elapsed simulation time `t`.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.t
    }

    /// Sets the current elapsed simulation time `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        self.t = t;
        self.t0 = t;
        self.step_dt = 0.0;
        self.step_count = 0;
    }

    #[inline]
    #[verified_engine::verified]
    fn advance_clock(&mut self, dt: f64) {
        if self.step_count == 0 || dt.to_bits() == self.step_dt.to_bits() {
            self.step_dt = dt;
            self.step_count = self.step_count.saturating_add(1);
            self.t = self.t0 + (self.step_count as f64) * dt;
        } else {
            self.t0 = self.t;
            self.step_dt = dt;
            self.step_count = 1;
            self.t = self.t0 + dt;
        }
    }
}

impl<State, Dynamics, S> OdeSystem<State> for OdeModel<State, Dynamics, S>
where
    State: VectorOperations,
    Dynamics: OdeSystem<State>,
    S: Solver<State>,
{
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &State) -> State {
        self.dynamics.derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &State, out: &mut State) {
        self.dynamics.derivative_in_place(t, state, out);
    }
}

impl<State, Dynamics, S> TimeStepper<State> for OdeModel<State, Dynamics, S>
where
    State: VectorOperations,
    Dynamics: OdeSystem<State>,
    S: Solver<State>,
{
    #[verified_engine::verified]
    fn get_state(&self) -> &State {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut State {
        &mut self.state
    }

    #[verified_engine::verified]
    fn get_time(&self) -> f64 {
        self.t
    }

    #[verified_engine::verified]
    fn set_time(&mut self, t: f64) {
        self.t = t;
        self.t0 = t;
        self.step_dt = 0.0;
        self.step_count = 0;
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        self.solver
            .step(&self.dynamics, self.t, &mut self.state, dt);
        self.advance_clock(dt);
    }
}
