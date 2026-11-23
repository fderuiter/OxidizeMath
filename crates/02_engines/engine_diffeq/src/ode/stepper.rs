//! Unified `TimeStepper` interface with defaulted simulation time accessors.
// [cite:engine_diffeq]

use super::explicit::RungeKutta4;
use super::traits::{OdeSystem, Solver, SolverExt, VectorOperations};

/// Trait for dynamical systems that advance step-by-step in time.
///
/// Autonomous domain models need only implement [`Self::get_state`] and [`Self::get_state_mut`];
/// non-autonomous containers such as [`super::model::OdeModel`] override [`Self::get_time`] and
/// [`Self::set_time`] so `t` advances accurately across steps.
pub trait TimeStepper<State: VectorOperations>: OdeSystem<State> {
    /// Returns a reference to the current state.
    #[verified_engine::verified]
    fn get_state(&self) -> &State;

    /// Returns a mutable reference to the current state.
    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut State;

    /// Returns the current simulation time `t` (defaults to `0.0` for autonomous models).
    #[verified_engine::verified]
    fn get_time(&self) -> f64 {
        0.0
    }

    /// Updates the current simulation time `t` (no-op by default for autonomous models).
    #[verified_engine::verified]
    fn set_time(&mut self, _t: f64) {}

    /// Advances the system by `dt` using a default `RungeKutta4` strategy and updates elapsed time.
    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.get_time();
        let mut solver = RungeKutta4::new(self.get_state());
        let new_state = solver.solve(self, t, self.get_state(), dt);
        *self.get_state_mut() = new_state;
        self.set_time(t + dt);
    }

    /// Advances the system by `dt` using a caller-supplied `solver` and updates elapsed time.
    #[verified_engine::verified]
    fn step_with<S: Solver<State>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.get_time();
        let current_state = self.get_state().clone();
        let new_state = solver.solve(self, t, &current_state, dt);
        *self.get_state_mut() = new_state;
        self.set_time(t + dt);
    }
}
