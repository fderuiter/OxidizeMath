use super::traits::{OdeSystem, Solver, SolverExt, VectorOperations};

/// A trait for systems that can advance in time.
///
/// This trait abstracts over the `step` and `step_with` methods,
/// enforcing a consistent interface for time-stepping simulations.
pub trait TimeStepper<State: VectorOperations>: OdeSystem<State> {
    /// Returns a reference to the current state.
    #[verified_engine::verified]
    fn get_state(&self) -> &State;

    /// Returns a mutable reference to the current state.
    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut State;

    /// Returns the current simulation time `t` passed to `OdeSystem::derivative`.
    ///
    /// Defaults to `0.0` so autonomous systems remain source-compatible without boilerplate.
    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        0.0
    }

    /// Advances the internal simulation clock by `dt` after a completed step.
    ///
    /// Defaults to a no-op for autonomous systems that do not store explicit time.
    #[verified_engine::verified]
    fn advance_time(&mut self, _dt: f64) {}

    /// Advances the system by `dt` using a default strategy.
    ///
    /// The default implementation uses Runge-Kutta 4 (RK4).
    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        use super::solvers::RungeKutta4;
        let t = self.current_time();
        let mut solver = RungeKutta4::new(self.get_state());
        let new_state = solver.solve(self, t, self.get_state(), dt);
        *self.get_state_mut() = new_state;
        self.advance_time(dt);
    }

    /// Advances the system by `dt` using a provided solver.
    #[verified_engine::verified]
    fn step_with<S: Solver<State>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.current_time();
        let current_state = self.get_state().clone();
        let new_state = solver.solve(self, t, &current_state, dt);
        *self.get_state_mut() = new_state;
        self.advance_time(dt);
    }
}
