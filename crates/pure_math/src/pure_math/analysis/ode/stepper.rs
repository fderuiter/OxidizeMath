use super::traits::{OdeSystem, Solver, VectorOperations};

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

    /// Returns the current simulation time `t`.
    ///
    /// Autonomous models without explicit time state return `0.0` by default.
    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        0.0
    }

    /// Advances the tracked simulation time by `dt` after a step.
    #[verified_engine::verified]
    fn advance_time(&mut self, _dt: f64) {}

    /// Advances the system at explicit simulation time `t` by `dt` in-place using a provided solver,
    /// without mutating tracked simulation time.
    fn step_with_at<S: Solver<State>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let mut state = self.get_state_mut().take_for_step();
        solver.step(self, t, &mut state, dt);
        *self.get_state_mut() = state;
    }

    /// Advances the system at explicit simulation time `t` by `dt` using the default strategy (RK4),
    /// without mutating tracked simulation time.
    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        use super::solvers::RungeKutta4;
        let mut solver = RungeKutta4::new(self.get_state());
        self.step_with_at(&mut solver, t, dt);
    }

    /// Advances the system by `dt` using a default strategy.
    ///
    /// The default implementation uses Runge-Kutta 4 (RK4) at [`Self::current_time`]
    /// and advances tracked simulation time via [`Self::advance_time`].
    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.current_time();
        self.step_at(t, dt);
        self.advance_time(dt);
    }

    /// Advances the system by `dt` in-place using a provided solver at [`Self::current_time`],
    /// advancing tracked simulation time via [`Self::advance_time`].
    fn step_with<S: Solver<State>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.current_time();
        self.step_with_at(solver, t, dt);
        self.advance_time(dt);
    }
}
