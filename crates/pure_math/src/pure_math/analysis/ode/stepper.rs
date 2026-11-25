use super::traits::{OdeSystem, Solver, VectorOperations};

/// Trait for initial-value ODE systems $\dot{y} = f(t, y)$ that advance state in place across time.
///
/// Abstracts single-step propagation (`step` defaulting to classical [`RungeKutta4`](super::solvers::RungeKutta4)
/// and `step_with` for caller-selected [`Solver`] strategies) without cloning heap-backed states.
/// Reference: Hairer, Nørsett, & Wanner (1993), *Solving Ordinary Differential Equations I*,
/// DOI: 10.1007/978-3-540-78862-1; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Stability
/// Default `step` inherits classical RK4 absolute stability ($|1 + z + z^2/2 + z^3/6 + z^4/24| \le 1$,
/// requiring $|h\lambda| \lesssim 2.785$ along the negative real axis).
///
/// # Domain of Validity
/// Requires locally Lipschitz right-hand side `f(t, y)` and finite step size `dt` within the
/// stability region of the active [`Solver`].
pub trait TimeStepper<State: VectorOperations>: OdeSystem<State> {
    /// Returns a reference to the current state.
    #[verified_engine::verified]
    fn get_state(&self) -> &State;

    /// Returns a mutable reference to the current state.
    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut State;

    /// Returns the current simulation clock $t$ passed to `OdeSystem::derivative(t, state)`.
    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        0.0
    }

    /// Advances the simulation clock by `dt` after a time step completes.
    #[verified_engine::verified]
    fn advance_time(&mut self, _dt: f64) {}

    /// Advances the system state in place at explicit time `t` by `dt` using `solver`.
    #[verified_engine::verified]
    fn step_with_at<S: Solver<State>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let mut state = self.get_state_mut().take_for_step();
        solver.step(self, t, &mut state, dt);
        *self.get_state_mut() = state;
    }

    /// Advances the system state in place at explicit time `t` by `dt` using [`RungeKutta4`](super::solvers::RungeKutta4).
    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        use super::solvers::RungeKutta4;
        let mut solver = RungeKutta4::new(self.get_state());
        self.step_with_at(&mut solver, t, dt);
    }

    /// Advances the system by `dt` at [`Self::current_time`] and records [`Self::advance_time`].
    ///
    /// The default implementation uses Runge-Kutta 4 (RK4). Steppers with pre-allocated
    /// solvers override this method to avoid per-step scratch initialization.
    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.current_time();
        self.step_at(t, dt);
        self.advance_time(dt);
    }

    /// Advances the system in place by `dt` using a caller-supplied `solver`, passing
    /// [`Self::current_time`] and advancing tracked simulation time by `dt`.
    #[verified_engine::verified]
    fn step_with<S: Solver<State>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.current_time();
        self.step_with_at(solver, t, dt);
        self.advance_time(dt);
    }
}
