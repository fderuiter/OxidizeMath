use super::traits::{OdeSystem, Solver, SolverExt, VectorOperations};

/// Trait for autonomous initial-value ODE systems $\dot{y} = f(t, y)$ that advance state in time.
///
/// Abstracts single-step propagation (`step` defaulting to classical [`RungeKutta4`](super::solvers::RungeKutta4)
/// and `step_with` for caller-selected [`Solver`] strategies).
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

    /// Advances the system by `dt` using a default strategy.
    ///
    /// The default implementation uses Runge-Kutta 4 (RK4).
    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        use super::solvers::RungeKutta4;
        use super::traits::SolverExt;
        let mut solver = RungeKutta4::new(self.get_state());
        let new_state = solver.solve(self, 0.0, self.get_state(), dt);
        *self.get_state_mut() = new_state;
    }

    /// Advances the system by `dt` using a provided solver.
    fn step_with<S: Solver<State>>(&mut self, solver: &mut S, dt: f64) {
        let current_state = self.get_state().clone();
        let new_state = solver.solve(self, 0.0, &current_state, dt);
        *self.get_state_mut() = new_state;
    }
}
