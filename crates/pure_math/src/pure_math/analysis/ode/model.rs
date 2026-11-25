//! Generic state-dynamics-solver ODE facade (`OdeModel`).
//!
//! **Maturity Tier:** `[CERT] Certified` — verified zero-allocation time-stepping contract and SemVer signature lock.

use super::stepper::TimeStepper;
use super::traits::{OdeSystem, Solver, VectorOperations};
use verified_engine::Theory;

/// A generic container for an ODE model, combining State, Dynamics, and a Solver.
///
/// This struct implements the Facade pattern, providing a unified interface
/// to simulate a system defined by `Dynamics` using a `Solver`.
///
/// Initial-value problem wrapper for autonomous/non-autonomous ODE systems $\frac{d\mathbf{y}}{dt} = \mathbf{f}(t, \mathbf{y})$, $\mathbf{y}(t_0) = \mathbf{y}_0$.
///
/// Reference: Hairer, Norsett & Wanner (1993), *Solving Ordinary Differential Equations I*, DOI: 10.1007/978-3-540-78862-1; [`docs/math_specs/pure_math.md`](../../../../../../docs/math_specs/pure_math.md).
///
/// # Generics
/// * `State` - The state vector type (must implement `VectorOperations`).
/// * `Dynamics` - The system definition (must implement `OdeSystem<State>`).
/// * `S` - The numerical solver strategy (must implement `Solver<State>`).
///
/// # Stability
/// Numerical stability is governed by the paired [`Solver`] stability polynomial $R(\lambda \Delta t)$ along the Jacobian spectrum.
///
/// # Domain of Validity
/// Valid for locally Lipschitz right-hand sides $\mathbf{f}(t, \mathbf{y})$ over finite time horizons with finite step sizes $\Delta t$.
#[derive(Debug, Clone, PartialEq, Theory)]
#[theory(
    description = "An Ordinary Differential Equation (ODE) model encapsulates a dynamical system's state, its governing differential equations, and a numerical solver to simulate its evolution over time.",
    citation = "Numerical Recipes: The Art of Scientific Computing (Press et al., 2007)",
    maturity = "Certified"
)]
pub struct OdeModel<State, Dynamics, S> {
    /// The current state of the system.
    pub state: State,
    /// The physics/dynamics definition.
    pub dynamics: Dynamics,
    /// The numerical solver.
    pub solver: S,
    /// The current simulation time.
    t: f64,
}

impl<State, Dynamics, S> OdeModel<State, Dynamics, S> {
    /// Constructs a new OdeModel initialized at `t = 0.0`.
    #[verified_engine::verified]
    pub fn new(state: State, dynamics: Dynamics, solver: S) -> Self {
        Self {
            state,
            dynamics,
            solver,
            t: 0.0,
        }
    }

    /// Returns the elapsed simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.t
    }

    /// Sets the current simulation time $t$.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        self.t = t;
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
    fn elapsed_time(&self) -> f64 {
        self.t
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.t += dt;
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        self.solver.step(&self.dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step_with_at<Sol: Solver<State>>(&mut self, solver: &mut Sol, t: f64, dt: f64) {
        solver.step(&self.dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        self.solver
            .step(&self.dynamics, self.t, &mut self.state, dt);
        self.t += dt;
    }

    #[verified_engine::verified]
    fn step_with<Sol: Solver<State>>(&mut self, solver: &mut Sol, dt: f64) {
        solver.step(&self.dynamics, self.t, &mut self.state, dt);
        self.t += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::ode::{ArrayState, RungeKutta4};

    struct NonAutonomousSineDecay {
        k: f64,
    }

    impl OdeSystem<ArrayState<1>> for NonAutonomousSineDecay {
        #[verified_engine::verified]
        fn derivative(&self, t: f64, state: &ArrayState<1>) -> ArrayState<1> {
            let mut out = ArrayState::zeros();
            self.derivative_in_place(t, state, &mut out);
            out
        }

        #[verified_engine::verified]
        fn derivative_in_place(&self, t: f64, state: &ArrayState<1>, out: &mut ArrayState<1>) {
            out.0[0] = -self.k * state.0[0] * t.sin();
        }
    }

    #[derive(Clone)]
    struct TimeRampOde;

    impl OdeSystem<ArrayState<1>> for TimeRampOde {
        #[verified_engine::verified]
        fn derivative(&self, t: f64, _state: &ArrayState<1>) -> ArrayState<1> {
            ArrayState([2.0 * t])
        }

        #[verified_engine::verified]
        fn derivative_in_place(&self, t: f64, _state: &ArrayState<1>, out: &mut ArrayState<1>) {
            out.0[0] = 2.0 * t;
        }
    }

    #[test]
    #[verified_engine::verified]
    fn test_ode_model_advances_elapsed_time_for_non_autonomous_rk4() {
        let init = ArrayState([1.0]);
        let mut model = OdeModel::new(
            init,
            NonAutonomousSineDecay { k: 1.0 },
            RungeKutta4::new(&init),
        );
        let dt = 0.05;
        let steps = 200;

        let (max_err, _) = verified_engine::engine::VerifiedEngine::run_verified(|| {
            let mut worst = 0.0_f64;
            for step_idx in 0..=steps {
                let t = (step_idx as f64) * dt;
                let exact = (t.cos() - 1.0).exp();
                let err = (model.get_state().0[0] - exact).abs();
                if err > worst {
                    worst = err;
                }
                if step_idx < steps {
                    model.step(dt);
                }
            }
            worst
        });

        assert!(
            max_err < 1e-4,
            "Non-autonomous RK4 max error {max_err:.6e} exceeded 1e-4"
        );
    }

    #[test]
    #[verified_engine::verified]
    fn test_ode_model_advances_explicit_time_t() {
        let init = ArrayState([0.0]);
        let mut model = OdeModel::new(init, TimeRampOde, RungeKutta4::new(&init));
        for _ in 0..20 {
            model.step(0.1);
        }
        // Integral of 2*t from 0 to 2.0 is t^2 = 4.0 (would be 0.2 if t stayed 0.0)
        assert!((model.get_state().0[0] - 4.0).abs() < 1e-10);
    }
}
