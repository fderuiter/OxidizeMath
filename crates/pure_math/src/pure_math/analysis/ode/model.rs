use super::stepper::TimeStepper;
use super::traits::{OdeSystem, Solver, VectorOperations};
use verified_engine::Theory;

/// A generic container for an ODE model, combining State, Dynamics, and a Solver.
///
/// This struct implements the Facade pattern, providing a unified interface
/// to simulate a system defined by `Dynamics` using a `Solver`.
///
/// # Generics
/// * `State` - The state vector type (must implement `VectorOperations`).
/// * `Dynamics` - The system definition (must implement `OdeSystem<State>`).
/// * `S` - The numerical solver strategy (must implement `Solver<State>`).
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "An Ordinary Differential Equation (ODE) model encapsulates a dynamical system's state, its governing differential equations, and a numerical solver to simulate its evolution over time.",
    citation = "Numerical Recipes: The Art of Scientific Computing (Press et al., 2007)"
)]
pub struct OdeModel<State, Dynamics, S> {
    /// The current state of the system.
    pub state: State,
    /// The physics/dynamics definition.
    pub dynamics: Dynamics,
    /// The numerical solver.
    pub solver: S,
    elapsed_time: f64,
}

impl<State, Dynamics, S> OdeModel<State, Dynamics, S> {
    /// Constructs a new OdeModel.
    #[verified_engine::verified]
    pub fn new(state: State, dynamics: Dynamics, solver: S) -> Self {
        Self {
            state,
            dynamics,
            solver,
            elapsed_time: 0.0,
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
    fn step(&mut self, dt: f64) {
        let t = self.elapsed_time;
        self.solver.step(&self.dynamics, t, &mut self.state, dt);
        self.elapsed_time += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::super::solvers::RungeKutta4;
    use super::super::state::ArrayState;
    use super::*;

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
}
