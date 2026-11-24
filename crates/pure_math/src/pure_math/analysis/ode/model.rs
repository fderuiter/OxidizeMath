use super::stepper::TimeStepper;
use super::traits::{OdeSystem, Solver, VectorOperations};
use verified_engine::Theory;

/// Generic initial-value problem facade integrating $\dot{\mathbf{y}}(t) = \mathbf{f}(t, \mathbf{y}(t))$ (`@dormand1980family`, `spec:pure_math_v1`).
///
/// - **Numerical stability**: Local truncation error $O(\Delta t^{p+1})$ and linear stability domain are determined by the injected `Solver` `S` (e.g. $p = 4$ for RK4); accumulated time `t` advances via `f64` addition.
/// - **Domain of validity**: Requires locally Lipschitz right-hand side `Dynamics` on $[t_0, t_{\max}]$ and step sizes $\Delta t$ inside the stability region of `S`.
///
/// # Generics
/// * `State` - The state vector type (must implement `VectorOperations`).
/// * `Dynamics` - The system definition (must implement `OdeSystem<State>`).
/// * `S` - The numerical solver strategy (must implement `Solver<State>`).
#[derive(Debug, Clone, Theory)]
#[theory(
    description = "An Ordinary Differential Equation (ODE) model encapsulates a dynamical system's state, its governing differential equations, and a numerical solver to simulate its evolution over time.",
    citation = "dormand1980family"
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
    /// Constructs a new [`OdeModel`] for $\dot{\mathbf{y}}(t) = \mathbf{f}(t, \mathbf{y}(t))$ initialized at $t_0 = 0$ (`@dormand1980family`, `spec:pure_math_v1`).
    /// Numerical stability: exact `t = 0.0` initialization with zero rounding error. Domain of validity: `state` and `solver` workspaces must share matching vector dimensions.
    #[verified_engine::verified]
    pub fn new(state: State, dynamics: Dynamics, solver: S) -> Self {
        Self {
            state,
            dynamics,
            solver,
            t: 0.0,
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
        self.solver
            .step(&self.dynamics, self.t, &mut self.state, dt);
        self.t += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::ode::{ArrayState, RungeKutta4};

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
