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
#[derive(Debug, Clone, Copy, PartialEq, Theory)]
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
    /// Elapsed simulation time $t$.
    time: f64,
}

impl<State, Dynamics, S> OdeModel<State, Dynamics, S> {
    /// Constructs a new OdeModel at initial simulation time `t = 0.0`.
    #[verified_engine::verified]
    pub fn new(state: State, dynamics: Dynamics, solver: S) -> Self {
        Self {
            state,
            dynamics,
            solver,
            time: 0.0,
        }
    }

    /// Returns the elapsed simulation time `t`.
    #[verified_engine::verified]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Sets the current simulation time `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, time: f64) {
        self.time = time;
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
    fn current_time(&self) -> f64 {
        self.time
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.time += dt;
    }

    fn step_with_at<OtherS: Solver<State>>(&mut self, solver: &mut OtherS, t: f64, dt: f64) {
        solver.step(&self.dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        self.solver.step(&self.dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.time;
        self.solver.step(&self.dynamics, t, &mut self.state, dt);
        self.time += dt;
    }

    fn step_with<OtherS: Solver<State>>(&mut self, solver: &mut OtherS, dt: f64) {
        let t = self.time;
        solver.step(&self.dynamics, t, &mut self.state, dt);
        self.time += dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pure_math::analysis::ode::solvers::{Euler, Heun, RungeKutta4};
    use crate::pure_math::analysis::ode::state::{ArrayState, VecState};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct LinearTimeForcing;

    impl OdeSystem<ArrayState<1>> for LinearTimeForcing {
        fn derivative(&self, t: f64, _state: &ArrayState<1>) -> ArrayState<1> {
            ArrayState([2.0 * t])
        }

        fn derivative_in_place(&self, t: f64, _state: &ArrayState<1>, out: &mut ArrayState<1>) {
            out.0[0] = 2.0 * t;
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct CustomVecStepper {
        state: VecState,
        t: f64,
    }

    impl OdeSystem<VecState> for CustomVecStepper {
        fn derivative(&self, t: f64, _state: &VecState) -> VecState {
            VecState(vec![2.0 * t])
        }

        fn derivative_in_place(&self, t: f64, _state: &VecState, out: &mut VecState) {
            out.0[0] = 2.0 * t;
        }
    }

    impl TimeStepper<VecState> for CustomVecStepper {
        fn get_state(&self) -> &VecState {
            &self.state
        }

        fn get_state_mut(&mut self) -> &mut VecState {
            &mut self.state
        }

        fn current_time(&self) -> f64 {
            self.t
        }

        fn advance_time(&mut self, dt: f64) {
            self.t += dt;
        }
    }

    #[test]
    fn test_non_autonomous_ode_model_and_time_stepper_advance_t() {
        let y0 = ArrayState([1.0]);
        let mut rk4_model = OdeModel::new(y0, LinearTimeForcing, RungeKutta4::new(&y0));
        let mut heun_model = OdeModel::new(y0, LinearTimeForcing, Heun::new(&y0));
        let mut euler_model = OdeModel::new(y0, LinearTimeForcing, Euler::new(&y0));
        let mut vec_stepper = CustomVecStepper {
            state: VecState(vec![1.0]),
            t: 0.0,
        };
        let mut vec_rk4 = RungeKutta4::new(&vec_stepper.state);

        for _ in 0..10 {
            rk4_model.step(0.1);
            heun_model.step(0.1);
            euler_model.step(0.1);
            vec_stepper.step_with(&mut vec_rk4, 0.1);
        }

        assert!((rk4_model.time() - 1.0).abs() < 1e-12);
        assert!((rk4_model.state.0[0] - 2.0).abs() < 1e-12);
        assert!((heun_model.state.0[0] - 2.0).abs() < 1e-12);
        assert!((euler_model.state.0[0] - 1.9).abs() < 1e-12);
        assert!((vec_stepper.current_time() - 1.0).abs() < 1e-12);
        assert!((vec_stepper.state.0[0] - 2.0).abs() < 1e-12);
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct HarmonicOscillator;

    impl OdeSystem<ArrayState<2>> for HarmonicOscillator {
        fn derivative(&self, _t: f64, s: &ArrayState<2>) -> ArrayState<2> {
            ArrayState([s.0[1], -s.0[0]])
        }

        fn derivative_in_place(&self, _t: f64, s: &ArrayState<2>, out: &mut ArrayState<2>) {
            out.0[0] = s.0[1];
            out.0[1] = -s.0[0];
        }
    }

    #[test]
    fn test_autonomous_solvers_bit_identical_across_direct_and_model_stepping() {
        let init = ArrayState([1.0_f64, 0.0_f64]);
        let dt = 0.025;

        let mut s_euler = init;
        let mut direct_euler = Euler::new(&init);
        let mut model_euler = OdeModel::new(init, HarmonicOscillator, Euler::new(&init));

        let mut s_heun = init;
        let mut direct_heun = Heun::new(&init);
        let mut model_heun = OdeModel::new(init, HarmonicOscillator, Heun::new(&init));

        let mut s_rk4 = init;
        let mut direct_rk4 = RungeKutta4::new(&init);
        let mut model_rk4 = OdeModel::new(init, HarmonicOscillator, RungeKutta4::new(&init));

        for step_idx in 0..8 {
            let t = f64::from(step_idx) * dt;
            direct_euler.step(&HarmonicOscillator, t, &mut s_euler, dt);
            model_euler.step(dt);
            direct_heun.step(&HarmonicOscillator, t, &mut s_heun, dt);
            model_heun.step(dt);
            direct_rk4.step(&HarmonicOscillator, t, &mut s_rk4, dt);
            model_rk4.step(dt);
        }

        assert_eq!(s_euler.0[0].to_bits(), model_euler.state.0[0].to_bits());
        assert_eq!(s_euler.0[1].to_bits(), model_euler.state.0[1].to_bits());
        assert_eq!(s_heun.0[0].to_bits(), model_heun.state.0[0].to_bits());
        assert_eq!(s_heun.0[1].to_bits(), model_heun.state.0[1].to_bits());
        assert_eq!(s_rk4.0[0].to_bits(), model_rk4.state.0[0].to_bits());
        assert_eq!(s_rk4.0[1].to_bits(), model_rk4.state.0[1].to_bits());
    }
}
