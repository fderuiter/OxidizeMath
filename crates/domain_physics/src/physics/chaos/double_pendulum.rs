// [cite:chaos_theory]
//! Mechanical Chaos (The Planar Double Pendulum)

use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    ArrayState, OdeSystem, RungeKutta4, Solver, TimeStepper,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Represents the 4D state $[\theta_1, \theta_2, \omega_1, \omega_2]$ of a planar double pendulum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoublePendulumState {
    /// Stack-allocated state vector $[\theta_1, \theta_2, \omega_1, \omega_2]$.
    pub state: ArrayState<4>,
}

impl DoublePendulumState {
    /// Constructs a new state from joint angles $(\theta_1, \theta_2)$ and angular velocities $(\omega_1, \omega_2)$.
    #[verified_engine::verified]
    pub fn new(theta1: f64, theta2: f64, omega1: f64, omega2: f64) -> Self {
        Self {
            state: ArrayState([theta1, theta2, omega1, omega2]),
        }
    }

    /// Returns $(\theta_1, \theta_2, \omega_1, \omega_2)$.
    #[verified_engine::verified]
    pub fn components(&self) -> (f64, f64, f64, f64) {
        let [t1, t2, w1, w2] = self.state.0;
        (t1, t2, w1, w2)
    }

    /// Returns a 3D phase-space projection $(\theta_1, \theta_2, \omega_1)$ for trajectory plotting.
    #[verified_engine::verified]
    pub fn phase_point_3d(&self) -> Vector3<f64> {
        let [t1, t2, w1, _] = self.state.0;
        Vector3::new(t1, t2, w1)
    }
}

/// Copyable physical parameters for zero-allocation Lagrangian derivative evaluation.
#[derive(Debug, Clone, Copy)]
struct DoublePendulumDynamics {
    m1: f64,
    m2: f64,
    l1: f64,
    l2: f64,
    g: f64,
}

impl OdeSystem<ArrayState<4>> for DoublePendulumDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &ArrayState<4>) -> ArrayState<4> {
        let mut out = ArrayState([0.0; 4]);
        self.derivative_in_place(0.0, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        let [t1, t2, w1, w2] = state.0;
        let delta = t1 - t2;
        let sin_d = delta.sin();
        let cos_d = delta.cos();
        let denom_core = (2.0 * self.m1 + self.m2 - self.m2 * (2.0 * delta).cos()).max(1e-12);
        let d1 = (self.l1 * denom_core).max(1e-12);
        let d2 = (self.l2 * denom_core).max(1e-12);

        let num1 = -self.g * (2.0 * self.m1 + self.m2) * t1.sin()
            - self.m2 * self.g * (t1 - 2.0 * t2).sin()
            - 2.0 * sin_d * self.m2 * (w2 * w2 * self.l2 + w1 * w1 * self.l1 * cos_d);

        let m_sum = self.m1 + self.m2;
        let num2 = 2.0
            * sin_d
            * (w1 * w1 * self.l1 * m_sum
                + self.g * m_sum * t1.cos()
                + w2 * w2 * self.l2 * self.m2 * cos_d);

        out.0 = [w1, w2, num1 / d1, num2 / d2];
    }
}

/// Builder for configuring a [`DoublePendulum`] with pre-allocated RK4 buffers.
#[derive(Debug, Clone)]
pub struct DoublePendulumBuilder {
    m1: f64,
    m2: f64,
    l1: f64,
    l2: f64,
    g: f64,
}

impl Default for DoublePendulumBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            m1: 1.0,
            m2: 1.0,
            l1: 1.0,
            l2: 1.0,
            g: 9.81,
        }
    }
}

impl DoublePendulumBuilder {
    /// Creates a new builder with unit masses and rod lengths $(m_1=m_2=1, l_1=l_2=1, g=9.81)$.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets upper bob mass $m_1$.
    #[verified_engine::verified]
    pub fn m1(mut self, val: f64) -> Self {
        self.m1 = val;
        self
    }

    /// Sets lower bob mass $m_2$.
    #[verified_engine::verified]
    pub fn m2(mut self, val: f64) -> Self {
        self.m2 = val;
        self
    }

    /// Sets upper rod length $l_1$.
    #[verified_engine::verified]
    pub fn l1(mut self, val: f64) -> Self {
        self.l1 = val;
        self
    }

    /// Sets lower rod length $l_2$.
    #[verified_engine::verified]
    pub fn l2(mut self, val: f64) -> Self {
        self.l2 = val;
        self
    }

    /// Sets gravitational acceleration $g$.
    #[verified_engine::verified]
    pub fn g(mut self, val: f64) -> Self {
        self.g = val;
        self
    }

    /// Builds a [`DoublePendulum`] with pre-allocated `RungeKutta4<ArrayState<4>>` scratch buffers.
    #[verified_engine::verified]
    pub fn build(self, state: DoublePendulumState) -> DoublePendulum {
        let solver = RungeKutta4::new(&state.state);
        DoublePendulum {
            m1: self.m1,
            m2: self.m2,
            l1: self.l1,
            l2: self.l2,
            g: self.g,
            state,
            solver,
        }
    }
}

/// Lagrangian planar double-pendulum solver backed by pre-allocated `RungeKutta4` buffers.
///
/// Angles $\theta_1, \theta_2$ are measured from the downward vertical. For high-energy
/// initial releases (e.g. $\theta_1 = \theta_2 = 2.4\text{ rad}$), the system exhibits
/// Hamiltonian chaos while conserving total mechanical energy $E = T + V$.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::double_pendulum::{
///     DoublePendulumBuilder, DoublePendulumState,
/// };
///
/// let initial = DoublePendulumState::new(2.2, 2.2, 0.0, 0.0);
/// let mut pendulum = DoublePendulumBuilder::new().build(initial);
/// let e0 = pendulum.total_energy();
/// pendulum.step(0.005);
/// assert!((pendulum.total_energy() - e0).abs() < 1e-5);
/// ```
#[derive(Debug, Clone)]
pub struct DoublePendulum {
    /// Upper bob mass $m_1$.
    pub m1: f64,
    /// Lower bob mass $m_2$.
    pub m2: f64,
    /// Upper rod length $l_1$.
    pub l1: f64,
    /// Lower rod length $l_2$.
    pub l2: f64,
    /// Gravitational acceleration $g$.
    pub g: f64,
    /// Current 4D state $[\theta_1, \theta_2, \omega_1, \omega_2]$.
    pub state: DoublePendulumState,
    solver: RungeKutta4<ArrayState<4>>,
}

impl DoublePendulum {
    /// Creates a chaotic double pendulum with unit masses and lengths.
    #[verified_engine::verified]
    pub fn default_chaotic(initial_state: DoublePendulumState) -> Self {
        DoublePendulumBuilder::new().build(initial_state)
    }

    #[verified_engine::verified]
    fn dynamics(&self) -> DoublePendulumDynamics {
        DoublePendulumDynamics {
            m1: self.m1,
            m2: self.m2,
            l1: self.l1,
            l2: self.l2,
            g: self.g,
        }
    }

    /// Advances the pendulum by `dt` in-place using the pre-allocated RK4 buffer.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let dyn_params = self.dynamics();
        self.solver
            .step(&dyn_params, 0.0, &mut self.state.state, dt);
    }

    /// Advances the pendulum by `dt` using a caller-supplied ODE solver.
    #[verified_engine::verified]
    pub fn step_with<S: Solver<ArrayState<4>>>(&mut self, solver: &mut S, dt: f64) {
        let dyn_params = self.dynamics();
        solver.step(&dyn_params, 0.0, &mut self.state.state, dt);
    }

    /// Computes Cartesian $(x, y)$ positions of both bobs relative to the pivot $(0, 0)$.
    #[verified_engine::verified]
    pub fn bob_positions(&self) -> ([f64; 2], [f64; 2]) {
        let (t1, t2, _, _) = self.state.components();
        let x1 = self.l1 * t1.sin();
        let y1 = -self.l1 * t1.cos();
        let x2 = x1 + self.l2 * t2.sin();
        let y2 = y1 - self.l2 * t2.cos();
        ([x1, y1], [x2, y2])
    }

    /// Computes total mechanical energy $E = T + V$.
    #[verified_engine::verified]
    pub fn total_energy(&self) -> f64 {
        let (t1, t2, w1, w2) = self.state.components();
        let kinetic = 0.5 * (self.m1 + self.m2) * self.l1 * self.l1 * w1 * w1
            + 0.5 * self.m2 * self.l2 * self.l2 * w2 * w2
            + self.m2 * self.l1 * self.l2 * w1 * w2 * (t1 - t2).cos();
        let potential = -(self.m1 + self.m2) * self.g * self.l1 * t1.cos()
            - self.m2 * self.g * self.l2 * t2.cos();
        kinetic + potential
    }
}

impl OdeSystem<ArrayState<4>> for DoublePendulum {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<4>) -> ArrayState<4> {
        self.dynamics().derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        self.dynamics().derivative_in_place(t, state, out);
    }
}

impl TimeStepper<ArrayState<4>> for DoublePendulum {
    #[verified_engine::verified]
    fn get_state(&self) -> &ArrayState<4> {
        &self.state.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut ArrayState<4> {
        &mut self.state.state
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let dyn_params = self.dynamics();
        self.solver
            .step(&dyn_params, 0.0, &mut self.state.state, dt);
    }
}

impl TheoryDescribable for DoublePendulum {
    fn theory_description(&self) -> String {
        let (t1, t2, w1, w2) = self.state.components();
        format!(
            "Planar double pendulum: theta1={:.2}, theta2={:.2}, omega1={:.2}, omega2={:.2}",
            t1, t2, w1, w2
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992). Chaos in a Double Pendulum. American Journal of Physics, 60(6), 491-499."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Double pendulum state".to_string());
        map
    }
}
