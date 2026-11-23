//! Canonical Hamiltonian Double Pendulum (`ArrayState<4>`) and Stateful Trajectory Stepper.
//!
//! Models two point masses $(m_1, m_2)$ connected by massless rigid rods $(l_1, l_2)$
//! in a uniform gravitational field $g$ using canonical phase-space coordinates
//! $(\theta_1, \theta_2, p_1, p_2)$. Because the $2 \times 2$ mass matrix depends on
//! $\Delta\theta = \theta_1 - \theta_2$, the Hamiltonian is **non-separable** and showcases
//! implicit symplectic integrators such as [`GaussLegendre4`](pure_math::pure_math::analysis::ode::GaussLegendre4)
//! alongside pre-allocated [`DoublePendulumSystem`] twin-trajectory stepping.

// [cite:chaos_theory]

use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    ArrayState, HamiltonianSystem, OdeSystem, RungeKutta4, Solver, TimeStepper,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Conservative planar double pendulum with canonical state $[\theta_1, \theta_2, p_1, p_2]$.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::double_pendulum::DoublePendulum;
/// use math_explorer::pure_math::analysis::ode::{GaussLegendre4, HamiltonianSystem, Solver};
///
/// let pendulum = DoublePendulum::default();
/// let mut state = DoublePendulum::default_chaotic_state();
/// let e0 = pendulum.hamiltonian(&state);
/// let mut solver = GaussLegendre4::new(&state);
/// for i in 0..200 {
///     solver.step(&pendulum, i as f64 * 0.02, &mut state, 0.02);
/// }
/// assert!(pendulum.relative_energy_error(&state, e0) < 1e-5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoublePendulum {
    /// Upper bob mass $m_1$ (kg).
    pub m1: f64,
    /// Lower bob mass $m_2$ (kg).
    pub m2: f64,
    /// Upper rod length $l_1$ (m).
    pub l1: f64,
    /// Lower rod length $l_2$ (m).
    pub l2: f64,
    /// Gravitational acceleration $g$ ($\text{m/s}^2$).
    pub g: f64,
}

impl Default for DoublePendulum {
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

impl DoublePendulum {
    /// Constructs a canonical double pendulum with explicit physical parameters.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(m1: f64, m2: f64, l1: f64, l2: f64, g: f64) -> Self {
        Self { m1, m2, l1, l2, g }
    }

    /// High-energy chaotic benchmark state $[\theta_1, \theta_2, p_1, p_2] = [2.16, 2.16, 0, 0]$.
    #[must_use]
    #[verified_engine::verified]
    pub fn default_chaotic_state() -> ArrayState<4> {
        ArrayState([2.16, 2.16, 0.0, 0.0])
    }

    /// Computes canonical angular velocities $(\dot{\theta}_1, \dot{\theta}_2) = \nabla_p H(q, p)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn angular_velocities(&self, state: &ArrayState<4>) -> (f64, f64) {
        let th1 = state.0[0];
        let th2 = state.0[1];
        let p1 = state.0[2];
        let p2 = state.0[3];
        let delta = th1 - th2;
        let (sd, cd) = delta.sin_cos();
        let eff_mass = self.m1 + self.m2 * sd * sd;
        let dth1 = (self.l2 * p1 - self.l1 * p2 * cd) / (self.l1 * self.l1 * self.l2 * eff_mass);
        let dth2 = ((self.m1 + self.m2) * self.l1 * p2 - self.m2 * self.l2 * p1 * cd)
            / (self.m2 * self.l1 * self.l2 * self.l2 * eff_mass);
        (dth1, dth2)
    }

    /// Computes Cartesian bob positions $((x_1, y_1), (x_2, y_2))$ relative to the pivot $(0, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn cartesian_positions(&self, state: &ArrayState<4>) -> ([f64; 2], [f64; 2]) {
        let (s1, c1) = state.0[0].sin_cos();
        let (s2, c2) = state.0[1].sin_cos();
        let x1 = self.l1 * s1;
        let y1 = -self.l1 * c1;
        let x2 = x1 + self.l2 * s2;
        let y2 = y1 - self.l2 * c2;
        ([x1, y1], [x2, y2])
    }

    /// Evaluates non-negative gravitational potential $V(\theta_1, \theta_2)$ referenced to $V(0, 0) = 0$.
    #[must_use]
    #[verified_engine::verified]
    pub fn potential_energy(&self, theta1: f64, theta2: f64) -> f64 {
        (self.m1 + self.m2) * self.g * self.l1 * (1.0 - theta1.cos())
            + self.m2 * self.g * self.l2 * (1.0 - theta2.cos())
    }

    /// Reconstructs a canonical state $[0, \theta_2, p_1, p_2]$ on the $\theta_1 = 0, \dot{\theta}_1 \ge 0$
    /// Poincaré surface of section for a prescribed total energy $E$.
    #[must_use]
    #[verified_engine::verified]
    pub fn seed_on_poincare_section(
        &self,
        energy: f64,
        theta2: f64,
        p2: f64,
    ) -> Option<ArrayState<4>> {
        if !energy.is_finite() || !theta2.is_finite() || !p2.is_finite() {
            return None;
        }
        let t_req = energy - self.potential_energy(0.0, theta2);
        if t_req < 0.0 {
            return None;
        }
        let (s2, c2) = theta2.sin_cos();
        let eff_mass = self.m1 + self.m2 * s2 * s2;
        let a = self.m2 * self.l2 * self.l2;
        let b = -2.0 * self.m2 * self.l1 * self.l2 * c2 * p2;
        let c = (self.m1 + self.m2) * self.l1 * self.l1 * p2 * p2
            - 2.0 * self.m2 * self.l1 * self.l1 * self.l2 * self.l2 * eff_mass * t_req;
        let disc = b * b - 4.0 * a * c;
        if disc < 0.0 || a <= 0.0 {
            return None;
        }
        let p1 = (-b + disc.sqrt()) / (2.0 * a);
        Some(ArrayState([0.0, theta2, p1, p2]))
    }
}

impl OdeSystem<ArrayState<4>> for DoublePendulum {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<4>) -> ArrayState<4> {
        let mut out = ArrayState::zeros();
        self.derivative_in_place(t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        let th1 = state.0[0];
        let th2 = state.0[1];
        let (dth1, dth2) = self.angular_velocities(state);
        let coupling = self.m2 * self.l1 * self.l2 * dth1 * dth2 * (th1 - th2).sin();
        out.0[0] = dth1;
        out.0[1] = dth2;
        out.0[2] = -coupling - (self.m1 + self.m2) * self.g * self.l1 * th1.sin();
        out.0[3] = coupling - self.m2 * self.g * self.l2 * th2.sin();
    }
}

impl HamiltonianSystem<ArrayState<4>> for DoublePendulum {
    #[verified_engine::verified]
    fn hamiltonian(&self, state: &ArrayState<4>) -> f64 {
        let th1 = state.0[0];
        let th2 = state.0[1];
        let p1 = state.0[2];
        let p2 = state.0[3];
        let (sd, cd) = (th1 - th2).sin_cos();
        let denom =
            2.0 * self.m2 * self.l1 * self.l1 * self.l2 * self.l2 * (self.m1 + self.m2 * sd * sd);
        let numer = self.m2 * self.l2 * self.l2 * p1 * p1
            + (self.m1 + self.m2) * self.l1 * self.l1 * p2 * p2
            - 2.0 * self.m2 * self.l1 * self.l2 * p1 * p2 * cd;
        numer / denom + self.potential_energy(th1, th2)
    }
}

impl TheoryDescribable for DoublePendulum {
    fn theory_description(&self) -> String {
        format!(
            "Canonical Hamiltonian double pendulum (m1={:.2}, m2={:.2}, l1={:.2}, l2={:.2}, g={:.2})",
            self.m1, self.m2, self.l1, self.l2, self.g
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Shinbrot, T., Grebogi, C., Wisdom, J., & Yorke, J. A. (1992). Chaos in a double pendulum. American Journal of Physics, 60(6), 491-499.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "default".to_string(),
            "Non-separable canonical Hamiltonian double pendulum".to_string(),
        );
        map
    }
}

/// Represents the 4D canonical phase-space state $[\theta_1, \theta_2, p_1, p_2]$ of a planar double pendulum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DoublePendulumState {
    /// Stack-allocated canonical state vector $[\theta_1, \theta_2, p_1, p_2]$.
    pub state: ArrayState<4>,
}

impl DoublePendulumState {
    /// Constructs a new state from joint angles $(\theta_1, \theta_2)$ and canonical momenta $(p_1, p_2)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn new(theta1: f64, theta2: f64, p1: f64, p2: f64) -> Self {
        Self {
            state: ArrayState([theta1, theta2, p1, p2]),
        }
    }

    /// Returns $(\theta_1, \theta_2, p_1, p_2)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn components(&self) -> (f64, f64, f64, f64) {
        let [t1, t2, p1, p2] = self.state.0;
        (t1, t2, p1, p2)
    }

    /// Returns 3D phase-space coordinates $(\theta_1, \theta_2, p_1)$ for trajectory plotting.
    #[must_use]
    #[verified_engine::verified]
    pub fn phase_point_3d(&self) -> Vector3<f64> {
        let [t1, t2, p1, _] = self.state.0;
        Vector3::new(t1, t2, p1)
    }
}

/// Builder for configuring a [`DoublePendulumSystem`] with pre-allocated RK4 buffers.
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
        let p = DoublePendulum::default();
        Self {
            m1: p.m1,
            m2: p.m2,
            l1: p.l1,
            l2: p.l2,
            g: p.g,
        }
    }
}

impl DoublePendulumBuilder {
    /// Creates a new builder with unit masses and rod lengths $(m_1=m_2=1, l_1=l_2=1, g=9.81)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets upper bob mass $m_1$.
    #[must_use]
    #[verified_engine::verified]
    pub fn m1(mut self, val: f64) -> Self {
        self.m1 = val;
        self
    }

    /// Sets lower bob mass $m_2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn m2(mut self, val: f64) -> Self {
        self.m2 = val;
        self
    }

    /// Sets upper rod length $l_1$.
    #[must_use]
    #[verified_engine::verified]
    pub fn l1(mut self, val: f64) -> Self {
        self.l1 = val;
        self
    }

    /// Sets lower rod length $l_2$.
    #[must_use]
    #[verified_engine::verified]
    pub fn l2(mut self, val: f64) -> Self {
        self.l2 = val;
        self
    }

    /// Sets gravitational acceleration $g$.
    #[must_use]
    #[verified_engine::verified]
    pub fn g(mut self, val: f64) -> Self {
        self.g = val;
        self
    }

    /// Builds a [`DoublePendulumSystem`] with pre-allocated `RungeKutta4<ArrayState<4>>` scratch buffers.
    #[must_use]
    #[verified_engine::verified]
    pub fn build(self, state: DoublePendulumState) -> DoublePendulumSystem {
        let solver = RungeKutta4::new(&state.state);
        DoublePendulumSystem {
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

/// Stateful planar double-pendulum trajectory stepper delegating canonical dynamics to [`DoublePendulum`].
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
pub struct DoublePendulumSystem {
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
    /// Current 4D canonical state $[\theta_1, \theta_2, p_1, p_2]$.
    pub state: DoublePendulumState,
    solver: RungeKutta4<ArrayState<4>>,
}

impl DoublePendulumSystem {
    /// Creates a chaotic double-pendulum stepper with unit masses and lengths.
    #[must_use]
    #[verified_engine::verified]
    pub fn default_chaotic(initial_state: DoublePendulumState) -> Self {
        DoublePendulumBuilder::new().build(initial_state)
    }

    /// Returns the underlying stateless canonical [`DoublePendulum`] Hamiltonian model.
    #[must_use]
    #[verified_engine::verified]
    pub fn model(&self) -> DoublePendulum {
        DoublePendulum::new(self.m1, self.m2, self.l1, self.l2, self.g)
    }

    /// Advances the pendulum by `dt` in-place using the pre-allocated RK4 buffer.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let model = self.model();
        self.solver.step(&model, 0.0, &mut self.state.state, dt);
    }

    /// Advances the pendulum by `dt` using a caller-supplied ODE solver.
    #[verified_engine::verified]
    pub fn step_with<S: Solver<ArrayState<4>>>(&mut self, solver: &mut S, dt: f64) {
        let model = self.model();
        solver.step(&model, 0.0, &mut self.state.state, dt);
    }

    /// Computes Cartesian $(x, y)$ positions of both bobs relative to the pivot $(0, 0)$.
    #[must_use]
    #[verified_engine::verified]
    pub fn bob_positions(&self) -> ([f64; 2], [f64; 2]) {
        self.model().cartesian_positions(&self.state.state)
    }

    /// Evaluates the canonical Hamiltonian energy $H(\theta_1, \theta_2, p_1, p_2)$ of the current state.
    #[must_use]
    #[verified_engine::verified]
    pub fn total_energy(&self) -> f64 {
        self.model().hamiltonian(&self.state.state)
    }
}

impl OdeSystem<ArrayState<4>> for DoublePendulumSystem {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &ArrayState<4>) -> ArrayState<4> {
        self.model().derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &ArrayState<4>, out: &mut ArrayState<4>) {
        self.model().derivative_in_place(t, state, out);
    }
}

impl HamiltonianSystem<ArrayState<4>> for DoublePendulumSystem {
    #[verified_engine::verified]
    fn hamiltonian(&self, state: &ArrayState<4>) -> f64 {
        self.model().hamiltonian(state)
    }
}

impl TimeStepper<ArrayState<4>> for DoublePendulumSystem {
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
        let model = self.model();
        self.solver.step(&model, 0.0, &mut self.state.state, dt);
    }
}

impl TheoryDescribable for DoublePendulumSystem {
    fn theory_description(&self) -> String {
        let (t1, t2, p1, p2) = self.state.components();
        format!(
            "Planar double pendulum: theta1={:.2}, theta2={:.2}, p1={:.2}, p2={:.2}",
            t1, t2, p1, p2
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        self.model().theory_citation()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        self.model().available_descriptions()
    }
}

#[cfg(test)]
mod tests;
