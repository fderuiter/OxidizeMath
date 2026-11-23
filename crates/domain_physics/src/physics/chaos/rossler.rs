// [cite:chaos_theory]
//! Continuous Chaos (The Rössler System)

use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Represents the 3D phase-space state $(x, y, z)$ of the Rössler system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RosslerState {
    /// Current 3D state vector $(x, y, z)$.
    pub vec: Vector3<f64>,
}

impl RosslerState {
    /// Creates a new `RosslerState` from Cartesian coordinates.
    #[verified_engine::verified]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self {
            vec: Vector3::new(x, y, z),
        }
    }
}

/// Copyable parameter kernel for zero-allocation Rössler derivative evaluation.
#[derive(Debug, Clone, Copy)]
struct RosslerDynamics {
    a: f64,
    b: f64,
    c: f64,
}

impl OdeSystem<Vector3<f64>> for RosslerDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let mut out = Vector3::zeros();
        self.derivative_in_place(0.0, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        let x = state.x;
        let y = state.y;
        let z = state.z;
        out.x = -y - z;
        out.y = x + self.a * y;
        out.z = self.b + z * (x - self.c);
    }
}

/// Builder for configuring a [`RosslerSystem`] with pre-allocated solver buffers.
#[derive(Debug, Clone)]
pub struct RosslerBuilder {
    a: f64,
    b: f64,
    c: f64,
}

impl Default for RosslerBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            a: 0.2,
            b: 0.2,
            c: 5.7,
        }
    }
}

impl RosslerBuilder {
    /// Creates a new builder with Rössler's classic chaotic parameters $(a=0.2, b=0.2, c=5.7)$.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets spiral-expansion parameter $a$.
    #[verified_engine::verified]
    pub fn a(mut self, val: f64) -> Self {
        self.a = val;
        self
    }

    /// Sets vertical injection parameter $b$.
    #[verified_engine::verified]
    pub fn b(mut self, val: f64) -> Self {
        self.b = val;
        self
    }

    /// Sets nonlinear folding threshold $c$.
    #[verified_engine::verified]
    pub fn c(mut self, val: f64) -> Self {
        self.c = val;
        self
    }

    /// Builds a [`RosslerSystem`] with pre-allocated `RungeKutta4` scratch buffers.
    #[verified_engine::verified]
    pub fn build(self, state: RosslerState) -> RosslerSystem {
        let solver = RungeKutta4::new(&state.vec);
        RosslerSystem {
            a: self.a,
            b: self.b,
            c: self.c,
            state,
            solver,
        }
    }
}

/// Continuous-time Rössler chaotic attractor solver.
///
/// Evolves the coupled ordinary differential equations:
/// * $\dot{x} = -y - z$
/// * $\dot{y} = x + a y$
/// * $\dot{z} = b + z(x - c)$
///
/// Maintains a pre-allocated [`RungeKutta4`] solver buffer so stepping performs
/// zero dynamic heap allocations under `VerifiedAllocator`.
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::rossler::{RosslerBuilder, RosslerState};
///
/// let mut system = RosslerBuilder::new().build(RosslerState::new(0.1, 0.0, 0.0));
/// system.step(0.01);
/// assert!(system.state.vec.norm().is_finite());
/// ```
#[derive(Debug, Clone)]
pub struct RosslerSystem {
    /// Spiral-expansion parameter $a$.
    pub a: f64,
    /// Vertical injection parameter $b$.
    pub b: f64,
    /// Nonlinear folding threshold $c$.
    pub c: f64,
    /// Current phase-space state $(x, y, z)$.
    pub state: RosslerState,
    solver: RungeKutta4<Vector3<f64>>,
}

impl RosslerSystem {
    /// Creates a Rössler system using standard chaotic constants $(a=0.2, b=0.2, c=5.7)$.
    #[verified_engine::verified]
    pub fn default_chaotic(initial_state: RosslerState) -> Self {
        RosslerBuilder::new().build(initial_state)
    }

    #[verified_engine::verified]
    fn dynamics(&self) -> RosslerDynamics {
        RosslerDynamics {
            a: self.a,
            b: self.b,
            c: self.c,
        }
    }

    /// Advances the system by `dt` in-place using the pre-allocated RK4 buffer.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        let dyn_params = self.dynamics();
        self.solver.step(&dyn_params, 0.0, &mut self.state.vec, dt);
    }

    /// Advances the system by `dt` using a caller-supplied ODE solver.
    #[verified_engine::verified]
    pub fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        let dyn_params = self.dynamics();
        solver.step(&dyn_params, 0.0, &mut self.state.vec, dt);
    }
}

impl OdeSystem<Vector3<f64>> for RosslerSystem {
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        self.dynamics().derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        self.dynamics().derivative_in_place(t, state, out);
    }
}

impl TimeStepper<Vector3<f64>> for RosslerSystem {
    #[verified_engine::verified]
    fn get_state(&self) -> &Vector3<f64> {
        &self.state.vec
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.state.vec
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let dyn_params = self.dynamics();
        self.solver.step(&dyn_params, 0.0, &mut self.state.vec, dt);
    }
}

impl TheoryDescribable for RosslerSystem {
    fn theory_description(&self) -> String {
        let v = self.state.vec;
        format!(
            "Rossler attractor (a={:.2}, b={:.2}, c={:.2}), state: x={:.2}, y={:.2}, z={:.2}",
            self.a, self.b, self.c, v.x, v.y, v.z
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Rossler, O. E. (1976). An Equation for Continuous Chaos. Physics Letters A, 57(5), 397-398."
            .to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Rossler attractor state".to_string());
        map
    }
}
