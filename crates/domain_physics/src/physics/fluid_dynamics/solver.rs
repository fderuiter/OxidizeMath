//! Solvers for Fluid Dynamics.
//!
//! This module adapts fluid dynamics equations (PDEs) into ODE systems that can be
//! solved using standard integrators like Euler or Runge-Kutta.

use super::conservation::MomentumEquation;
use super::types::{FlowState, FluidProperties, SpatialGradients};
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};

struct FluidParticleDynamics<'a, M: MomentumEquation> {
    properties: &'a FluidProperties,
    momentum_equation: &'a M,
    gradients: &'a SpatialGradients,
    body_force: Vector3<f64>,
}

impl<'a, M: MomentumEquation> OdeSystem<FlowState> for FluidParticleDynamics<'a, M> {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &FlowState) -> FlowState {
        let mut out = *state;
        self.derivative_in_place(_t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &FlowState, out: &mut FlowState) {
        let acceleration = self.momentum_equation.acceleration(
            self.properties,
            state,
            self.gradients,
            self.body_force,
        );
        *out = FlowState {
            velocity: acceleration,
            pressure: 0.0,
        };
    }
}

/// A Lagrangian particle in a flow field.
///
/// Tracks the state of a fluid element as it moves through space and time.
/// The `OdeSystem` implementation computes the acceleration (change in velocity)
/// based on the provided momentum equation and spatial gradients.
///
/// **Note:** This is a simplified model where spatial gradients are provided externally
/// or assumed constant during the time step. In a full simulation, gradients would be
/// re-evaluated at each step based on the particle's new position relative to neighbors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidParticleSystem<'a, M: MomentumEquation> {
    /// Fluid properties (density, viscosity).
    pub properties: &'a FluidProperties,
    /// The momentum equation strategy (Navier-Stokes, Euler).
    pub momentum_equation: M,
    /// Spatial gradients at the particle's location.
    pub gradients: SpatialGradients,
    /// External body force acceleration (e.g., gravity).
    pub body_force: Vector3<f64>,
    /// Current state.
    pub state: FlowState,
    solver: RungeKutta4<FlowState>,
    time: f64,
}

impl<'a, M: MomentumEquation> FluidParticleSystem<'a, M> {
    /// Creates a new fluid particle system with pre-allocated solver buffers.
    #[verified_engine::verified]
    pub fn new(
        properties: &'a FluidProperties,
        momentum_equation: M,
        gradients: SpatialGradients,
        body_force: Vector3<f64>,
        initial_state: FlowState,
    ) -> Self {
        let solver = RungeKutta4::new(&initial_state);
        Self {
            properties,
            momentum_equation,
            gradients,
            body_force,
            state: initial_state,
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

impl<'a, M: MomentumEquation> OdeSystem<FlowState> for FluidParticleSystem<'a, M> {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &FlowState) -> FlowState {
        let mut out = *state;
        self.derivative_in_place(_t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &FlowState, out: &mut FlowState) {
        let acceleration = self.momentum_equation.acceleration(
            self.properties,
            state,
            &self.gradients,
            self.body_force,
        );
        *out = FlowState {
            velocity: acceleration,
            pressure: 0.0,
        };
    }
}

impl<'a, M: MomentumEquation> TimeStepper<FlowState> for FluidParticleSystem<'a, M> {
    #[verified_engine::verified]
    fn get_state(&self) -> &FlowState {
        &self.state
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut FlowState {
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

    fn step_with_at<S: Solver<FlowState>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let dynamics = FluidParticleDynamics {
            properties: self.properties,
            momentum_equation: &self.momentum_equation,
            gradients: &self.gradients,
            body_force: self.body_force,
        };
        solver.step(&dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        let dynamics = FluidParticleDynamics {
            properties: self.properties,
            momentum_equation: &self.momentum_equation,
            gradients: &self.gradients,
            body_force: self.body_force,
        };
        self.solver.step(&dynamics, t, &mut self.state, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.time;
        self.step_at(t, dt);
        self.time += dt;
    }

    fn step_with<S: Solver<FlowState>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.time;
        self.step_with_at(solver, t, dt);
        self.time += dt;
    }
}
