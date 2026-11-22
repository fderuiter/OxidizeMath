//! Solvers for Fluid Dynamics.
//!
//! This module adapts fluid dynamics equations (PDEs) into ODE systems that can be
//! solved using standard integrators like Euler or Runge-Kutta.

use super::conservation::{Euler, MomentumEquation};
use super::potential_flow::{FlowElement, PotentialFlowField};
use super::types::{FlowState, FluidProperties, SpatialGradients};
use nalgebra::{Matrix3, Vector2, Vector3};
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};

/// Telemetry evaluated at a 2D potential-flow dye probe coupled with [`FluidParticleSystem`] and [`Euler`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TracerProbeReadout {
    /// Probe coordinates $(x, y)$.
    pub position: Vector2<f64>,
    /// Clamped 2D velocity vector $(u, v)$.
    pub velocity: Vector2<f64>,
    /// Convective acceleration $(\mathbf{u} \cdot \nabla)\mathbf{u}$ evaluated via [`Euler`].
    pub convective_acceleration: Vector2<f64>,
    /// Static pressure $p = p_\infty + \frac{1}{2}\rho(U_\infty^2 - \|\mathbf{u}\|^2)$ from Bernoulli's equation.
    pub bernoulli_pressure: f64,
    /// Dimensionless pressure coefficient $C_p = 1 - \|\mathbf{u}\|^2 / U_\infty^2$.
    pub pressure_coefficient: f64,
    /// Next Lagrangian particle state integrated over `dt` by [`FluidParticleSystem`].
    pub predicted_state: FlowState,
}

/// Lagrangian 2D dye tracer system advected through a [`PotentialFlowField`] with [`RungeKutta4`].
///
/// # Example
///
/// ```rust
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::nalgebra::Vector2;
/// use math_explorer::physics::fluid_dynamics::potential_flow::{PotentialFlowField, UniformFlow};
/// use math_explorer::physics::fluid_dynamics::solver::PotentialTracerSystem;
/// use math_explorer::pure_math::analysis::ode::TimeStepper;
///
/// let mut field = PotentialFlowField::new();
/// field.add(Box::new(UniformFlow::new(5.0, 0.0)));
/// let mut tracer = PotentialTracerSystem::new(&field, Vector2::new(0.0, 1.0), 50.0);
/// tracer.step(0.2);
/// assert!((tracer.position.x - 1.0).abs() < 1e-9);
/// ```
pub struct PotentialTracerSystem<'a> {
    /// Superposed potential flow field.
    pub field: &'a PotentialFlowField,
    /// Current 2D tracer position $(x, y)$.
    pub position: Vector2<f64>,
    /// Maximum speed clamp near singularity cores.
    pub max_speed: f64,
}

impl<'a> PotentialTracerSystem<'a> {
    /// Creates a new 2D potential-flow tracer at `position` with core speed cap `max_speed`.
    #[verified_engine::verified]
    pub fn new(field: &'a PotentialFlowField, position: Vector2<f64>, max_speed: f64) -> Self {
        let clamped_cap = if max_speed.is_finite() && max_speed > 0.0 {
            max_speed
        } else {
            100.0
        };
        debug_assert!(clamped_cap > 0.0);
        Self {
            field,
            position,
            max_speed: clamped_cap,
        }
    }

    /// Evaluates velocity at `pos`, returning zero inside `TOLERANCE_HIGH` singularity cores and clamping to `max_speed`.
    #[verified_engine::verified]
    pub fn clamped_velocity(&self, pos: Vector2<f64>) -> Vector2<f64> {
        if !pos.x.is_finite()
            || !pos.y.is_finite()
            || self
                .field
                .is_near_singularity(pos.x, pos.y, math_commons::registry::TOLERANCE_HIGH)
        {
            return Vector2::zeros();
        }
        let raw = self.field.velocity(pos.x, pos.y);
        if !raw.x.is_finite() || !raw.y.is_finite() {
            return Vector2::zeros();
        }
        let speed = raw.norm();
        debug_assert!(self.max_speed > 0.0);
        if speed > self.max_speed {
            raw * (self.max_speed / speed)
        } else {
            raw
        }
    }

    /// Evaluates Bernoulli static pressure at `pos` relative to `(freestream_speed, reference_pressure)`.
    #[verified_engine::verified]
    pub fn bernoulli_pressure_at(
        &self,
        pos: Vector2<f64>,
        density: f64,
        freestream_speed: f64,
        reference_pressure: f64,
    ) -> f64 {
        let v = self.clamped_velocity(pos);
        debug_assert!(density > 0.0);
        reference_pressure
            + 0.5 * density * (freestream_speed * freestream_speed - v.norm_squared())
    }

    /// Evaluates central-difference velocity Jacobian and Bernoulli pressure gradient at `p`.
    #[verified_engine::verified]
    fn probe_gradients(
        &self,
        p: Vector2<f64>,
        rho: f64,
        u_inf: f64,
        p_inf: f64,
    ) -> (Matrix3<f64>, Vector3<f64>) {
        let h = 1e-4_f64;
        let dv_dx = (self.clamped_velocity(Vector2::new(p.x + h, p.y))
            - self.clamped_velocity(Vector2::new(p.x - h, p.y)))
            / (2.0 * h);
        let dv_dy = (self.clamped_velocity(Vector2::new(p.x, p.y + h))
            - self.clamped_velocity(Vector2::new(p.x, p.y - h)))
            / (2.0 * h);
        let dp_dx = (self.bernoulli_pressure_at(Vector2::new(p.x + h, p.y), rho, u_inf, p_inf)
            - self.bernoulli_pressure_at(Vector2::new(p.x - h, p.y), rho, u_inf, p_inf))
            / (2.0 * h);
        let dp_dy = (self.bernoulli_pressure_at(Vector2::new(p.x, p.y + h), rho, u_inf, p_inf)
            - self.bernoulli_pressure_at(Vector2::new(p.x, p.y - h), rho, u_inf, p_inf))
            / (2.0 * h);
        debug_assert!(h > 0.0);
        let vel_grad = Matrix3::new(dv_dx.x, dv_dy.x, 0.0, dv_dx.y, dv_dy.y, 0.0, 0.0, 0.0, 0.0);
        (vel_grad, Vector3::new(dp_dx, dp_dy, 0.0))
    }

    /// Couples this tracer with [`FluidParticleSystem`] and [`Euler`] to compute convective acceleration and Bernoulli pressure.
    #[verified_engine::verified]
    pub fn evaluate_probe(
        &self,
        properties: &FluidProperties,
        freestream_speed: f64,
        reference_pressure: f64,
        dt: f64,
    ) -> TracerProbeReadout {
        let p = self.position;
        let v = self.clamped_velocity(p);
        let rho = properties.density();
        let pressure = self.bernoulli_pressure_at(p, rho, freestream_speed, reference_pressure);
        let cp = if freestream_speed.abs() > math_commons::registry::TOLERANCE_HIGH {
            1.0 - v.norm_squared() / (freestream_speed * freestream_speed)
        } else {
            0.0
        };
        let (vel_grad, grad_p) = self.probe_gradients(p, rho, freestream_speed, reference_pressure);
        let state = FlowState::new(Vector3::new(v.x, v.y, 0.0), pressure);
        let conv_sys = FluidParticleSystem::new(
            properties,
            Euler,
            SpatialGradients::new(vel_grad, Vector3::zeros(), Vector3::zeros()),
            Vector3::zeros(),
            state,
        );
        let conv3 = -conv_sys.derivative(0.0, &state).velocity;
        let mut lagrangian_sys = FluidParticleSystem::new(
            properties,
            Euler,
            SpatialGradients::new(Matrix3::zeros(), grad_p, Vector3::zeros()),
            Vector3::zeros(),
            state,
        );
        lagrangian_sys.step(dt);
        debug_assert!(pressure.is_finite());
        TracerProbeReadout {
            position: p,
            velocity: v,
            convective_acceleration: Vector2::new(conv3.x, conv3.y),
            bernoulli_pressure: pressure,
            pressure_coefficient: cp,
            predicted_state: *lagrangian_sys.get_state(),
        }
    }
}

impl<'a> OdeSystem<Vector2<f64>> for PotentialTracerSystem<'a> {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector2<f64>) -> Vector2<f64> {
        self.clamped_velocity(*state)
    }
}

impl<'a> TimeStepper<Vector2<f64>> for PotentialTracerSystem<'a> {
    #[verified_engine::verified]
    fn get_state(&self) -> &Vector2<f64> {
        &self.position
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut Vector2<f64> {
        &mut self.position
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let mut solver = RungeKutta4::new(&self.position);
        let mut next = self.position;
        solver.step(self, 0.0, &mut next, dt);
        if next.x.is_finite() && next.y.is_finite() {
            self.position = next;
        }
        debug_assert!(self.position.x.is_finite() && self.position.y.is_finite());
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
}

impl<'a, M: MomentumEquation> FluidParticleSystem<'a, M> {
    /// Creates a new fluid particle system.
    #[verified_engine::verified]
    pub fn new(
        properties: &'a FluidProperties,
        momentum_equation: M,
        gradients: SpatialGradients,
        body_force: Vector3<f64>,
        initial_state: FlowState,
    ) -> Self {
        Self {
            properties,
            momentum_equation,
            gradients,
            body_force,
            state: initial_state,
        }
    }
}

impl<'a, M: MomentumEquation> OdeSystem<FlowState> for FluidParticleSystem<'a, M> {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &FlowState) -> FlowState {
        // Compute acceleration d(velocity)/dt
        let acceleration = self.momentum_equation.acceleration(
            self.properties,
            state,
            &self.gradients,
            self.body_force,
        );

        // Pressure evolution is not modeled here (requires Poisson solver).
        // We return 0 for pressure change, treating it as constant for the ODE step.
        FlowState {
            velocity: acceleration,
            pressure: 0.0,
        }
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
    fn step(&mut self, dt: f64) {
        use pure_math::pure_math::analysis::ode::RungeKutta4;
        use pure_math::pure_math::analysis::ode::SolverExt;
        let mut solver = RungeKutta4::new(self.get_state());
        let new_state = solver.solve(self, 0.0, self.get_state(), dt);
        *self.get_state_mut() = new_state;
    }
}
