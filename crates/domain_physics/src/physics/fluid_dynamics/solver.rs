//! Solvers for Fluid Dynamics.
//!
//! This module adapts fluid dynamics equations (PDEs) into ODE systems that can be
//! solved using standard integrators like Euler or Runge-Kutta.

use super::analysis::bernoulli_constant;
use super::conservation::{Euler as InviscidEuler, MomentumEquation};
use super::potential_flow::{FlowElement, PotentialFlowField};
use super::types::{FlowState, FluidProperties, SpatialGradients};
use nalgebra::{Matrix3, Vector2, Vector3};
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver, TimeStepper};

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
}

/// Instantaneous Lagrangian probe readout coupling [`PotentialFlowField`] with [`FluidParticleSystem`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TracerProbeReadout {
    /// Probe coordinates `(x, y)`.
    pub position: Vector2<f64>,
    /// Local Eulerian velocity `(u, v)`.
    pub velocity: Vector2<f64>,
    /// Material acceleration $D\mathbf{u}/Dt = -\nabla p / \rho$ evaluated via [`InviscidEuler`].
    pub convective_acceleration: Vector2<f64>,
    /// Static pressure $p$ from steady Bernoulli balance relative to freestream.
    pub bernoulli_pressure: f64,
    /// Dimensionless pressure coefficient $C_p = 1 - (|\mathbf{u}| / U_\infty)^2$.
    pub pressure_coefficient: f64,
    /// Predicted next [`FlowState`] after advancing [`FluidParticleSystem`] one `RungeKutta4` step.
    pub predicted_state: FlowState,
}

/// Lagrangian dye tracer advected through a 2D [`PotentialFlowField`] via fourth-order `RungeKutta4`,
/// coupling irrotational kinematics with steady Bernoulli pressure and inviscid Euler acceleration:
/// $$\frac{d\mathbf{x}}{dt} = \mathbf{u}(\mathbf{x}) = \nabla\phi(\mathbf{x}) = \left(\frac{\partial\psi}{\partial y}, -\frac{\partial\psi}{\partial x}\right), \qquad p + \frac{1}{2}\rho|\mathbf{u}|^2 = H_0.$$
///
/// Reference: Batchelor (2000), *An Introduction to Fluid Dynamics*, Cambridge Univ. Press,
/// DOI: 10.1017/CBO9780511800955;
/// [`docs/math_specs/domain_physics.md`](../../../../../docs/math_specs/domain_physics.md).
///
/// # Stability
/// Halts advection inside singularity exclusion cores (`is_near_singularity`) and clamps speed to
/// `max_speed >= 1e-3`, preventing `RungeKutta4` overshoot near $O(1/r)$ vortex/source poles and
/// $O(1/r^2)$ doublet cores while retaining $O(\Delta t^4)$ accuracy on smooth streamlines.
///
/// # Domain of Validity
/// Models steady, inviscid, incompressible, irrotational 2D potential flow away from point
/// singularities, requiring finite tracer coordinates, finite `dt`, and positive fluid density
/// $\rho > 0$ when evaluating Bernoulli/Euler probe readouts.
#[derive(Debug, Clone, Copy)]
pub struct PotentialTracerSystem<'a> {
    /// Underlying potential flow superposition field.
    pub field: &'a PotentialFlowField,
    /// Current 2D tracer position.
    pub position: Vector2<f64>,
    /// Maximum speed clamp guarding singularity cores (`TOLERANCE_HIGH`).
    pub max_speed: f64,
}

impl<'a> PotentialTracerSystem<'a> {
    /// Creates a new Lagrangian tracer at `position` with singularity speed clamp `max_speed`.
    #[verified_engine::verified]
    pub fn new(field: &'a PotentialFlowField, position: Vector2<f64>, max_speed: f64) -> Self {
        Self {
            field,
            position,
            max_speed: max_speed.max(1e-3),
        }
    }

    /// Evaluates velocity at `pos`, clamping magnitude near singularity cores.
    #[verified_engine::verified]
    pub fn clamped_velocity(&self, pos: Vector2<f64>) -> Vector2<f64> {
        if !pos.x.is_finite()
            || !pos.y.is_finite()
            || self.field.is_near_singularity(
                pos.x,
                pos.y,
                math_commons::registry::TOLERANCE_HIGH * 16.0,
            )
        {
            return Vector2::zeros();
        }
        let vel = self.field.velocity(pos.x, pos.y);
        let speed = vel.norm();
        if !speed.is_finite() {
            Vector2::zeros()
        } else if speed > self.max_speed {
            vel * (self.max_speed / speed)
        } else {
            vel
        }
    }

    /// Couples local potential-flow kinematics with [`FluidParticleSystem`] and [`InviscidEuler`].
    #[verified_engine::verified]
    pub fn evaluate_probe(
        &self,
        properties: &FluidProperties,
        freestream_speed: f64,
        freestream_pressure: f64,
        dt: f64,
    ) -> TracerProbeReadout {
        let p = self.position;
        let vel = self.clamped_velocity(p);
        let h = 1e-4;
        let vx_p = self.clamped_velocity(Vector2::new(p.x + h, p.y));
        let vx_m = self.clamped_velocity(Vector2::new(p.x - h, p.y));
        let vy_p = self.clamped_velocity(Vector2::new(p.x, p.y + h));
        let vy_m = self.clamped_velocity(Vector2::new(p.x, p.y - h));
        let du_dx = (vx_p - vx_m) / (2.0 * h);
        let du_dy = (vy_p - vy_m) / (2.0 * h);

        let rho = properties.density();
        let ref_state = FlowState::new(
            Vector3::new(freestream_speed, 0.0, 0.0),
            freestream_pressure,
        );
        let total_head = bernoulli_constant(&ref_state, properties, 0.0, 0.0);
        let static_p = total_head - 0.5 * rho * vel.norm_squared();
        let cp = if freestream_speed.abs() > 1e-6 {
            1.0 - vel.norm_squared() / (freestream_speed * freestream_speed)
        } else {
            0.0
        };

        // Steady inviscid Euler balance: grad(p) = -rho * (u . grad) u
        let conv = Vector2::new(
            vel.x * du_dx.x + vel.y * du_dy.x,
            vel.x * du_dx.y + vel.y * du_dy.y,
        );
        let vel_grad = Matrix3::new(du_dx.x, du_dy.x, 0.0, du_dx.y, du_dy.y, 0.0, 0.0, 0.0, 0.0);
        let p_grad = Vector3::new(-rho * conv.x, -rho * conv.y, 0.0);
        let gradients = SpatialGradients::new(vel_grad, p_grad, Vector3::zeros());
        let state = FlowState::new(Vector3::new(vel.x, vel.y, 0.0), static_p);
        let mut particle = FluidParticleSystem::new(
            properties,
            InviscidEuler,
            gradients,
            Vector3::zeros(),
            state,
        );
        let accel3 = particle.derivative(0.0, &state).velocity;
        if dt.is_finite() && dt.abs() > 0.0 {
            particle.step(dt);
        }

        TracerProbeReadout {
            position: p,
            velocity: vel,
            convective_acceleration: Vector2::new(accel3.x, accel3.y),
            bernoulli_pressure: static_p,
            pressure_coefficient: cp,
            predicted_state: particle.state,
        }
    }
}

impl OdeSystem<Vector2<f64>> for PotentialTracerSystem<'_> {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector2<f64>) -> Vector2<f64> {
        self.clamped_velocity(*state)
    }
}

impl TimeStepper<Vector2<f64>> for PotentialTracerSystem<'_> {
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
        if !dt.is_finite() || dt.abs() < math_commons::registry::TOLERANCE_HIGH {
            return;
        }
        let mut solver = RungeKutta4::new(&self.position);
        let mut next_pos = self.position;
        solver.step(self, 0.0, &mut next_pos, dt);
        self.position = next_pos;
    }
}
