// [cite:chaos_theory]
//! Continuous Chaos (The Lorenz System)

use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    Euler, Heun, IntegrationMethod, OdeSystem, RungeKutta4, Solver, TimeStepper,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

#[derive(Clone, Copy)]
struct LorenzSolverBuffers {
    euler: Euler<Vector3<f64>>,
    heun: Heun<Vector3<f64>>,
    rk4: RungeKutta4<Vector3<f64>>,
}

impl LorenzSolverBuffers {
    #[verified_engine::verified]
    fn new(example: &Vector3<f64>) -> Self {
        Self {
            euler: Euler::new(example),
            heun: Heun::new(example),
            rk4: RungeKutta4::new(example),
        }
    }
}

impl PartialEq for LorenzSolverBuffers {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl std::fmt::Debug for LorenzSolverBuffers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LorenzSolverBuffers")
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy)]
struct LorenzDynamics {
    sigma: f64,
    rho: f64,
    beta: f64,
}

impl OdeSystem<Vector3<f64>> for LorenzDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let mut out = Vector3::zeros();
        self.derivative_in_place(_t, state, &mut out);
        out
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, _t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        let x = state.x;
        let y = state.y;
        let z = state.z;
        *out = Vector3::new(
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        );
    }
}

/// Represents the state of the Lorenz system $(x, y, z)$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LorenzState {
    /// The 3D state vector.
    pub vec: Vector3<f64>,
}

impl LorenzState {
    #[allow(missing_docs)]
    #[verified_engine::verified]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        LorenzState {
            vec: Vector3::new(x, y, z),
        }
    }
}

/// A builder for the `LorenzSystem`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LorenzBuilder {
    sigma: f64,
    rho: f64,
    beta: f64,
    integration_method: IntegrationMethod,
}

impl Default for LorenzBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            integration_method: IntegrationMethod::RungeKutta4,
        }
    }
}

impl LorenzBuilder {
    /// Creates a new builder with standard chaotic constants.
    #[verified_engine::verified]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the Prandtl number $\sigma$.
    #[verified_engine::verified]
    pub fn sigma(mut self, val: f64) -> Self {
        self.sigma = val;
        self
    }

    /// Sets the Rayleigh number $\rho$.
    #[verified_engine::verified]
    pub fn rho(mut self, val: f64) -> Self {
        self.rho = val;
        self
    }

    /// Sets the geometric factor $\beta$.
    #[verified_engine::verified]
    pub fn beta(mut self, val: f64) -> Self {
        self.beta = val;
        self
    }

    /// Sets the numerical integration method used by [`LorenzSystem::step`].
    #[verified_engine::verified]
    pub fn integration_method(mut self, method: IntegrationMethod) -> Self {
        self.integration_method = method;
        self
    }

    /// Builds the `LorenzSystem` with pre-allocated solver buffers and initial state.
    #[verified_engine::verified]
    pub fn build(self, state: LorenzState) -> LorenzSystem {
        let solvers = LorenzSolverBuffers::new(&state.vec);
        LorenzSystem {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
            state,
            integration_method: self.integration_method,
            time: 0.0,
            solvers,
        }
    }
}

/// The Lorenz System simulator.
///
/// The Lorenz equations are:
/// * $\dot{x} = \sigma(y - x)$
/// * $\dot{y} = x(\rho - z) - y$
/// * $\dot{z} = xy - \beta z$
///
/// These equations originally modeled atmospheric convection but became the seminal example of deterministic chaos.
///
/// # Parameters
/// * $\sigma$ (Prandtl number): Ratio of momentum diffusivity to thermal diffusivity.
/// * $\rho$ (Rayleigh number): Temperature difference driving the convection.
/// * $\beta$ (Geometric factor): Related to the aspect ratio of the convection rolls.
///
/// # Example
///
/// ```
/// # use domain_physics::__doctest_facade as math_explorer;
/// use math_explorer::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
/// use math_explorer::pure_math::analysis::ode::TimeStepper;
///
/// // 1. Initialize the system state close to the attractor
/// let state = LorenzState::new(10.0, 10.0, 10.0);
///
/// // 2. Build the system with standard chaotic parameters
/// let mut lorenz = LorenzBuilder::new()
///     .sigma(10.0)
///     .rho(28.0)
///     .beta(8.0 / 3.0)
///     .build(state);
///
/// // 3. Step forward in time (dt = 0.01)
/// lorenz.step(0.01);
///
/// let new_state = lorenz.state.vec;
/// println!("New State: ({:.2}, {:.2}, {:.2})", new_state.x, new_state.y, new_state.z);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LorenzSystem {
    /// The Prandtl number $\sigma$, representing the ratio of momentum diffusivity to thermal diffusivity.
    pub sigma: f64,
    /// The Rayleigh number $\rho$, representing the temperature difference driving the convection.
    pub rho: f64,
    /// The geometric factor $\beta$, related to the aspect ratio of the convection rolls.
    pub beta: f64,
    /// The current state of the system.
    pub state: LorenzState,
    integration_method: IntegrationMethod,
    time: f64,
    solvers: LorenzSolverBuffers,
}

impl LorenzSystem {
    #[verified_engine::verified]
    fn dynamics(&self) -> LorenzDynamics {
        LorenzDynamics {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
        }
    }

    #[verified_engine::verified]
    fn step_configured_at(&mut self, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        match self.integration_method {
            IntegrationMethod::RungeKutta4 => {
                self.solvers.rk4.step(&dynamics, t, &mut self.state.vec, dt);
            }
            IntegrationMethod::Euler => {
                self.solvers
                    .euler
                    .step(&dynamics, t, &mut self.state.vec, dt);
            }
            IntegrationMethod::Heun => {
                self.solvers
                    .heun
                    .step(&dynamics, t, &mut self.state.vec, dt);
            }
        }
    }

    #[verified_engine::verified]
    fn step_configured(&mut self, dt: f64) {
        let t = self.time;
        self.step_configured_at(t, dt);
        self.time += dt;
    }

    /// Creates a new LorenzSystem with standard chaotic constants: $\sigma=10, \rho=28, \beta=8/3$.
    ///
    /// # Deprecation Notice
    /// Prefer using `LorenzBuilder::new().build(initial_state)` for better composability.
    #[verified_engine::verified]
    pub fn default_chaotic(initial_state: LorenzState) -> Self {
        LorenzBuilder::new().build(initial_state)
    }

    /// Returns the active [`IntegrationMethod`] used when stepping.
    #[verified_engine::verified]
    pub fn integration_method(&self) -> IntegrationMethod {
        self.integration_method
    }

    /// Updates the active [`IntegrationMethod`] without reallocating scratch buffers.
    #[verified_engine::verified]
    pub fn set_integration_method(&mut self, method: IntegrationMethod) {
        self.integration_method = method;
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

    /// Advances the system at explicit simulation time `t` by `dt` using the configured integrator,
    /// without mutating [`Self::time`].
    #[verified_engine::verified]
    pub fn step_at(&mut self, t: f64, dt: f64) {
        self.step_configured_at(t, dt);
    }

    /// Advances the system by time `dt` using the configured [`IntegrationMethod`] and pre-allocated buffers.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        self.step_configured(dt);
    }

    /// Advances the system by time `dt` using a provided solver strategy.
    ///
    /// This allows the user to switch integrators (e.g., Euler, RK4) dynamically.
    pub fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step_with(self, solver, dt);
    }
}

impl TimeStepper<Vector3<f64>> for LorenzSystem {
    #[verified_engine::verified]
    fn get_state(&self) -> &Vector3<f64> {
        &self.state.vec
    }

    #[verified_engine::verified]
    fn get_state_mut(&mut self) -> &mut Vector3<f64> {
        &mut self.state.vec
    }

    #[verified_engine::verified]
    fn current_time(&self) -> f64 {
        self.time
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.time += dt;
    }

    fn step_with_at<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        solver.step(&dynamics, t, &mut self.state.vec, dt);
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        self.step_configured_at(t, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        self.step_configured(dt);
    }

    fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.time;
        let dynamics = self.dynamics();
        solver.step(&dynamics, t, &mut self.state.vec, dt);
        self.time += dt;
    }
}

impl OdeSystem<Vector3<f64>> for LorenzSystem {
    /// Calculates the derivative at a given state.
    #[verified_engine::verified]
    fn derivative(&self, t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        self.dynamics().derivative(t, state)
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        self.dynamics().derivative_in_place(t, state, out);
    }
}

impl TheoryDescribable for LorenzSystem {
    fn theory_description(&self) -> String {
        let x = self.state.vec.x;
        let y = self.state.vec.y;
        let z = self.state.vec.z;
        let regime = if self.rho > 24.74 {
            "Chaotic regime"
        } else {
            "Stable regime"
        };
        format!(
            "Lorenz attractor in {}, state: x={:.2}, y={:.2}, z={:.2}",
            regime, x, y, z
        )
    }

    fn phonetic_description(&self) -> String {
        self.theory_description()
    }

    fn theory_citation(&self) -> String {
        "Lorenz, E. N. (1963). Deterministic Nonperiodic Flow. Journal of the Atmospheric Sciences, 20(2), 130-141.".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Lorenz attractor state".to_string());
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[verified_engine::verified]
    fn test_lorenz_integrator_parity_time_and_derives() {
        let init = LorenzState::new(1.0, 2.0, 3.0);
        let dt = 0.01;
        let methods = [
            IntegrationMethod::Euler,
            IntegrationMethod::Heun,
            IntegrationMethod::RungeKutta4,
        ];
        let mut finals = Vec::new();

        for method in methods {
            let builder = LorenzBuilder::new().integration_method(method);
            assert_eq!(builder, builder.clone());
            assert!(!format!("{builder:?}").is_empty());

            let mut inherent = builder.build(init);
            let mut via_trait = builder.build(init);
            assert_eq!(inherent, via_trait);
            assert_eq!(inherent.integration_method(), method);

            for _ in 0..5 {
                inherent.step(dt);
                <LorenzSystem as TimeStepper<Vector3<f64>>>::step(&mut via_trait, dt);
            }
            assert_eq!(inherent.state, via_trait.state);
            assert_eq!(inherent.time().to_bits(), via_trait.time().to_bits());
            assert_eq!(inherent, via_trait);

            let mut rebuilt = builder.build(inherent.state);
            rebuilt.set_time(inherent.time());
            assert_eq!(inherent, rebuilt);
            assert!(!format!("{inherent:?}").is_empty());
            finals.push(inherent.state.vec);
        }

        assert_ne!(finals[0], finals[1]);
        assert_ne!(finals[1], finals[2]);
    }
}
