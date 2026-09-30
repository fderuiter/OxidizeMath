//! Continuous Chaos (The Lorenz System)

use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{OdeSystem, Solver, TimeStepper};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Represents the state of the Lorenz system $(x, y, z)$.
#[derive(Debug, Clone, Copy)]
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
#[derive(Debug, Clone)]
pub struct LorenzBuilder {
    sigma: f64,
    rho: f64,
    beta: f64,
    dt: f64,
    integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod,
}

impl Default for LorenzBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
            dt: 0.01,
            integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod::RungeKutta4,
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

    /// Sets the time step `dt`.
    #[verified_engine::verified]
    pub fn dt(mut self, val: f64) -> Self {
        self.dt = val;
        self
    }

    /// Sets the numerical integration method.
    #[verified_engine::verified]
    pub fn integration_method(
        mut self,
        val: pure_math::pure_math::analysis::ode::IntegrationMethod,
    ) -> Self {
        self.integration_method = val;
        self
    }

    /// Builds the `LorenzSystem` with the configured parameters and initial state.
    #[verified_engine::verified]
    pub fn build(self, state: LorenzState) -> LorenzSystem {
        LorenzSystem {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
            state,
            dt: self.dt,
            integration_method: self.integration_method,
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
/// use domain_physics::physics::chaos::lorenz::{LorenzBuilder, LorenzState};
/// use pure_math::pure_math::analysis::ode::TimeStepper;
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
#[derive(Debug, Clone)]
pub struct LorenzSystem {
    /// The Prandtl number $\sigma$, representing the ratio of momentum diffusivity to thermal diffusivity.
    pub sigma: f64,
    /// The Rayleigh number $\rho$, representing the temperature difference driving the convection.
    pub rho: f64,
    /// The geometric factor $\beta$, related to the aspect ratio of the convection rolls.
    pub beta: f64,
    /// The current state of the system.
    pub state: LorenzState,
    /// The time step `dt`.
    pub dt: f64,
    /// The integration method to use.
    pub integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod,
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
}

impl LorenzSystem {
    /// Creates a new LorenzSystem with standard chaotic constants: $\sigma=10, \rho=28, \beta=8/3$.
    ///
    /// # Deprecation Notice
    /// Prefer using `LorenzBuilder::new().build(initial_state)` for better composability.
    #[verified_engine::verified]
    pub fn default_chaotic(initial_state: LorenzState) -> Self {
        LorenzBuilder::new().build(initial_state)
    }

    /// Advances the system by time `dt` using the Runge-Kutta 4 (RK4) method.
    ///
    /// This now delegates to the generic `RungeKutta4` solver via `TimeStepper`.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step(self, dt);
    }

    /// Advances the system by time `dt` using a provided solver strategy.
    ///
    /// This allows the user to switch integrators (e.g., Euler, RK4) dynamically.
    pub fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step_with(self, solver, dt);
    }
}

impl OdeSystem<Vector3<f64>> for LorenzSystem {
    /// Calculates the derivative at a given state.
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let x = state.x;
        let y = state.y;
        let z = state.z;

        let dx = self.sigma * (y - x);
        let dy = x * (self.rho - z) - y;
        let dz = x * y - self.beta * z;

        Vector3::new(dx, dy, dz)
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
        "[cite:chaos_theory]".to_string()
    }

    fn available_descriptions(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert("default".to_string(), "Lorenz attractor state".to_string());
        map
    }
}

use oxidize_core::{ModelConfig, ModelState, SimulationModel};
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[allow(missing_docs)]
pub struct LorenzConfig {
    #[allow(missing_docs)]
    pub sigma: f64,
    #[allow(missing_docs)]
    pub rho: f64,
    #[allow(missing_docs)]
    pub beta: f64,
    #[allow(missing_docs)]
    pub dt: f64,
    #[serde(default)]
    #[allow(missing_docs)]
    pub integration_method: pure_math::pure_math::analysis::ode::IntegrationMethod,
}

impl ModelConfig for LorenzConfig {}
impl ModelState for LorenzState {}

impl SimulationModel for LorenzSystem {
    type Config = LorenzConfig;
    type State = LorenzState;
    type Error = std::io::Error;

    #[verified_engine::verified]
    fn initialize<R: rand::RngCore>(
        config: Self::Config,
        _provider: R,
    ) -> Result<Self, Self::Error> {
        // We initialize to an arbitrary point, maybe near the attractor.
        let state = LorenzState::new(10.0, 10.0, 10.0);
        Ok(LorenzBuilder::new()
            .sigma(config.sigma)
            .rho(config.rho)
            .beta(config.beta)
            .dt(config.dt)
            .integration_method(config.integration_method)
            .build(state))
    }

    #[verified_engine::verified(opt_out = "inherent method call false positive")]
    fn step(&mut self) -> Result<(), Self::Error> {
        let dt = self.dt;
        match self.integration_method {
            pure_math::pure_math::analysis::ode::IntegrationMethod::Euler => {
                let mut solver = pure_math::pure_math::analysis::ode::Euler::new(&self.state.vec);
                <Self as pure_math::pure_math::analysis::ode::TimeStepper<Vector3<f64>>>::step_with(
                    self,
                    &mut solver,
                    dt,
                );
            }
            pure_math::pure_math::analysis::ode::IntegrationMethod::RungeKutta4 => {
                <Self as pure_math::pure_math::analysis::ode::TimeStepper<Vector3<f64>>>::step(
                    self, dt,
                );
            }
        }
        Ok(())
    }

    #[verified_engine::verified]
    fn get_state(&self) -> Self::State {
        self.state
    }
}
