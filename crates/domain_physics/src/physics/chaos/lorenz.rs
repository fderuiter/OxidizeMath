// [cite:chaos_theory]
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
    /// Constructs a 3D Lorenz phase-space state vector $(x, y, z)$.
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
}

impl Default for LorenzBuilder {
    #[verified_engine::verified]
    fn default() -> Self {
        Self {
            sigma: 10.0,
            rho: 28.0,
            beta: 8.0 / 3.0,
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

    /// Builds the `LorenzSystem` with the configured parameters and initial state.
    #[verified_engine::verified]
    pub fn build(self, state: LorenzState) -> LorenzSystem {
        LorenzSystem {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
            state,
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

#[verified_engine::verified]
fn lorenz_hopf_rho(sigma: f64, beta: f64) -> Option<f64> {
    let denom = sigma - beta - 1.0;
    if sigma > 0.0 && beta > 0.0 && denom > 1e-9 {
        let rho_h = sigma * (sigma + beta + 3.0) / denom;
        rho_h.is_finite().then_some(rho_h)
    } else {
        None
    }
}

#[verified_engine::verified]
fn classify_lorenz_regime(system: &LorenzSystem) -> (&'static str, f64, Option<f64>) {
    use pure_math::pure_math::analysis::ode::{RungeKutta4, SolverExt};

    let rho_h = lorenz_hopf_rho(system.sigma, system.beta);
    let dt = 0.01;
    let mut seed = Vector3::new(1.0, 1.0, 1.0);
    let mut rk4 = RungeKutta4::new(&seed);
    for _ in 0..400 {
        let next = rk4.solve(system, 0.0, &seed, dt);
        if !(next.x.is_finite() && next.y.is_finite() && next.z.is_finite()) || next.norm() > 1e6 {
            break;
        }
        seed = next;
    }
    let lambda =
        super::metrics::lorenz_lyapunov(system, &mut rk4, seed, dt, 40, 0.25).unwrap_or(f64::NAN);

    if system.rho <= 1.0 {
        return ("Origin sink regime", lambda, rho_h);
    }
    if lambda.is_finite() && lambda > 0.05 {
        return ("Chaotic regime", lambda, rho_h);
    }
    if rho_h.is_some_and(|rh| system.rho >= rh) {
        return ("Periodic window regime", lambda, rho_h);
    }
    ("Twin-lobe equilibrium regime", lambda, rho_h)
}

impl TheoryDescribable for LorenzSystem {
    fn theory_description(&self) -> String {
        let x = self.state.vec.x;
        let y = self.state.vec.y;
        let z = self.state.vec.z;
        let (regime, lambda, rho_h) = classify_lorenz_regime(self);
        match (lambda.is_finite(), rho_h) {
            (true, Some(rh)) => format!(
                "Lorenz attractor in {regime} (λ₁={lambda:.2}, ρ_H={rh:.2}), state: x={x:.2}, y={y:.2}, z={z:.2}"
            ),
            (true, None) => format!(
                "Lorenz attractor in {regime} (λ₁={lambda:.2}), state: x={x:.2}, y={y:.2}, z={z:.2}"
            ),
            (false, _) => {
                format!("Lorenz attractor in {regime}, state: x={x:.2}, y={y:.2}, z={z:.2}")
            }
        }
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
