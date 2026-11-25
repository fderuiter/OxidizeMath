// [cite:chaos_theory]
//! Continuous Chaos (The Lorenz System)

use crate::error::ChaosError;
use nalgebra::Vector3;
use pure_math::pure_math::analysis::ode::{
    IntegrationMethod, OdeSystem, PreallocatedOdeSolver, Solver, TimeStepper,
};
use scientific_metadata::theory::TheoryDescribable;
use std::collections::HashMap;

/// Represents the state of the Lorenz system $(x, y, z)$.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
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

#[derive(Debug, Clone, Copy)]
struct LorenzDynamics {
    sigma: f64,
    rho: f64,
    beta: f64,
}

impl OdeSystem<Vector3<f64>> for LorenzDynamics {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector3<f64>) -> Vector3<f64> {
        let x = state.x;
        let y = state.y;
        let z = state.z;
        Vector3::new(
            self.sigma * (y - x),
            x * (self.rho - z) - y,
            x * y - self.beta * z,
        )
    }

    #[verified_engine::verified]
    fn derivative_in_place(&self, t: f64, state: &Vector3<f64>, out: &mut Vector3<f64>) {
        *out = self.derivative(t, state);
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
    /// Creates a new builder with standard chaotic constants and [`IntegrationMethod::RungeKutta4`].
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

    /// Selects the numerical [`IntegrationMethod`] used by `LorenzSystem::step`.
    #[verified_engine::verified]
    pub fn integration_method(mut self, method: IntegrationMethod) -> Self {
        self.integration_method = method;
        self
    }

    /// Builds the `LorenzSystem` with pre-allocated solver scratch buffers and `t = 0.0`.
    #[verified_engine::verified]
    pub fn build(self, state: LorenzState) -> LorenzSystem {
        let solver = PreallocatedOdeSolver::new(&state.vec);
        LorenzSystem {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
            state,
            integration_method: self.integration_method,
            t: 0.0,
            solver,
        }
    }

    /// Validates that parameters and initial state are finite before building a [`LorenzSystem`].
    #[verified_engine::verified]
    pub fn try_build(self, state: LorenzState) -> Result<LorenzSystem, ChaosError> {
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            return Err(ChaosError::InvalidParameter(format!(
                "sigma must be finite and positive, got {}",
                self.sigma
            )));
        }
        if !self.rho.is_finite() || self.rho < 0.0 {
            return Err(ChaosError::InvalidParameter(format!(
                "rho must be finite and non-negative, got {}",
                self.rho
            )));
        }
        if !self.beta.is_finite() || self.beta <= 0.0 {
            return Err(ChaosError::InvalidParameter(format!(
                "beta must be finite and positive, got {}",
                self.beta
            )));
        }
        if !(state.vec.x.is_finite() && state.vec.y.is_finite() && state.vec.z.is_finite()) {
            return Err(ChaosError::InvalidParameter(
                "initial LorenzState coordinates must be finite".to_string(),
            ));
        }
        Ok(self.build(state))
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
    /// Selected numerical integration scheme honored by `step`.
    pub integration_method: IntegrationMethod,
    t: f64,
    solver: PreallocatedOdeSolver<Vector3<f64>>,
}

impl PartialEq for LorenzSystem {
    #[verified_engine::verified]
    fn eq(&self, other: &Self) -> bool {
        self.sigma == other.sigma
            && self.rho == other.rho
            && self.beta == other.beta
            && self.state == other.state
            && self.integration_method == other.integration_method
            && self.t == other.t
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
        self.t
    }

    #[verified_engine::verified]
    fn advance_time(&mut self, dt: f64) {
        self.t += dt;
    }

    #[verified_engine::verified]
    fn step_at(&mut self, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        self.solver.step_method(
            self.integration_method,
            &dynamics,
            t,
            &mut self.state.vec,
            dt,
        );
    }

    #[verified_engine::verified]
    fn step_with_at<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, t: f64, dt: f64) {
        let dynamics = self.dynamics();
        solver.step(&dynamics, t, &mut self.state.vec, dt);
    }

    #[verified_engine::verified]
    fn step(&mut self, dt: f64) {
        let t = self.t;
        self.step_at(t, dt);
        self.t += dt;
    }

    #[verified_engine::verified]
    fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        let t = self.t;
        self.step_with_at(solver, t, dt);
        self.t += dt;
    }
}

impl LorenzSystem {
    #[inline]
    #[verified_engine::verified]
    fn dynamics(&self) -> LorenzDynamics {
        LorenzDynamics {
            sigma: self.sigma,
            rho: self.rho,
            beta: self.beta,
        }
    }

    /// Creates a new LorenzSystem with standard chaotic constants: $\sigma=10, \rho=28, \beta=8/3$.
    ///
    /// # Deprecation Notice
    /// Prefer using `LorenzBuilder::new().build(initial_state)` for better composability.
    #[verified_engine::verified]
    pub fn default_chaotic(initial_state: LorenzState) -> Self {
        LorenzBuilder::new().build(initial_state)
    }

    /// Returns the elapsed simulation time $t$.
    #[must_use]
    #[verified_engine::verified]
    pub const fn time(&self) -> f64 {
        self.t
    }

    /// Sets the simulation clock to `t`.
    #[verified_engine::verified]
    pub fn set_time(&mut self, t: f64) {
        self.t = t;
    }

    /// Updates the active [`IntegrationMethod`] without reallocating scratch buffers.
    #[verified_engine::verified]
    pub fn set_integration_method(&mut self, method: IntegrationMethod) {
        self.integration_method = method;
    }

    /// Advances the system by time `dt` using the configured [`IntegrationMethod`] and pre-allocated buffers.
    #[verified_engine::verified]
    pub fn step(&mut self, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step(self, dt);
    }

    /// Advances the system by time `dt` using a provided solver strategy.
    ///
    /// This allows the user to switch integrators (e.g., Euler, RK4) dynamically.
    #[verified_engine::verified]
    pub fn step_with<S: Solver<Vector3<f64>>>(&mut self, solver: &mut S, dt: f64) {
        <Self as TimeStepper<Vector3<f64>>>::step_with(self, solver, dt);
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
    use pure_math::pure_math::analysis::ode::RungeKutta4;

    let rho_h = lorenz_hopf_rho(system.sigma, system.beta);
    let dt = 0.01;
    let mut seed = Vector3::new(1.0, 1.0, 1.0);
    let mut rk4 = RungeKutta4::new(&seed);
    for _ in 0..400 {
        let mut next = seed;
        rk4.step(system, 0.0, &mut next, dt);
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
    if system.sigma <= 0.0 || system.beta <= 0.0 {
        return ("Twin-lobe equilibrium regime", lambda, rho_h);
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
