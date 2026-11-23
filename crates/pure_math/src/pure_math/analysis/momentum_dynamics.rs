// [cite:advanced_linear_algebra]
//! Zero-Allocation 2D Symplectic Hamiltonian and Langevin Momentum Solvers.
//!
//! Implements stack-allocated phase-space integrators unified under [`EvolutionEngine`]:
//! - [`PhasePoint2D`]: Canonical 2D coordinates $q = (w, b)$ and conjugate momenta $p = (p_w, p_b)$.
//! - [`HamiltonianLeapfrog2D`]: Conformal symplectic Störmer-Verlet (kick-drift-kick) integrator.
//! - [`LangevinBaoab2D`]: Second-order BAOAB Langevin thermostat splitting with exact
//!   Ornstein-Uhlenbeck fluctuation-dissipation sampling via `rand_distr::Normal`.
//!
//! # Example
//!
//! ```rust
//! # use pure_math::__doctest_facade as math_explorer;
//! use math_explorer::nalgebra::Vector2;
//! use math_explorer::pure_math::analysis::evolution::EvolutionEngine;
//! use math_explorer::pure_math::analysis::momentum_dynamics::{
//!     HamiltonianLeapfrog2D, PhasePoint2D,
//! };
//! use math_explorer::rng::OxidizeRng;
//!
//! let mut state = PhasePoint2D::new(Vector2::new(1.0, 0.0), Vector2::new(0.0, 1.0));
//! let mut solver = HamiltonianLeapfrog2D::new(1.0, 0.0);
//! let mut rng = OxidizeRng::new(42);
//! let mut grad_fn = |q: Vector2<f64>| q;
//! solver.step(&mut state, &mut grad_fn, &mut rng, 0.01).unwrap();
//! assert!((state.q.norm_squared() + state.p.norm_squared() - 2.0).abs() < 1e-4);
//! ```

use super::evolution::{EvolutionEngine, EvolutionError};
use nalgebra::Vector2;
use rand::RngCore;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// Minimum safe particle mass preventing division-by-zero in kinetic velocity derivatives.
pub const MIN_MASS: f64 = 0.05;
/// Maximum clamped particle mass for interactive stability.
pub const MAX_MASS: f64 = 50.0;
/// Minimum viscous damping coefficient $\gamma \ge 0$.
pub const MIN_DAMPING: f64 = 0.0;
/// Maximum viscous damping coefficient $\gamma$.
pub const MAX_DAMPING: f64 = 20.0;
/// Minimum thermal energy $k_B T \ge 0$.
pub const MIN_TEMPERATURE: f64 = 0.0;
/// Maximum thermal energy $k_B T$.
pub const MAX_TEMPERATURE: f64 = 50.0;
/// Minimum numerical integration step size $\Delta t$.
pub const MIN_DT: f64 = 1e-4;
/// Maximum numerical integration step size $\Delta t$.
pub const MAX_DT: f64 = 0.1;

/// Clamps particle mass $m \in [0.05, 50.0]$.
#[verified_engine::verified]
pub fn clamp_mass(mass: f64) -> f64 {
    if mass.is_finite() {
        mass.clamp(MIN_MASS, MAX_MASS)
    } else {
        1.0
    }
}

/// Clamps viscous damping $\gamma \in [0.0, 20.0]$.
#[verified_engine::verified]
pub fn clamp_damping(damping: f64) -> f64 {
    if damping.is_finite() {
        damping.clamp(MIN_DAMPING, MAX_DAMPING)
    } else {
        0.0
    }
}

/// Clamps thermal energy $k_B T \in [0.0, 50.0]$.
#[verified_engine::verified]
pub fn clamp_temperature(temperature: f64) -> f64 {
    if temperature.is_finite() {
        temperature.clamp(MIN_TEMPERATURE, MAX_TEMPERATURE)
    } else {
        0.0
    }
}

/// Clamps integration time step $\Delta t \in [10^{-4}, 0.1]$.
#[verified_engine::verified]
pub fn clamp_dt(dt: f64) -> f64 {
    if dt.is_finite() {
        dt.clamp(MIN_DT, MAX_DT)
    } else {
        0.01
    }
}

/// Canonical 2D phase-space state $(q, p) \in \mathbb{R}^2 \times \mathbb{R}^2$.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhasePoint2D {
    /// Generalized position coordinates $q = (w, b)$.
    pub q: Vector2<f64>,
    /// Conjugate momentum coordinates $p = (p_w, p_b)$.
    pub p: Vector2<f64>,
}

impl Default for PhasePoint2D {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::at_rest(Vector2::zeros())
    }
}

impl PhasePoint2D {
    /// Constructs a phase-space point from position `q` and conjugate momentum `p`.
    #[verified_engine::verified]
    pub fn new(q: Vector2<f64>, p: Vector2<f64>) -> Self {
        Self { q, p }
    }

    /// Constructs a stationary phase-space point at position `q` with $p = (0, 0)$.
    #[verified_engine::verified]
    pub fn at_rest(q: Vector2<f64>) -> Self {
        Self {
            q,
            p: Vector2::zeros(),
        }
    }

    /// Constructs a phase-space point from position `q`, initial velocity `v`, and mass `mass`.
    #[verified_engine::verified]
    pub fn from_velocity(q: Vector2<f64>, v: Vector2<f64>, mass: f64) -> Self {
        let m = clamp_mass(mass);
        Self { q, p: v * m }
    }

    /// Returns `true` when all position and momentum components are finite.
    #[verified_engine::verified]
    pub fn is_finite(&self) -> bool {
        self.q.x.is_finite() && self.q.y.is_finite() && self.p.x.is_finite() && self.p.y.is_finite()
    }

    /// Evaluates quadratic kinetic energy $K(p) = \frac{\|p\|^2}{2m}$ with clamped mass.
    #[verified_engine::verified]
    pub fn kinetic_energy(&self, mass: f64) -> f64 {
        let m = clamp_mass(mass);
        0.5 * self.p.norm_squared() / m
    }

    /// Reflects `q` and `p` inelastically against `[q_min, q_max]` with coefficient `restitution` $\in [0, 1]$.
    #[verified_engine::verified]
    pub fn reflect_bounds(&mut self, q_min: Vector2<f64>, q_max: Vector2<f64>, restitution: f64) {
        let r = if restitution.is_finite() {
            restitution.clamp(0.0, 1.0)
        } else {
            0.5
        };
        if self.q.x < q_min.x {
            self.q.x = q_min.x;
            if self.p.x < 0.0 {
                self.p.x = -self.p.x * r;
            }
        } else if self.q.x > q_max.x {
            self.q.x = q_max.x;
            if self.p.x > 0.0 {
                self.p.x = -self.p.x * r;
            }
        }
        if self.q.y < q_min.y {
            self.q.y = q_min.y;
            if self.p.y < 0.0 {
                self.p.y = -self.p.y * r;
            }
        } else if self.q.y > q_max.y {
            self.q.y = q_max.y;
            if self.p.y > 0.0 {
                self.p.y = -self.p.y * r;
            }
        }
    }
}

/// Zero-allocation 2D conformal symplectic Leapfrog (Störmer-Verlet) momentum integrator.
///
/// When `damping == 0.0`, reduces to exact volume-preserving Velocity-Verlet.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HamiltonianLeapfrog2D {
    /// Inertial mass $m \in [0.05, 50.0]$.
    pub mass: f64,
    /// Optional conformal viscous damping $\gamma \in [0.0, 20.0]$.
    pub damping: f64,
}

impl Default for HamiltonianLeapfrog2D {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(1.0, 0.0)
    }
}

impl HamiltonianLeapfrog2D {
    /// Creates a new 2D symplectic Leapfrog solver with clamped mass and damping.
    #[verified_engine::verified]
    pub fn new(mass: f64, damping: f64) -> Self {
        Self {
            mass: clamp_mass(mass),
            damping: clamp_damping(damping),
        }
    }

    /// Advances `(q, p)` by one conformal Leapfrog step using a custom 1D kinetic derivative $\partial_p H(p)$.
    #[verified_engine::verified]
    pub fn step_with_kinetic<G, K>(
        &self,
        state: &mut PhasePoint2D,
        grad_fn: &mut G,
        kinetic_deriv: &mut K,
        dt: f64,
    ) -> Result<(), EvolutionError>
    where
        G: FnMut(Vector2<f64>) -> Vector2<f64>,
        K: FnMut(f64) -> f64,
    {
        if !state.is_finite() || !dt.is_finite() || dt <= 0.0 || !self.mass.is_finite() {
            return Err(EvolutionError::InvalidState);
        }
        let h = clamp_dt(dt);
        let gamma = clamp_damping(self.damping);
        let damp = if gamma > 0.0 {
            (-0.5 * gamma * h).exp()
        } else {
            1.0
        };

        let g0 = grad_fn(state.q);
        if !g0.x.is_finite() || !g0.y.is_finite() {
            return Err(EvolutionError::InvalidState);
        }
        let p_damped = state.p * damp;
        let p_half = p_damped - g0 * (0.5 * h);
        let vx = kinetic_deriv(p_half.x);
        let vy = kinetic_deriv(p_half.y);
        let q_next = state.q + Vector2::new(vx, vy) * h;
        let g1 = grad_fn(q_next);
        if !g1.x.is_finite() || !g1.y.is_finite() {
            return Err(EvolutionError::InvalidState);
        }
        let p_next = (p_half - g1 * (0.5 * h)) * damp;
        if !q_next.x.is_finite()
            || !q_next.y.is_finite()
            || !p_next.x.is_finite()
            || !p_next.y.is_finite()
        {
            return Err(EvolutionError::InvalidState);
        }
        state.q = q_next;
        state.p = p_next;
        Ok(())
    }
}

impl<G> EvolutionEngine<PhasePoint2D, G> for HamiltonianLeapfrog2D
where
    G: FnMut(Vector2<f64>) -> Vector2<f64>,
{
    #[verified_engine::verified]
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut PhasePoint2D,
        aux: &mut G,
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let inv_m = 1.0 / clamp_mass(self.mass);
        let mut kinetic_deriv = |p_axis: f64| p_axis * inv_m;
        self.step_with_kinetic(state, aux, &mut kinetic_deriv, dt)
    }
}

impl EvolutionEngine<PhasePoint2D, Vector2<f64>> for HamiltonianLeapfrog2D {
    #[verified_engine::verified]
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut PhasePoint2D,
        aux: &mut Vector2<f64>,
        _rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        if !state.is_finite()
            || !aux.x.is_finite()
            || !aux.y.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
        {
            return Err(EvolutionError::InvalidState);
        }
        let m = clamp_mass(self.mass);
        let gamma = clamp_damping(self.damping);
        let h = clamp_dt(dt);
        let damp = if gamma > 0.0 { (-gamma * h).exp() } else { 1.0 };
        let p_next = state.p * damp - *aux * h;
        let q_next = state.q + p_next * (h / m);
        state.q = q_next;
        state.p = p_next;
        Ok(())
    }
}

/// Zero-allocation 2D Langevin BAOAB stochastic momentum integrator.
///
/// Splits the underdamped Langevin SDE $dq = \partial_p H(p)\,dt$,
/// $dp = -\nabla V(q)\,dt - \gamma p\,dt + \sqrt{2 m \gamma k_B T}\,dW_t$ into
/// half-kick (B), half-drift (A), exact Ornstein-Uhlenbeck thermostat (O), half-drift (A),
/// and half-kick (B).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LangevinBaoab2D {
    /// Inertial mass $m \in [0.05, 50.0]$.
    pub mass: f64,
    /// Friction / damping rate $\gamma \in [0.0, 20.0]$.
    pub damping: f64,
    /// Thermal bath energy $k_B T \in [0.0, 50.0]$.
    pub temperature: f64,
}

impl Default for LangevinBaoab2D {
    #[verified_engine::verified]
    fn default() -> Self {
        Self::new(1.0, 1.0, 1.0)
    }
}

impl LangevinBaoab2D {
    /// Creates a new 2D BAOAB Langevin integrator with clamped physical parameters.
    #[verified_engine::verified]
    pub fn new(mass: f64, damping: f64, temperature: f64) -> Self {
        Self {
            mass: clamp_mass(mass),
            damping: clamp_damping(damping),
            temperature: clamp_temperature(temperature),
        }
    }

    /// Advances `(q, p)` by one BAOAB step using a custom 1D kinetic derivative $\partial_p H(p)$.
    #[verified_engine::verified]
    pub fn step_with_kinetic<R, G, K>(
        &self,
        state: &mut PhasePoint2D,
        grad_fn: &mut G,
        kinetic_deriv: &mut K,
        rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError>
    where
        R: RngCore + ?Sized,
        G: FnMut(Vector2<f64>) -> Vector2<f64>,
        K: FnMut(f64) -> f64,
    {
        if !state.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || !self.mass.is_finite()
            || !self.damping.is_finite()
            || !self.temperature.is_finite()
        {
            return Err(EvolutionError::InvalidState);
        }
        let m = clamp_mass(self.mass);
        let gamma = clamp_damping(self.damping);
        let kbt = clamp_temperature(self.temperature);
        let h = clamp_dt(dt);

        let g0 = grad_fn(state.q);
        if !g0.x.is_finite() || !g0.y.is_finite() {
            return Err(EvolutionError::InvalidState);
        }
        let p_b = state.p - g0 * (0.5 * h);
        let v_b = Vector2::new(kinetic_deriv(p_b.x), kinetic_deriv(p_b.y));
        let q_a = state.q + v_b * (0.5 * h);

        let c1 = (-gamma * h).exp();
        let p_o = if kbt > 0.0 && gamma > 0.0 {
            let normal = Normal::new(0.0, 1.0).map_err(|_| EvolutionError::ConversionError)?;
            let sigma = (m * kbt * (1.0 - c1 * c1)).max(0.0).sqrt();
            let xi = Vector2::new(normal.sample(rng), normal.sample(rng));
            p_b * c1 + xi * sigma
        } else {
            p_b * c1
        };

        let v_o = Vector2::new(kinetic_deriv(p_o.x), kinetic_deriv(p_o.y));
        let q_next = q_a + v_o * (0.5 * h);
        let g1 = grad_fn(q_next);
        if !g1.x.is_finite() || !g1.y.is_finite() {
            return Err(EvolutionError::InvalidState);
        }
        let p_next = p_o - g1 * (0.5 * h);
        if !q_next.x.is_finite()
            || !q_next.y.is_finite()
            || !p_next.x.is_finite()
            || !p_next.y.is_finite()
        {
            return Err(EvolutionError::InvalidState);
        }
        state.q = q_next;
        state.p = p_next;
        Ok(())
    }
}

impl<G> EvolutionEngine<PhasePoint2D, G> for LangevinBaoab2D
where
    G: FnMut(Vector2<f64>) -> Vector2<f64>,
{
    #[verified_engine::verified]
    fn step<R: RngCore + ?Sized>(
        &mut self,
        state: &mut PhasePoint2D,
        aux: &mut G,
        rng: &mut R,
        dt: f64,
    ) -> Result<(), EvolutionError> {
        let inv_m = 1.0 / clamp_mass(self.mass);
        let mut kinetic_deriv = |p_axis: f64| p_axis * inv_m;
        self.step_with_kinetic(state, aux, &mut kinetic_deriv, rng, dt)
    }
}
