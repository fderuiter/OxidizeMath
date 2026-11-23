//! Lumped two-dimensional ordinary differential equation model for normal and cannibalistic cohorts.

// dN/dt = beta_N(N, C) * N + beta_C(N, C) * C - K(N) * N - phi(N, C) - mu_N(N, C) * N
// dC/dt = K(N) * N - mu_C(N, C) * C

use super::death_rate::sanitize_density;
use nalgebra::Vector2;
use pure_math::pure_math::analysis::ode::{OdeSystem, RungeKutta4, Solver};

/// Parameters for the Cannibalism ODE system.
///
/// Encapsulates the coefficients for the 2D population model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CannibalismModel {
    /// Birth rate of normal individuals ($\beta_N$)
    pub beta_n: f64,
    /// Birth rate contribution from cannibalistic individuals ($\beta_C$)
    pub beta_c: f64,
    /// Rate at which normal individuals transition to cannibals ($k_N$)
    pub k_n: f64,
    /// Constant loss term for normal individuals due to cannibalism ($\phi$)
    pub phi_n_c: f64,
    /// Mortality rate of normal individuals ($\mu_N$)
    pub mu_n: f64,
    /// Mortality rate of cannibalistic individuals ($\mu_C$)
    pub mu_c: f64,
}

impl CannibalismModel {
    /// Creates a new `CannibalismModel` with the specified parameters.
    #[verified_engine::verified]
    pub fn new(beta_n: f64, beta_c: f64, k_n: f64, phi_n_c: f64, mu_n: f64, mu_c: f64) -> Self {
        Self {
            beta_n,
            beta_c,
            k_n,
            phi_n_c,
            mu_n,
            mu_c,
        }
    }

    /// Advances `(N(t), C(t))` by one 4th-order [`RungeKutta4`] step with non-negative population clamping.
    #[verified_engine::verified]
    pub fn step_rk4(&self, t: f64, state: &Vector2<f64>, dt: f64) -> Vector2<f64> {
        let mut clamped = Vector2::new(sanitize_density(state[0]), sanitize_density(state[1]));
        if !dt.is_finite() || dt <= 0.0 || (clamped[0] == 0.0 && clamped[1] == 0.0) {
            return clamped;
        }
        let mut rk4 = RungeKutta4::new(&clamped);
        rk4.step(self, t, &mut clamped, dt);
        Vector2::new(sanitize_density(clamped[0]), sanitize_density(clamped[1]))
    }
}

/// Implementation of the ODE System trait.
///
/// State vector components:
/// - index 0: Normal population ($N$)
/// - index 1: Cannibal population ($C$)
impl OdeSystem<Vector2<f64>> for CannibalismModel {
    #[verified_engine::verified]
    fn derivative(&self, _t: f64, state: &Vector2<f64>) -> Vector2<f64> {
        let n = state[0];
        let c = state[1];

        let dndt = self.beta_n * n + self.beta_c * c - self.k_n * n - self.phi_n_c - self.mu_n * n;
        let dcdt = self.k_n * n - self.mu_c * c;

        Vector2::new(dndt, dcdt)
    }
}
