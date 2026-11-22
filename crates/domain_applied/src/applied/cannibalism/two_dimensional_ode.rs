// Two-Dimensional ODE Model for Cannibalism

// dN/dt = beta_N(N, C) * N + beta_C(N, C) * C - K(N) * N - phi(N, C) - mu_N(N, C) * N
// dC/dt = K(N) * N - mu_C(N, C) * C

use nalgebra::Vector2;
use pure_math::pure_math::analysis::ode::OdeSystem;

/// Parameters for the Cannibalism ODE system.
///
/// Encapsulates the coefficients for the 2D population model.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
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
